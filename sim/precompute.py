# -*- coding: utf-8 -*-
"""离线预处理：把仿真/真实数据目录打包成单个 HDF5（多进程读取+特征计算）。

每个场景一个 group：
  mic      (M, N) float16  阵列波形
  mic_pos  (M, 3) float32  麦克风坐标（相对阵列中心, m）
  foa      (4, N) float16  理想/参考 FOA 波形（可选，没有则不存）
  foa_feat (7, T, 64) float16  FOA 的 logmel+IV 特征（教师输入，可选）
  labels   (K, 6)  float32  DCASE 行 [frame, class, track, azi, ele, dist_cm]
  attrs: array_type, n_mics, fs, n_frames
阵列 STFT 不预存（体积太大，且在 GPU 上计算很便宜），训练时在线计算。
"""
import argparse
import json
import os
from multiprocessing import Pool

import h5py
import numpy as np
import soundfile as sf
import torch

from seld.features import LogmelIV, stft_reim
from seld.labels import read_dcase_csv


def _load(args):
    root, sid, feat_cfg = args
    mic, fs = sf.read(os.path.join(root, "mic", sid + ".wav"), dtype="float32", always_2d=True)
    with open(os.path.join(root, "geometry", sid + ".json")) as f:
        geo = json.load(f)
    out = dict(sid=sid, mic=mic.T, mic_pos=np.array(geo["mic_coords"], np.float32), fs=fs,
               array_type=geo.get("array_type", "unknown"), labels=read_dcase_csv(os.path.join(root, "metadata", sid + ".csv")))
    fp = os.path.join(root, "foa", sid + ".wav")
    if os.path.exists(fp):
        foa, _ = sf.read(fp, dtype="float32", always_2d=True)
        out["foa"] = foa.T
        if feat_cfg["foa_feat"]:
            torch.set_num_threads(1)
            ext = LogmelIV(fs, feat_cfg["n_fft"], feat_cfg["n_mels"], feat_cfg["iv"])
            re, im = stft_reim(torch.from_numpy(foa.T[None]), feat_cfg["n_fft"], feat_cfg["hop"])
            out["foa_feat"] = ext(re, im)[0].numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="包含 mic/ foa/ geometry/ metadata/ 的目录")
    ap.add_argument("--out", required=True, help="输出 .h5")
    ap.add_argument("--n-fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=240)
    ap.add_argument("--n-mels", type=int, default=64)
    ap.add_argument("--no-foa-feat", action="store_true")
    ap.add_argument("--iv-xyz-index", default="1,2,3", help="须与配置 audio.iv_xyz_index 一致")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    sids = sorted(f[:-4] for f in os.listdir(os.path.join(a.root, "mic")) if f.endswith(".wav"))
    fc = dict(n_fft=a.n_fft, hop=a.hop, n_mels=a.n_mels, foa_feat=not a.no_foa_feat, iv=tuple(int(x) for x in a.iv_xyz_index.split(",")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with h5py.File(a.out, "w") as h5, Pool(max(1, a.workers)) as pool:
        for d in pool.imap(_load, [(a.root, s, fc) for s in sids], chunksize=2):
            g = h5.create_group(d["sid"])
            g.create_dataset("mic", data=d["mic"].astype(np.float16), compression="lzf")
            g.create_dataset("mic_pos", data=d["mic_pos"])
            g.create_dataset("labels", data=d["labels"])
            if "foa" in d:
                g.create_dataset("foa", data=d["foa"].astype(np.float16), compression="lzf")
            if "foa_feat" in d:
                g.create_dataset("foa_feat", data=d["foa_feat"].astype(np.float16), compression="lzf")
            g.attrs["array_type"] = d["array_type"]
            g.attrs["n_mics"] = d["mic"].shape[0]
            g.attrs["fs"] = d["fs"]
            g.attrs["n_samples"] = d["mic"].shape[1]
        h5.attrs["n_fft"], h5.attrs["hop"], h5.attrs["n_mels"] = a.n_fft, a.hop, a.n_mels
    print(f"wrote {len(sids)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
