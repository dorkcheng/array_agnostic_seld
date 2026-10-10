# -*- coding: utf-8 -*-
"""诊断脚本：比较 LS 编码 FOA 与理想 FOA（逐频带归一化误差），并检查理想 FOA 的强度向量方向是否指向标注方向。
用法：python scripts/check_ls_foa.py --h5 data/h5/train.h5 [--n 5]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import h5py
import numpy as np
import torch

from models.ls_foa_encoder import LSFOAEncoder
from seld.features import stft_reim
from seld.utils import angular_distance_deg, sph2cart


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    enc = LSFOAEncoder()
    bands = [(100, 500), (500, 1500), (1500, 4000), (4000, 8000)]
    f = np.linspace(0, 12000, 513)
    with h5py.File(a.h5) as h5:
        for sid in sorted(h5.keys())[: a.n]:
            g = h5[sid]
            mic = torch.from_numpy(g["mic"][()].astype(np.float32))[None]
            foa = torch.from_numpy(g["foa"][()].astype(np.float32))[None]
            pos = torch.from_numpy(g["mic_pos"][()])[None]
            mask = torch.ones(1, pos.shape[1], dtype=torch.bool)
            Xr, Xi = stft_reim(mic)
            Fr, Fi = stft_reim(foa)
            Er, Ei = enc.encoding_matrix(pos, mask)
            Lr, Li = enc.apply(Er, Ei, Xr, Xi)
            msg = []
            for lo, hi in bands:
                sel = (f >= lo) & (f < hi)
                err = ((Lr - Fr) ** 2 + (Li - Fi) ** 2)[..., sel].sum((0, 2, 3))
                ref = (Fr ** 2 + Fi ** 2)[..., sel].sum((0, 2, 3)) + 1e-12
                msg.append(f"{lo}-{hi}Hz NMSE(W,Y,Z,X)=" + ",".join(f"{10*np.log10(e/r):.1f}" for e, r in zip(err.numpy(), ref.numpy())))
            # 理想 FOA 强度向量（全频带、事件第一帧前后）方向
            rows = g["labels"][()]
            doa_err = ""
            if len(rows):
                fr = int(rows[0, 0])
                t0, t1 = fr * 10, fr * 10 + 10
                ivx = (Fr[0, 0] * Fr[0, 3] + Fi[0, 0] * Fi[0, 3])[t0:t1, 10:200].sum()
                ivy = (Fr[0, 0] * Fr[0, 1] + Fi[0, 0] * Fi[0, 1])[t0:t1, 10:200].sum()
                ivz = (Fr[0, 0] * Fr[0, 2] + Fi[0, 0] * Fi[0, 2])[t0:t1, 10:200].sum()
                doa_err = f"IV-vs-label={angular_distance_deg(np.array([ivx, ivy, ivz]), sph2cart(rows[0,3], rows[0,4])):.1f}deg"
            print(sid, g.attrs["array_type"], f"M={pos.shape[1]}", doa_err, "|", " | ".join(msg))


if __name__ == "__main__":
    main()
