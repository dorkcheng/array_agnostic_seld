# -*- coding: utf-8 -*-
"""把 STARSS23 (DCASE2024 Task3, 含距离标签) 整理为与仿真相同的目录结构，然后可直接用 sim/precompute.py。

STARSS23 目录：
  mic_dev/dev-train-sony/*.wav, mic_dev/dev-test-tau/*.wav ...  (4ch, 24kHz)
  foa_dev/...  同名 FOA 录音（作为教师输入）
  metadata_dev/... 同名 csv: frame,class,source,azi,ele,dist(cm)
MIC 格式坐标：四面体，半径 4.2cm，(azi,ele) = (45,35),(-45,-35),(135,-35),(-135,35) （见 seld/utils.py）。
注意：STARSS23 的 MIC 是安装在刚性球（em32）上的，并非自由场 —— 平面波 LS 编码在高频有偏差，
残差网络正好学习这部分；如需更准确，可在 ls_foa_encoder 中加入刚性球散射模型（TODO）。

用法：python -m sim.prepare_starss23 --starss /kaggle/input/starss23 --out data/starss23_train --split dev-train
"""
import argparse
import glob
import json
import os

from seld.utils import starss23_mic_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starss", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="dev-train", help="dev-train 或 dev-test（匹配子目录前缀）")
    a = ap.parse_args()
    for d in ["mic", "foa", "geometry", "metadata"]:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    coords = starss23_mic_coords().tolist()
    n = 0
    for wav in sorted(glob.glob(os.path.join(a.starss, "mic_dev", a.split + "*", "*.wav"))):
        sub, fn = os.path.basename(os.path.dirname(wav)), os.path.basename(wav)
        sid = fn[:-4]
        meta = os.path.join(a.starss, "metadata_dev", sub, sid + ".csv")
        foa = os.path.join(a.starss, "foa_dev", sub, fn)
        if not os.path.exists(meta):
            continue
        for src, dst in [(wav, f"mic/{sid}.wav"), (meta, f"metadata/{sid}.csv"), (foa, f"foa/{sid}.wav")]:
            dst = os.path.join(a.out, dst)
            if os.path.exists(src) and not os.path.exists(dst):
                os.symlink(os.path.abspath(src), dst)
        with open(os.path.join(a.out, "geometry", sid + ".json"), "w") as f:
            json.dump(dict(mic_coords=coords, array_type="starss23_tetra", n_mics=4), f)
        n += 1
    print(f"prepared {n} STARSS23 recordings -> {a.out}")


if __name__ == "__main__":
    main()
