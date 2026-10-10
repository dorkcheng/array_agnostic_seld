# -*- coding: utf-8 -*-
"""生成用于测试的“伪声音事件”单声道片段（合成音调/噪声），并写出 csv 映射。

用法：python -m sim.dummy_sources --out data/dummy_sources --n-per-class 10
csv 格式（与 FSD50K 子集映射一致）：filename,class_idx,class_name
"""
import argparse
import csv
import os

import numpy as np
import soundfile as sf

DUMMY_CLASSES = ["tone_low", "tone_high", "chirp", "white_noise_burst", "am_tone"]


def synth(cls, dur, fs, rng):
    t = np.arange(int(dur * fs)) / fs
    if cls == "tone_low":
        x = np.sin(2 * np.pi * rng.uniform(200, 400) * t) + 0.3 * np.sin(2 * np.pi * rng.uniform(600, 900) * t)
    elif cls == "tone_high":
        x = np.sin(2 * np.pi * rng.uniform(1500, 3000) * t)
    elif cls == "chirp":
        f0, f1 = rng.uniform(300, 800), rng.uniform(2000, 6000)
        x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))
    elif cls == "white_noise_burst":
        x = rng.standard_normal(len(t))
    elif cls == "am_tone":
        x = np.sin(2 * np.pi * rng.uniform(500, 1200) * t) * (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 8) * t))
    else:
        raise ValueError(cls)
    env = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.02)  # 20ms 淡入淡出
    x = x * env
    return (0.5 * x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-per-class", type=int, default=10)
    ap.add_argument("--fs", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for ci, c in enumerate(DUMMY_CLASSES):
        for k in range(a.n_per_class):
            fn = f"{c}_{k:03d}.wav"
            sf.write(os.path.join(a.out, fn), synth(c, rng.uniform(0.5, 2.0), a.fs, rng), a.fs)
            rows.append([fn, ci, c])
    with open(os.path.join(a.out, "sources.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "class_idx", "class_name"])
        w.writerows(rows)
    print(f"wrote {len(rows)} clips, {len(DUMMY_CLASSES)} classes -> {a.out}/sources.csv")


if __name__ == "__main__":
    main()
