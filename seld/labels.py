# -*- coding: utf-8 -*-
"""标签：DCASE 风格 csv <-> 多 ACCDOA (ADPIT) 稠密目标。

csv 每行：frame(100ms), class, source/track id, azimuth(度), elevation(度), distance(cm)
（与 DCASE2024 Task3 STARSS23 metadata 格式一致，距离单位 cm）。

ADPIT 目标张量: (T, 6, 5, C)
  6 个“虚拟轨道”: A0 | B0 B1 | C0 C1 C2 —— 同类同时出现 1/2/3 个声源时分别使用 A/B/C 组
  5 = [activity, x, y, z, distance/dist_scale]
"""
import csv

import numpy as np

from .utils import sph2cart


def read_dcase_csv(path):
    rows = []
    with open(path, "r") as f:
        for r in csv.reader(f):
            if not r or not r[0].strip().lstrip("-").isdigit():
                continue
            fr, cl, tr = int(r[0]), int(r[1]), int(r[2])
            azi, ele = float(r[3]), float(r[4])
            dist = float(r[5]) if len(r) > 5 else 100.0  # 无距离时给 1m
            rows.append([fr, cl, tr, azi, ele, dist])
    return np.array(rows, dtype=np.float32).reshape(-1, 6)


def write_dcase_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([int(r[0]), int(r[1]), int(r[2]), round(float(r[3]), 2), round(float(r[4]), 2), round(float(r[5]), 1)])


def rows_to_adpit(rows, n_frames, n_classes, dist_scale=5.0):
    """rows: (K,6) [frame, class, track, azi, ele, dist_cm] -> (T,6,5,C) float32"""
    tgt = np.zeros((n_frames, 6, 5, n_classes), np.float32)
    if len(rows) == 0:
        return tgt
    by = {}
    for r in rows:
        fr, cl = int(r[0]), int(r[1])
        if 0 <= fr < n_frames and 0 <= cl < n_classes:
            by.setdefault((fr, cl), []).append(r)
    for (fr, cl), lst in by.items():
        lst = sorted(lst, key=lambda r: r[2])[:3]
        slots = {1: [0], 2: [1, 2], 3: [3, 4, 5]}[len(lst)]
        for s, r in zip(slots, lst):
            xyz = sph2cart(r[3], r[4], 1.0)
            tgt[fr, s, 0, cl] = 1.0
            tgt[fr, s, 1:4, cl] = xyz
            tgt[fr, s, 4, cl] = (r[5] / 100.0) / dist_scale
    return tgt
