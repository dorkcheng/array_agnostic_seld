# -*- coding: utf-8 -*-
"""DCASE 风格 SELD 指标（帧级近似实现，非官方代码；官方实现见 DCASE2024 Task3 baseline 的 SELD_evaluation_metrics.py）。

  - F-score (location-aware, 20°)：TP 要求类别正确、角度误差 ≤ 20°（且可选相对距离误差 ≤ 阈值）
  - DOA error (度)：类别正确且匹配上的预测-真值对的平均角度误差
  - Relative distance error：|d_pred - d_gt| / d_gt 的平均
  - 每帧每类用匈牙利算法匹配（多个同类声源）。宏平均到类别。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .utils import angular_distance_deg


def decode_multi_accdoa(out, thr=0.5, merge_deg=15.0, dist_scale=5.0):
    """out: (T,3,4,C) numpy -> list[T] of list of (cls, xyz(3), dist_m)"""
    T, K, _, C = out.shape
    res = [[] for _ in range(T)]
    for t in range(T):
        for c in range(C):
            cands = []
            for k in range(K):
                v = out[t, k, :3, c]
                if np.linalg.norm(v) > thr:
                    cands.append((v / np.linalg.norm(v), out[t, k, 3, c] * dist_scale))
            merged = []
            for u, d in cands:  # 合并方向相近的重复轨道
                for m in merged:
                    if angular_distance_deg(u, m[0]) < merge_deg:
                        m[0] = (m[0] + u) / np.linalg.norm(m[0] + u)
                        m[1] = (m[1] + d) / 2
                        break
                else:
                    merged.append([u, d])
            res[t] += [(c, m[0], m[1]) for m in merged]
    return res


def target_to_list(target, dist_scale=5.0):
    """target (T,6,5,C) -> list[T] of (cls, xyz, dist_m)"""
    T, _, _, C = target.shape
    res = [[] for _ in range(T)]
    for t in range(T):
        for s in range(6):
            for c in np.nonzero(target[t, s, 0] > 0.5)[0]:
                res[t].append((int(c), target[t, s, 1:4, c], target[t, s, 4, c] * dist_scale))
    return res


class SELDMetrics:
    def __init__(self, n_classes, doa_thr=20.0, dist_thr=None):
        self.C, self.doa_thr, self.dist_thr = n_classes, doa_thr, dist_thr
        self.tp = np.zeros(n_classes)
        self.fp = np.zeros(n_classes)
        self.fn = np.zeros(n_classes)
        self.doa_sum = np.zeros(n_classes)
        self.rde_sum = np.zeros(n_classes)
        self.n_match = np.zeros(n_classes)

    def update(self, pred_list, gt_list):
        for P, G in zip(pred_list, gt_list):
            for c in range(self.C):
                p = [x for x in P if x[0] == c]
                g = [x for x in G if x[0] == c]
                if not p and not g:
                    continue
                if p and g:
                    cost = np.array([[angular_distance_deg(a[1], b[1]) for b in g] for a in p])
                    ri, ci = linear_sum_assignment(cost)
                    ok = 0
                    for i, j in zip(ri, ci):
                        ang = cost[i, j]
                        rde = abs(p[i][2] - g[j][2]) / max(g[j][2], 1e-3)
                        self.doa_sum[c] += ang
                        self.rde_sum[c] += rde
                        self.n_match[c] += 1
                        if ang <= self.doa_thr and (self.dist_thr is None or rde <= self.dist_thr):
                            ok += 1
                    self.tp[c] += ok
                    self.fp[c] += len(p) - ok
                    self.fn[c] += len(g) - ok
                else:
                    self.fp[c] += len(p)
                    self.fn[c] += len(g)

    def compute(self):
        f = 2 * self.tp / np.maximum(2 * self.tp + self.fp + self.fn, 1e-9)
        valid = (self.tp + self.fn) > 0  # 只对测试集中出现过的类做宏平均
        doa = self.doa_sum / np.maximum(self.n_match, 1)
        rde = self.rde_sum / np.maximum(self.n_match, 1)
        vm = valid & (self.n_match > 0)
        F = float(f[valid].mean()) if valid.any() else 0.0
        DOA = float(doa[vm].mean()) if vm.any() else 180.0
        RDE = float(rde[vm].mean()) if vm.any() else 1.0
        return dict(F20=F, DOAE=DOA, RDE=RDE, SELD=float((1 - F + DOA / 180 + min(RDE, 1)) / 3))
