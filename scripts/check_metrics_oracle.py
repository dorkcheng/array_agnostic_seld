# -*- coding: utf-8 -*-
"""单元检查：把真值转成“完美预测”喂给指标，应得到 F20=1, DOAE=0, RDE=0。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from torch.utils.data import DataLoader

from seld.dataset import SELDH5Dataset, collate_pad
from seld.engine import evaluate_predictions


def oracle(t):  # (T,6,5,C) -> (T,3,4,C)
    T, _, _, C = t.shape
    o = np.zeros((T, 3, 4, C), np.float32)
    for c in range(C):
        for fr in range(T):
            act = t[fr, :, 0, c]
            if act[0]:
                o[fr, :, :, c] = t[fr, 0, 1:, c]
            elif act[1]:
                o[fr, 0], o[fr, 1] = 0, 0
                o[fr, 0, :, c], o[fr, 1, :, c] = t[fr, 1, 1:, c], t[fr, 2, 1:, c]
            elif act[3]:
                for k in range(3):
                    o[fr, k, :, c] = t[fr, 3 + k, 1:, c]
    return o


h5 = sys.argv[1]
n_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
ds = SELDH5Dataset(h5, n_classes, 2.0, mode="eval", use_foa=False, use_foa_feat=False)
preds = {}
for b in DataLoader(ds, batch_size=1, collate_fn=collate_pad):
    t = b["target"][0].numpy()
    preds.setdefault(b["sid"][0], []).append(t)
preds = {k: (oracle(np.concatenate(v)), np.concatenate(v)) for k, v in preds.items()}
print("oracle:", evaluate_predictions(preds, n_classes))
