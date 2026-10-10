# -*- coding: utf-8 -*-
"""损失函数：
  1) Multi-ACCDOA + 距离 的 ADPIT 置换不变损失（Shimada et al. ICASSP 2022；DCASE2024 基线加入距离）
  2) 教师-学生蒸馏（帧级）：学生 = 骨干(前端输出 FOA)，教师 = 骨干(理想 FOA)
  3) 辅助 FOA 重建损失（归一化复数 MSE，可限制频带）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ADPITLoss(nn.Module):
    """output: (B,T,3,4,C)   target: (B,T,6,5,C) [act,x,y,z,dist]"""

    def __init__(self, dist_weight=1.0):
        super().__init__()
        self.dw = dist_weight

    def _err(self, out, tgt):  # (B,T,3,4,C) -> (B,T,C)
        e = (out - tgt) ** 2
        return e[:, :, :, :3].mean((2, 3)) * 0.75 + self.dw * e[:, :, :, 3].mean(2) * 0.25

    def forward(self, out, target):
        act = target[:, :, :, 0:1]
        vec = act * target[:, :, :, 1:5]  # (B,T,6,4,C)
        A0, B0, B1, C0, C1, C2 = [vec[:, :, i] for i in range(6)]
        st = lambda a, b, c: torch.stack([a, b, c], 2)
        tA = [st(A0, A0, A0)]
        tB = [st(B0, B0, B1), st(B0, B1, B0), st(B0, B1, B1), st(B1, B0, B0), st(B1, B0, B1), st(B1, B1, B0)]
        tC = [st(C0, C1, C2), st(C0, C2, C1), st(C1, C0, C2), st(C1, C2, C0), st(C2, C0, C1), st(C2, C1, C0)]
        padA, padB, padC = tB[0] + tC[0], tA[0] + tC[0], tA[0] + tB[0]
        losses = [self._err(out, t + padA) for t in tA] + [self._err(out, t + padB) for t in tB] + \
                 [self._err(out, t + padC) for t in tC]
        return torch.stack(losses, 0).min(0).values.mean()


_PERMS3 = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def distill_loss(student, teacher, emb_s=None, emb_t=None, emb_weight=0.0, conf_thr=0.0):
    """帧级置换不变蒸馏（frame-level tPIT KD，参考 arXiv 2609.18040 §3.3）。

    student/teacher: (B,T,3,4,C) multi-ACCDOA(+dist)。多轨道输出的轨道顺序本身是任意的，
    因此对每一帧在 3! 个教师轨道排列中取误差最小者。
    conf_thr>0 时：教师某轨道某类的 ACCDOA 模长 < conf_thr 的位置，只蒸馏“不活跃”（目标置 0），
    相当于原文“只在教师置信时蒸馏 DOA”的简化版（mACCDOA 没有独立的 SED logit，所以无法逐字复现 BCE 部分）。
    """
    s, t = student.float(), teacher.detach().float()
    if conf_thr > 0:
        conf = t[:, :, :, :3].norm(dim=3, keepdim=True)
        t = t * (conf >= conf_thr).float()
    errs = [((s - t[:, :, list(p)]) ** 2).mean((2, 3, 4)) for p in _PERMS3]  # 每个 (B,T)
    l = torch.stack(errs, 0).min(0).values.mean()
    if emb_weight > 0 and emb_s is not None and emb_t is not None and emb_s.shape == emb_t.shape:
        l = l + emb_weight * (1 - F.cosine_similarity(emb_s.float(), emb_t.detach().float(), dim=-1)).mean()
    return l


def foa_recon_loss(p_re, p_im, t_re, t_im, f_lo_bin=0, f_hi_bin=None):
    """逐样本归一化的复数 MSE（对整体电平不敏感）。输入 (B,4,T,F)。"""
    sl = slice(f_lo_bin, f_hi_bin)
    p_re, p_im, t_re, t_im = [x[..., sl].float() for x in (p_re, p_im, t_re, t_im)]
    err = ((p_re - t_re) ** 2 + (p_im - t_im) ** 2).sum((1, 2, 3))
    ref = (t_re ** 2 + t_im ** 2).sum((1, 2, 3)) + 1e-6
    return (err / ref).mean()
