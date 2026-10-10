# -*- coding: utf-8 -*-
"""解析最小二乘 FOA 编码器（思路来自 arXiv 2609.18040 的 analytic LS 部分）。

平面波模型：方向 u 的单位幅度平面波在麦克风 m（相对阵列中心坐标 r_m）处的 STFT 为
    a_m(f, u) = exp(+j 2π f/c · u·r_m)        （离声源近的麦克风先到达）
理想 FOA（ACN/SN3D）对该平面波的响应为 y(u) = [1, u_y, u_z, u_x]。
在方向网格 {u_q} 上求每个频点的编码矩阵 E(f) ∈ C^{4×M}：
    E(f) = argmin_E Σ_q ||E a(f,u_q) - y(u_q)||² + λ(f) ||E||²
         = Y A^H (A A^H + λ I)^{-1}
λ(f) = reg · tr(A A^H)/M（Tikhonov 相对正则）。低频处偶极子分量病态 → 被正则压低，
高频超过空间混叠频率后误差变大 —— 这两部分正是残差网络要修正的。

所有计算按 batch 进行，掩码掉的麦克风对应的 A 行为 0，从而 E 对应列为 0；
因此同一模型支持 2~8 个任意排列的麦克风。复数仅在求 E 时使用（float32/complex64，
不进 autocast）；与 STFT 相乘时用实/虚分离的实数运算，便于 ONNX 导出。
"""
import math

import numpy as np
import torch
import torch.nn as nn

from seld.utils import SPEED_OF_SOUND, fibonacci_sphere


class LSFOAEncoder(nn.Module):
    def __init__(self, fs=24000, n_fft=1024, n_dirs=240, reg=1e-2, c=SPEED_OF_SOUND):
        super().__init__()
        dirs = fibonacci_sphere(n_dirs).astype(np.float32)
        Y = np.stack([np.ones(n_dirs), dirs[:, 1], dirs[:, 2], dirs[:, 0]], 0).astype(np.float32)  # (4,Q)
        self.register_buffer("dirs", torch.from_numpy(dirs), persistent=False)
        self.register_buffer("Y", torch.from_numpy(Y), persistent=False)
        self.register_buffer("freqs", torch.linspace(0, fs / 2, n_fft // 2 + 1), persistent=False)
        self.reg, self.c = reg, c

    @torch.no_grad()
    def encoding_matrix(self, mic_pos, mic_mask):
        """mic_pos (B,M,3) m, mic_mask (B,M) bool -> E_re, E_im (B,F,4,M) float32"""
        with torch.autocast(device_type=mic_pos.device.type, enabled=False):
            pos = mic_pos.float()
            B, M, _ = pos.shape
            # 用有效麦克风的质心作为参考点（FOA 在质心处）
            w = mic_mask.float()[..., None]
            pos = pos - (pos * w).sum(1, keepdim=True) / w.sum(1, keepdim=True).clamp(min=1)
            proj = torch.einsum("bmd,qd->bmq", pos, self.dirs)  # (B,M,Q)
            k = 2 * math.pi * self.freqs / self.c  # (F,)
            phase = k[None, :, None, None] * proj[:, None]  # (B,F,M,Q)
            A = torch.polar(torch.ones_like(phase), phase) * mic_mask[:, None, :, None].to(phase.dtype)
            G = A @ A.conj().transpose(-1, -2)  # (B,F,M,M)
            n_valid = mic_mask.float().sum(1).clamp(min=1)[:, None, None, None]
            tr = torch.diagonal(G, dim1=-2, dim2=-1).real.sum(-1, keepdim=True)[..., None]
            lam = self.reg * tr / n_valid + 1e-6
            eye = torch.eye(M, device=pos.device, dtype=torch.float32)
            G = G + (lam * eye).to(G.dtype)
            rhs = A @ self.Y.T.to(A.dtype)  # (B,F,M,4)
            EH = torch.linalg.solve(G, rhs)  # (B,F,M,4) = E^H
            E = EH.conj().transpose(-1, -2)  # (B,F,4,M)
            E = E * mic_mask[:, None, None, :].to(E.dtype)
            return E.real.contiguous(), E.imag.contiguous()

    @staticmethod
    def apply(E_re, E_im, X_re, X_im):
        """E (B,F,4,M) 或 (B,4,M,T,F) 形式的逐时频权重; X (B,M,T,F) -> FOA (B,4,T,F)"""
        if E_re.dim() == 4:
            Er = E_re.permute(0, 2, 3, 1)[:, :, :, None, :]  # (B,4,M,1,F)
            Ei = E_im.permute(0, 2, 3, 1)[:, :, :, None, :]
        else:
            Er, Ei = E_re, E_im
        Xr, Xi = X_re[:, None], X_im[:, None]
        return (Er * Xr - Ei * Xi).sum(2), (Er * Xi + Ei * Xr).sum(2)
