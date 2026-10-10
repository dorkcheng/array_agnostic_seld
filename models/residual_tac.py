# -*- coding: utf-8 -*-
"""神经残差 FOA 修正前端：对麦克风数量与顺序不变（permutation/count invariant）。

结构：
  1) 每个麦克风通道共享的 CNN 编码器（权重共享 → 与通道顺序无关）
     输入特征（每通道 9 维/时频点）:
       log|X_m|², 与 LS-W 的相位差 cos/sin, 以及频率相关的相对位置编码 cos/sin(2πf/c · r_m·e_xyz)
  2) 相对麦克风位置编码 rMPE（参考 AGG-RL 的 rMPE 思路）：r_m 相对有效麦克风质心，
     经 Fourier 特征 + MLP 得到每通道嵌入，加到通道特征上
  3) TAC 块（Transform-Average-Concatenate, Luo et al. arXiv 1910.14104），带掩码的跨通道平均
  4) 每通道输出逐时频的复数编码权重修正 ΔE_m(t,f) ∈ C^4，最终
       FOA(t,f) = Σ_m (E_LS,m(f) + ΔE_m(t,f)) · X_m(t,f)
     输出层零初始化 → 训练初始时等价于纯 LS 编码。
频率轴在内部以 freq_stride 下采样以节省显存，输出时线性插值回原频率分辨率。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from seld.utils import SPEED_OF_SOUND


def masked_mean(x, mask, dim=1):
    """x (B,M,...) mask (B,M)"""
    m = mask.float().view(mask.shape + (1,) * (x.dim() - 2))
    return (x * m).sum(dim) / m.sum(dim).clamp(min=1)


class RelMicPosEnc(nn.Module):
    def __init__(self, dim, n_freq=8, max_radius=0.15):
        super().__init__()
        self.register_buffer("bands", (2 ** torch.arange(n_freq).float()) * math.pi / max_radius, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(3 * 2 * n_freq + 4, dim), nn.PReLU(), nn.Linear(dim, dim))

    def forward(self, rel):  # rel (B,M,3)
        ang = rel[..., None] * self.bands  # (B,M,3,K)
        feat = torch.cat([torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2), rel * 10, rel.norm(dim=-1, keepdim=True) * 10], -1)
        return self.mlp(feat)


class TACBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.ctx = nn.Conv2d(h, h, 3, padding=1, groups=h)  # 每通道局部时频上下文
        self.t_in = nn.Sequential(nn.Conv2d(h, h, 1), nn.PReLU())
        self.t_avg = nn.Sequential(nn.Conv2d(h, h, 1), nn.PReLU())
        self.t_out = nn.Sequential(nn.Conv2d(2 * h, h, 1), nn.PReLU())
        self.norm = nn.GroupNorm(1, h)

    def forward(self, x, mask):  # x (B,M,H,T,F)
        B, M, H, T, Fq = x.shape
        y = x.reshape(B * M, H, T, Fq)
        y = self.t_in(self.ctx(y))
        avg = masked_mean(y.view(B, M, H, T, Fq), mask)  # (B,H,T,F)
        avg = self.t_avg(avg)[:, None].expand(B, M, H, T, Fq).reshape(B * M, H, T, Fq)
        y = self.t_out(torch.cat([y, avg], 1))
        y = self.norm(y).view(B, M, H, T, Fq)
        return (x + y) * mask.float()[:, :, None, None, None]


class ResidualTACFrontend(nn.Module):
    def __init__(self, fs=24000, n_fft=1024, hidden=32, n_tac=3, freq_stride=4, c=SPEED_OF_SOUND):
        super().__init__()
        self.freq_stride, self.c = freq_stride, c
        self.register_buffer("freqs", torch.linspace(0, fs / 2, n_fft // 2 + 1), persistent=False)
        cin = 9
        self.enc = nn.Sequential(
            nn.Conv2d(cin, hidden, 3, padding=1), nn.PReLU(),
            nn.Conv2d(hidden, hidden, (3, freq_stride + 2), stride=(1, freq_stride), padding=(1, 1)), nn.PReLU(),
            nn.GroupNorm(1, hidden))
        self.rmpe = RelMicPosEnc(hidden)
        self.tac = nn.ModuleList([TACBlock(hidden) for _ in range(n_tac)])
        self.head = nn.Conv2d(hidden, 8, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, X_re, X_im, W_re, W_im, mic_pos, mask):
        """X (B,M,T,F); W = LS 编码得到的 FOA W 通道 (B,T,F); 返回 ΔE_re, ΔE_im (B,4,M,T,F)"""
        B, M, T, Fq = X_re.shape
        maskf = mask.float()
        rel = mic_pos - (mic_pos * maskf[..., None]).sum(1, keepdim=True) / maskf.sum(1, keepdim=True)[..., None].clamp(min=1)
        rel = rel * maskf[..., None]
        pw = X_re ** 2 + X_im ** 2
        logp = torch.log(pw + 1e-8)
        logp = logp - masked_mean(logp, mask).mean(dim=(-1, -2), keepdim=True)[:, None]
        cr = X_re * W_re[:, None] + X_im * W_im[:, None]  # X · conj(W)
        ci = X_im * W_re[:, None] - X_re * W_im[:, None]
        nrm = torch.sqrt(cr ** 2 + ci ** 2) + 1e-8
        k = 2 * math.pi * self.freqs / self.c  # (F,)
        ph = rel[..., None] * k  # (B,M,3,F)
        geo = torch.cat([torch.cos(ph), torch.sin(ph)], 2)[:, :, :, None, :].expand(B, M, 6, T, Fq)
        feat = torch.cat([logp[:, :, None], (cr / nrm)[:, :, None], (ci / nrm)[:, :, None], geo], 2)
        h = self.enc(feat.reshape(B * M, 9, T, Fq))
        H, Fd = h.shape[1], h.shape[-1]
        h = h.view(B, M, H, T, Fd) + self.rmpe(rel)[:, :, :, None, None]
        h = h * maskf[:, :, None, None, None]
        for blk in self.tac:
            h = blk(h, mask)
        d = self.head(h.reshape(B * M, H, T, Fd))
        d = F.interpolate(d, size=(T, Fq), mode="bilinear", align_corners=False)
        d = d.view(B, M, 8, T, Fq) * maskf[:, :, None, None, None]
        dE_re = d[:, :, :4].transpose(1, 2)
        dE_im = d[:, :, 4:].transpose(1, 2)
        return dE_re, dE_im
