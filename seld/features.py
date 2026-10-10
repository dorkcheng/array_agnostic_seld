# -*- coding: utf-8 -*-
"""STFT / log-mel / 强度向量（IV）特征。全部使用实数张量（re, im 分开），便于 ONNX 导出。

与 PSELDNets 保持一致：sr=24k, nfft=1024, hop=240, 64 mel (slaney), f_min=20,
特征 = 4 通道 log-mel + 3 通道归一化强度向量（mel 投影），共 7 通道。
"""
import math

import numpy as np
import torch
import torch.nn as nn


def mel_filterbank(sr, n_fft, n_mels, f_min=20.0, f_max=None):
    """mel 滤波器组 (n_freq, n_mels)：HTK mel 刻度 + slaney 面积归一化，
    等价于 PSELDNets 使用的 torchaudio.transforms.MelScale(norm='slaney')（其默认 mel_scale='htk'）。"""
    f_max = f_max or sr / 2
    n_freq = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_freq)
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)
    m_pts = np.linspace(hz2mel(f_min), hz2mel(f_max), n_mels + 2)
    f_pts = mel2hz(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = np.maximum(0, np.minimum(down, up))
    enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
    fb = fb * enorm[None, :]
    return fb.astype(np.float32)


def stft_reim(wav, n_fft=1024, hop=240):
    """wav: (B, C, N) -> re, im: (B, C, T, F)。"""
    B, C, N = wav.shape
    win = torch.hann_window(n_fft, device=wav.device, dtype=torch.float32)
    X = torch.stft(wav.reshape(B * C, N).float(), n_fft, hop, n_fft, win, center=True, return_complex=True)
    X = X.transpose(1, 2).reshape(B, C, X.shape[-1], X.shape[-2])
    return X.real.contiguous(), X.imag.contiguous()


class LogmelIV(nn.Module):
    """输入 FOA STFT (re, im): (B,4,T,F)，通道顺序由 `foa_order` 指定。
    输出 (B,7,T,n_mels)：[4×logmel, IVx, IVy, IVz]。

    通道顺序：内部 FOA 统一为 ACN (W,Y,Z,X)，与 DCASE FOA 文件一致。PSELDNets 的 intensityvector()
    直接把通道 1/2/3 当作 IV 的三个分量（见其 src/utils/feature.py）。`iv_xyz_index` 指定 IV 三个分量
    取自哪些通道。默认 [1,2,3] = 与 PSELDNets 完全一致；我们在仿真 FOA 上实测：用官方
    mACCDOA-CNN14-Conformer 权重 + 原 170 类头，[1,2,3] 的 DOA 中位误差 31°，而“物理直观”的
    [3,1,2] 为 112°。因此使用预训练权重时务必保持 [1,2,3]。
    """

    def __init__(self, sr=24000, n_fft=1024, n_mels=64, iv_xyz_index=(1, 2, 3), logmel_order=(0, 1, 2, 3)):
        super().__init__()
        self.register_buffer("melW", torch.from_numpy(mel_filterbank(sr, n_fft, n_mels)), persistent=False)
        self.iv_idx = list(iv_xyz_index)
        self.lm_idx = list(logmel_order)

    def forward(self, re, im):
        # 注意：STFT 功率可能超过 fp16 上限，这里强制 fp32 计算
        with torch.autocast(device_type=re.device.type, enabled=False):
            return self._forward(re.float(), im.float())

    def _forward(self, re, im):
        power = re ** 2 + im ** 2
        mel = torch.matmul(power, self.melW)
        logmel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        logmel = logmel[:, self.lm_idx]
        pr, pi = re[:, 0], im[:, 0]
        iv = [pr * re[:, c] + pi * im[:, c] for c in self.iv_idx]
        norm = torch.sqrt(iv[0] ** 2 + iv[1] ** 2 + iv[2] ** 2) + 1e-8
        iv = torch.stack([torch.matmul(v / norm, self.melW) for v in iv], 1)
        return torch.cat([logmel, iv], 1)
