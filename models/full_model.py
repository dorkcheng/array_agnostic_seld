# -*- coding: utf-8 -*-
"""完整模型：任意阵列 STFT -> [LS FOA 编码 + 神经残差修正] -> 4ch FOA STFT -> logmel+IV -> SELD 骨干。"""
import torch
import torch.nn as nn

from seld.features import LogmelIV, stft_reim
from .backbone import build_backbone
from .ls_foa_encoder import LSFOAEncoder
from .residual_tac import ResidualTACFrontend


class ArrayAgnosticSELD(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        a, m = cfg.audio, cfg.model
        self.n_fft, self.hop, self.fs = a.n_fft, a.hop, a.fs
        self.feat_per_label = int(round(a.label_hop * a.fs)) // a.hop
        self.ls = LSFOAEncoder(a.fs, a.n_fft, m.ls.n_dirs, m.ls.reg)
        self.use_residual = m.residual.enabled
        self.residual = ResidualTACFrontend(a.fs, a.n_fft, m.residual.hidden, m.residual.n_tac, m.residual.freq_stride) if self.use_residual else None
        self.feat = LogmelIV(a.fs, a.n_fft, a.n_mels, tuple(a.get("iv_xyz_index", (1, 2, 3))))
        self.backbone = build_backbone(m.backbone, cfg.data.n_classes)

    # ---- 前端 ----
    def encode_foa(self, X_re, X_im, mic_pos, mic_mask, E_re=None, E_im=None):
        if E_re is None:
            E_re, E_im = self.ls.encoding_matrix(mic_pos, mic_mask)
        ls_re, ls_im = LSFOAEncoder.apply(E_re, E_im, X_re, X_im)
        if not self.use_residual:
            return ls_re, ls_im, ls_re, ls_im
        dE_re, dE_im = self.residual(X_re, X_im, ls_re[:, 0], ls_im[:, 0], mic_pos, mic_mask)
        r_re, r_im = LSFOAEncoder.apply(dE_re, dE_im, X_re, X_im)
        return ls_re + r_re, ls_im + r_im, ls_re, ls_im

    def n_frames(self, Tf):
        return Tf // self.feat_per_label

    def backbone_from_foa_stft(self, re, im):
        feat = self.feat(re, im)
        nf = self.n_frames(feat.shape[2])
        feat = feat[:, :, : nf * self.feat_per_label]
        return self.backbone(feat, nf), feat

    def forward_stft(self, X_re, X_im, mic_pos, mic_mask, E_re=None, E_im=None):
        foa_re, foa_im, ls_re, ls_im = self.encode_foa(X_re, X_im, mic_pos, mic_mask, E_re, E_im)
        res, feat = self.backbone_from_foa_stft(foa_re, foa_im)
        res.update(foa_re=foa_re, foa_im=foa_im, ls_re=ls_re, ls_im=ls_im, feat=feat)
        return res

    def forward(self, mic, mic_pos, mic_mask):
        X_re, X_im = stft_reim(mic, self.n_fft, self.hop)
        m = mic_mask.float()[:, :, None, None]
        return self.forward_stft(X_re * m, X_im * m, mic_pos, mic_mask)

    # ---- 教师路径：理想 FOA ----
    def backbone_on_foa(self, foa_wav=None, foa_feat=None, backbone=None):
        bb = backbone or self.backbone
        if foa_feat is None:
            re, im = stft_reim(foa_wav, self.n_fft, self.hop)
            foa_feat = self.feat(re, im)
        nf = self.n_frames(foa_feat.shape[2])
        return bb(foa_feat[:, :, : nf * self.feat_per_label], nf)


class ONNXWrapper(nn.Module):
    """导出用：输入阵列 STFT 实/虚部 + 预先在设备上算好的 LS 编码矩阵（只依赖麦克风坐标，算一次即可）。"""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, X_re, X_im, mic_pos, mic_mask_f, E_re, E_im):
        mask = mic_mask_f > 0.5
        r = self.m.forward_stft(X_re, X_im, mic_pos, mask, E_re, E_im)
        return r["out"]
