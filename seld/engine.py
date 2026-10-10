# -*- coding: utf-8 -*-
"""训练/推理共用逻辑：训练包装模块（便于 DDP 一次 forward 完成学生+教师+全部损失）、推理拼接、评测。"""
import copy
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from models.full_model import ArrayAgnosticSELD
from seld.features import stft_reim
from seld.losses import ADPITLoss, distill_loss, foa_recon_loss
from seld.metrics import SELDMetrics, decode_multi_accdoa, target_to_list
from seld.utils import load_config


def load_model_weights(model, path, prefix_strip=("module.", "model.")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    for p in prefix_strip:
        sd = {k[len(p):] if k.startswith(p) else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[init] loaded {path}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


class TrainWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = ArrayAgnosticSELD(cfg)
        if cfg.train.get("init_from"):
            load_model_weights(self.model, cfg.train.init_from)
        t = cfg.teacher
        self.teacher_mode = t.mode
        self.teacher_bb, self.teacher_full = None, None
        self.foa_only = bool(cfg.train.get("foa_only", False))
        if self.foa_only:
            t.mode = "none"  # 阶段零：只在理想 FOA 上训练骨干（得到教师）
        if t.mode == "frozen_backbone":
            self.teacher_bb = copy.deepcopy(self.model.backbone)
            if not t.get("ckpt"):
                warnings.warn("[teacher] frozen_backbone 未给 teacher.ckpt：教师的任务头是随机初始化的，蒸馏信号无意义。"
                              "请先运行 configs/teacher_foa.yaml（阶段零）并把 teacher.ckpt 指向其 best.pt")
            if t.get("ckpt"):
                sd = torch.load(t.ckpt, map_location="cpu", weights_only=False)
                sd = sd.get("model", sd)
                sd = {k.split("backbone.", 1)[1]: v for k, v in sd.items() if "backbone." in k}
                print("[teacher] backbone load:", self.teacher_bb.load_state_dict(sd, strict=False))
        elif t.mode == "full_model":
            tcfg = load_config(t.cfg)
            self.teacher_full = load_model_weights(ArrayAgnosticSELD(tcfg), t.ckpt)
        for m in (self.teacher_bb, self.teacher_full):
            if m is not None:
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False
        self.adpit = ADPITLoss(cfg.loss.dist_weight)

    def train(self, mode=True):
        super().train(mode)
        for m in (self.teacher_bb, self.teacher_full):
            if m is not None:
                m.eval()  # 教师始终 eval（BN 统计不变）
        return self

    def forward(self, batch):
        L, cfg = {}, self.cfg
        mic, pos, mask, tgt = batch["mic"], batch["mic_pos"], batch["mic_mask"], batch["target"]
        if self.foa_only:
            tr = self.model.backbone_on_foa(batch["foa"], batch.get("foa_feat"))
            T = min(tr["out"].shape[1], tgt.shape[1])
            L["seld_foa"] = self.adpit(tr["out"][:, :T].float(), tgt[:, :T])
            L["total"] = L["seld_foa"]
            return L
        res = self.model(mic, pos, mask)
        out = res["out"]
        T = min(out.shape[1], tgt.shape[1])
        L["seld"] = self.adpit(out[:, :T].float(), tgt[:, :T])
        total = cfg.loss.seld * L["seld"]
        has_foa = "foa" in batch
        # ---- 教师 ----
        t_out = t_emb = None
        if self.teacher_mode in ("frozen_backbone", "shared") and has_foa:
            feat = batch.get("foa_feat")
            if self.teacher_mode == "shared":
                tr = self.model.backbone_on_foa(batch["foa"], feat)
                if cfg.loss.get("ideal_seld", 0) > 0:  # 骨干同时在理想 FOA 上学 SELD
                    L["ideal_seld"] = self.adpit(tr["out"][:, :T].float(), tgt[:, :T])
                    total = total + cfg.loss.ideal_seld * L["ideal_seld"]
            else:
                with torch.no_grad():
                    tr = self.model.backbone_on_foa(batch["foa"], feat, backbone=self.teacher_bb)
            t_out, t_emb = tr["out"], tr["emb"]
        elif self.teacher_mode == "full_model":
            with torch.no_grad():
                tr = self.teacher_full(mic, pos, mask)
            t_out = tr["out"]
        if t_out is not None and cfg.loss.distill > 0:
            T2 = min(T, t_out.shape[1])
            L["distill"] = distill_loss(out[:, :T2], t_out[:, :T2], res["emb"][:, :T2],
                                        None if t_emb is None else t_emb[:, :T2], cfg.loss.get("distill_emb", 0.0),
                                        cfg.loss.get("distill_conf_thr", 0.0))
            total = total + cfg.loss.distill * L["distill"]
        # ---- FOA 重建 ----
        if has_foa and cfg.loss.foa_recon > 0:
            with torch.no_grad():
                f_re, f_im = stft_reim(batch["foa"], self.model.n_fft, self.model.hop)
            n_bins = f_re.shape[-1]
            hi = int(n_bins * cfg.loss.get("foa_recon_fmax", 8000) / (self.model.fs / 2))
            L["foa_recon"] = foa_recon_loss(res["foa_re"], res["foa_im"], f_re, f_im, 2, hi)
            total = total + cfg.loss.foa_recon * L["foa_recon"]
        L["total"] = total
        return L


@torch.no_grad()
def run_inference(model, loader, device, amp=False, use_foa=False):
    """返回 {sid: (pred (T,3,4,C), target (T,6,5,C))}，按 start 拼接所有切片。
    use_foa=True: 骨干直接吃数据中的（理想）FOA，用于评估教师/阶段零。"""
    model.eval()
    chunks = defaultdict(list)
    for b in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            if use_foa:
                r = model.backbone_on_foa(b["foa"].to(device))
            else:
                r = model(b["mic"].to(device), b["mic_pos"].to(device), b["mic_mask"].to(device))
        out = r["out"].float().cpu().numpy()
        for i, sid in enumerate(b["sid"]):
            chunks[sid].append((int(b["start"][i]), out[i], b["target"][i].numpy()))
    res = {}
    for sid, lst in chunks.items():
        lst.sort(key=lambda x: x[0])
        res[sid] = (np.concatenate([x[1] for x in lst]), np.concatenate([x[2][: x[1].shape[0]] for x in lst]))
    return res


def evaluate_predictions(preds, n_classes, thr=0.5, dist_scale=5.0, doa_thr=20.0, dist_thr=None):
    m = SELDMetrics(n_classes, doa_thr, dist_thr)
    for sid, (p, t) in preds.items():
        T = min(len(p), len(t))
        m.update(decode_multi_accdoa(p[:T], thr, dist_scale=dist_scale), target_to_list(t[:T], dist_scale))
    return m.compute()
