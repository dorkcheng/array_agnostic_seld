# -*- coding: utf-8 -*-
"""通用工具：配置读取、几何/球谐函数、随机种子等。"""
import copy
import math
import os
import random

import numpy as np
import torch
import yaml

SPEED_OF_SOUND = 343.0


class Cfg(dict):
    """支持属性访问的字典（cfg.a.b）。"""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError:
            raise AttributeError(k)
        return v

    def __setattr__(self, k, v):
        self[k] = v


def _to_cfg(d):
    if isinstance(d, dict):
        return Cfg({k: _to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_cfg(v) for v in d]
    return d


def _deep_update(base, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_value(s):
    try:
        return yaml.safe_load(s)
    except Exception:
        return s


def load_config(path, overrides=None):
    """读取 yaml 配置。支持 `_base_: other.yaml` 继承，以及命令行覆盖 a.b.c=value。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_base_" in cfg:
        base_path = os.path.join(os.path.dirname(path), cfg.pop("_base_"))
        base = load_config(base_path)
        cfg = _deep_update(copy.deepcopy(dict(base)), cfg)
    for ov in overrides or []:
        k, v = ov.split("=", 1)
        node = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            node = node.setdefault(kk, {})
        node[keys[-1]] = _parse_value(v)
    return _to_cfg(_plain(cfg))


def _plain(d):
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_plain(v) for v in d]
    return d


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# 几何 & 一阶球谐（ACN 通道顺序 W,Y,Z,X；SN3D 归一化，与 DCASE FOA 一致）
# ---------------------------------------------------------------------------
def sph2cart(azi_deg, ele_deg, r=1.0):
    azi, ele = np.deg2rad(azi_deg), np.deg2rad(ele_deg)
    return np.stack([r * np.cos(ele) * np.cos(azi), r * np.cos(ele) * np.sin(azi), r * np.sin(ele)], -1)


def cart2sph(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.linalg.norm(xyz, axis=-1)
    azi = np.rad2deg(np.arctan2(xyz[..., 1], xyz[..., 0]))
    ele = np.rad2deg(np.arcsin(np.clip(xyz[..., 2] / np.maximum(r, 1e-9), -1, 1)))
    return azi, ele, r


def foa_sh(u):
    """u: (..., 3) 单位方向向量 -> (..., 4) 一阶实球谐 [W, Y, Z, X]（ACN/SN3D）。"""
    u = np.asarray(u)
    return np.stack([np.ones_like(u[..., 0]), u[..., 1], u[..., 2], u[..., 0]], -1)


def foa_sh_torch(u):
    return torch.stack([torch.ones_like(u[..., 0]), u[..., 1], u[..., 2], u[..., 0]], -1)


def fibonacci_sphere(n):
    """近似均匀的球面方向网格，(n,3)。"""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], -1)


def angular_distance_deg(u, v):
    """u, v: (...,3) 向量（不必归一化），返回夹角（度）。"""
    u = u / np.maximum(np.linalg.norm(u, axis=-1, keepdims=True), 1e-9)
    v = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)
    return np.rad2deg(np.arccos(np.clip((u * v).sum(-1), -1, 1)))


# STARSS23 / TAU-SRIR 的 MIC 格式：Eigenmike em32 上的四面体子阵 (capsule 6,10,26,22)，半径 4.2cm
# 来源：DCASE2023/2024 Task3 官方说明 "MIC: (azimuth, elevation, r)"
STARSS23_MIC_SPH = [(45.0, 35.0, 0.042), (-45.0, -35.0, 0.042), (135.0, -35.0, 0.042), (-135.0, 35.0, 0.042)]


def starss23_mic_coords():
    return np.stack([sph2cart(a, e, r) for a, e, r in STARSS23_MIC_SPH], 0).astype(np.float32)
