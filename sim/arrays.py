# -*- coding: utf-8 -*-
"""随机麦克风阵列几何生成（坐标单位 m，相对阵列中心，世界坐标轴方向）。"""
import numpy as np

ARRAY_TYPES = ["linear", "circular", "phone", "random", "tetra"]


def _rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot_small_tilt(rng, max_deg=15):
    a, b = np.deg2rad(rng.uniform(-max_deg, max_deg, 2))
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return rx @ ry


def make_array(kind, n_mics, rng):
    if kind == "linear":
        d = rng.uniform(0.01, 0.05)
        pos = np.zeros((n_mics, 3))
        pos[:, 0] = (np.arange(n_mics) - (n_mics - 1) / 2) * d
    elif kind == "circular":
        r = rng.uniform(0.02, 0.06)
        ang = 2 * np.pi * np.arange(n_mics) / n_mics + rng.uniform(0, 2 * np.pi)
        pos = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n_mics)], 1)
    elif kind == "phone":
        # 手机外形：约 15×7×0.8 cm 的长方体表面上的若干位置（底部、顶部、背面）
        L, W, H = rng.uniform(0.13, 0.17), rng.uniform(0.06, 0.08), 0.008
        cand = np.array([[-L / 2, -W / 4, 0], [-L / 2, W / 4, 0], [L / 2, 0, 0], [L / 2 - 0.01, W / 3, H],
                         [0.3 * L, -W / 3, H], [-L / 2, 0, H], [L / 2, -W / 4, 0], [0, W / 2, 0]])
        pos = cand[rng.permutation(len(cand))[:n_mics]]
        pos = pos[:, [0, 2, 1]] if rng.random() < 0.5 else pos  # 竖屏/平放两种姿态
    elif kind == "tetra":
        from seld.utils import starss23_mic_coords
        base = starss23_mic_coords()
        pos = base[:min(n_mics, 4)] * rng.uniform(0.7, 1.5)
    elif kind == "random":
        r = rng.uniform(0.02, 0.08)
        v = rng.standard_normal((n_mics, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pos = v * r * rng.uniform(0.3, 1.0, (n_mics, 1))
    else:
        raise ValueError(kind)
    R = _rot_z(rng.uniform(0, 2 * np.pi)) @ _rot_small_tilt(rng)
    pos = pos @ R.T
    pos -= pos.mean(0, keepdims=True)
    return pos.astype(np.float64)
