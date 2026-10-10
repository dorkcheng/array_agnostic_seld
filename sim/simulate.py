# -*- coding: utf-8 -*-
"""基于 pyroomacoustics 的阵列无关 SELD 场景仿真（多进程 CLI）。

每个场景输出：
  mic/{id}.wav        任意阵列 (M=2..8 通道) 录音
  foa/{id}.wav        同一场景的“理想 FOA”渲染（4 通道, ACN/SN3D: W,Y,Z,X）
  geometry/{id}.json  麦克风坐标（相对阵列中心, m）+ 房间信息
  metadata/{id}.csv   每 100ms 帧标签: frame,class,track,azi,ele,dist_cm
  scenes.csv          场景列表

理想 FOA 的做法（见 README）：用 pra 的镜像源模型 (ISM) 求出每个声源的全部镜像源位置及其
反射衰减，在阵列中心放一个“虚拟理想一阶球谐麦克风”：每个镜像源按 damping/d 的幅度（与 pra 约定一致）、
d/c 的分数延迟（与 pra 相同的 81 点加窗 sinc，以及相同的 40 样点固定延迟，使与阵列录音时间对齐）、
并乘以方向 u 的一阶实球谐 [1, u_y, u_z, u_x] 叠加。直达声 + 镜像源与阵列 RIR 完全来自同一 ISM，
因此两者描述的是同一个声场（只是 ISM 阶数截断的晚期混响尾部两者都没有）。
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import numpy as np
import soundfile as sf

from sim.arrays import ARRAY_TYPES, make_array
from seld.utils import cart2sph, foa_sh

LABEL_HOP = 0.1


def load_source_table(csv_path):
    root = os.path.dirname(os.path.abspath(csv_path))
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            p = r["filename"]
            rows.append((p if os.path.isabs(p) else os.path.join(root, p), int(r["class_idx"])))
    return rows


def _frac_delay_filters(frac, L=81):
    n = np.arange(L)[None, :] - (L - 1) / 2 - frac[:, None]
    return np.sinc(n) * np.hanning(L)[None, :]


def render_foa_rir(images, damping, center, fs, c, rir_len):
    """images:(3,K) damping:(K,) -> (4, rir_len) 理想 FOA RIR（与 pra 时间轴对齐）。"""
    L = 81
    d = np.linalg.norm(images - center[:, None], axis=0)
    d = np.maximum(d, 1e-3)
    u = (images - center[:, None]) / d[None]
    g = damping / d  # 与 pra 的 RIR 幅度约定一致（1/d，不含 4π）
    t = d / c * fs
    ti = np.floor(t).astype(int)
    h = _frac_delay_filters(t - ti, L) * g[:, None]  # (K,L)
    sh = foa_sh(u.T)  # (K,4)
    out = np.zeros((4, rir_len + L + 1))
    for k in range(len(d)):
        s = ti[k]
        if s + L >= out.shape[1]:
            continue
        out[:, s:s + L] += sh[k][:, None] * h[k][None, :]
    return out[:, :rir_len]


def simulate_scene(args):
    idx, cfg = args
    import pyroomacoustics as pra

    rng = np.random.default_rng(cfg["seed"] * 100003 + idx)
    fs, dur = cfg["fs"], cfg["duration"]
    N = int(fs * dur)
    n_frames = int(round(dur / LABEL_HOP))
    src_table = cfg["sources"]

    # ---- 房间 ----
    room_dim = np.array([rng.uniform(3, 10), rng.uniform(3, 10), rng.uniform(2.5, 4.0)])
    rt60 = rng.uniform(*cfg["rt60"])
    e_abs, max_order = pra.inverse_sabine(rt60, room_dim)
    max_order = int(min(max_order, cfg["max_order"]))
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_abs), max_order=max_order, air_absorption=False)

    # ---- 阵列 ----
    kind = cfg["array_types"][rng.integers(len(cfg["array_types"]))]
    M = int(rng.integers(cfg["mics_min"], cfg["mics_max"] + 1))
    rel = make_array(kind, M, rng)
    M = rel.shape[0]
    center = np.array([rng.uniform(1.0, room_dim[0] - 1.0), rng.uniform(1.0, room_dim[1] - 1.0), rng.uniform(1.0, 2.0)])
    room.add_microphone_array((rel + center).T)

    # ---- 声源事件（静止声源）----
    n_src = int(rng.integers(1, cfg["max_sources"] + 1))
    events, signals = [], []
    for k in range(n_src):
        for _ in range(100):
            dist = rng.uniform(0.5, 4.0)
            azi, ele = rng.uniform(-180, 180), rng.uniform(-40, 40)
            u = np.array([np.cos(np.deg2rad(ele)) * np.cos(np.deg2rad(azi)),
                          np.cos(np.deg2rad(ele)) * np.sin(np.deg2rad(azi)), np.sin(np.deg2rad(ele))])
            pos = center + dist * u
            if np.all(pos > 0.3) and np.all(pos < room_dim - 0.3):
                break
        else:
            continue
        fn, cl = src_table[rng.integers(len(src_table))]
        x, sr = sf.read(fn, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != fs:
            from scipy.signal import resample_poly
            x = resample_poly(x, fs, sr).astype(np.float32)
        x = x[: int(fs * min(dur * 0.9, len(x) / fs))]
        x = x / (np.sqrt(np.mean(x ** 2)) + 1e-9) * 10 ** (rng.uniform(-6, 0) / 20) * 0.05
        onset = rng.uniform(0, max(0.0, dur - len(x) / fs))
        on_s = int(onset * fs)
        sig = np.zeros(N, np.float32)
        sig[on_s:on_s + len(x)] = x[: N - on_s]
        room.add_source(pos, signal=sig)
        signals.append(sig)
        a, e, r = cart2sph(pos - center)
        f0 = int(np.floor(onset / LABEL_HOP))
        f1 = int(np.ceil((onset + len(x) / fs) / LABEL_HOP))
        events.append(dict(cls=cl, track=k, azi=float(a), ele=float(e), dist=float(r), f0=f0, f1=min(f1, n_frames)))

    room.compute_rir()
    c = pra.constants.get("c")
    mic = np.zeros((M, N))
    foa = np.zeros((4, N))
    for s_i, src in enumerate(room.sources):
        sig = signals[s_i]
        for m in range(M):
            mic[m] += np.convolve(sig, room.rir[m][s_i])[:N]
        damping = src.damping.mean(0)
        h_foa = render_foa_rir(src.images, damping, center, fs, c, len(room.rir[0][s_i]))
        for ch in range(4):
            foa[ch] += np.convolve(sig, h_foa[ch])[:N]
    # 阵列加少量传感器噪声（FOA 教师输入保持干净）
    snr = rng.uniform(*cfg["snr"])
    p = np.mean(mic ** 2) + 1e-12
    mic += rng.standard_normal(mic.shape) * np.sqrt(p / 10 ** (snr / 10))
    peak = max(np.abs(mic).max(), np.abs(foa).max(), 1e-6)
    scale = 0.9 / peak if peak > 0.9 else 1.0

    sid = f"{cfg['prefix']}{idx:06d}"
    out = cfg["out"]
    sf.write(os.path.join(out, "mic", sid + ".wav"), (mic * scale).T.astype(np.float32), fs, subtype="FLOAT")
    sf.write(os.path.join(out, "foa", sid + ".wav"), (foa * scale).T.astype(np.float32), fs, subtype="FLOAT")
    with open(os.path.join(out, "geometry", sid + ".json"), "w") as f:
        json.dump(dict(mic_coords=rel.tolist(), array_type=kind, n_mics=M, room_dim=room_dim.tolist(),
                       rt60=rt60, max_order=max_order, array_center=center.tolist(), fs=fs,
                       coord_frame="相对阵列中心，世界坐标轴；FOA 虚拟麦克风位于阵列中心"), f, ensure_ascii=False, indent=1)
    rows = []
    for ev in events:
        for fr in range(ev["f0"], ev["f1"]):
            rows.append([fr, ev["cls"], ev["track"], ev["azi"], ev["ele"], ev["dist"] * 100])
    rows.sort(key=lambda r: (r[0], r[2]))
    from seld.labels import write_dcase_csv
    write_dcase_csv(os.path.join(out, "metadata", sid + ".csv"), rows)
    return sid, kind, M, n_src


def main():
    ap = argparse.ArgumentParser(description="阵列无关 SELD 场景仿真")
    ap.add_argument("--source-csv", required=True, help="filename,class_idx[,class_name] 单声道事件表")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--duration", type=float, default=10.0, help="每个场景时长 (s)")
    ap.add_argument("--fs", type=int, default=24000)
    ap.add_argument("--array-types", default=",".join(ARRAY_TYPES[:4]), help=f"可选: {ARRAY_TYPES}")
    ap.add_argument("--mics-min", type=int, default=2)
    ap.add_argument("--mics-max", type=int, default=8)
    ap.add_argument("--max-sources", type=int, default=3)
    ap.add_argument("--rt60-min", type=float, default=0.2)
    ap.add_argument("--rt60-max", type=float, default=0.8)
    ap.add_argument("--max-order", type=int, default=12, help="ISM 最大阶数上限（越大越慢）")
    ap.add_argument("--snr-min", type=float, default=20)
    ap.add_argument("--snr-max", type=float, default=40)
    ap.add_argument("--prefix", default="scene_")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for d in ["mic", "foa", "geometry", "metadata"]:
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    cfg = dict(fs=a.fs, duration=a.duration, sources=load_source_table(a.source_csv),
               array_types=a.array_types.split(","), mics_min=a.mics_min, mics_max=a.mics_max,
               max_sources=a.max_sources, rt60=(a.rt60_min, a.rt60_max), max_order=a.max_order,
               snr=(a.snr_min, a.snr_max), prefix=a.prefix, out=a.out, seed=a.seed)
    jobs = [(i, cfg) for i in range(a.n_scenes)]
    with Pool(max(1, a.workers)) as pool:
        res = list(pool.imap_unordered(simulate_scene, jobs, chunksize=1))
    res.sort()
    with open(os.path.join(a.out, "scenes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene", "array_type", "n_mics", "n_sources"])
        w.writerows(res)
    print(f"done: {len(res)} scenes -> {a.out}")


if __name__ == "__main__":
    main()
