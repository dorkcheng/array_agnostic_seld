# -*- coding: utf-8 -*-
"""导出蒸馏小模型为 ONNX（手机端）。

ONNX 图输入（全部 float32）：
  X_re, X_im   (1, M, T, F)   阵列 STFT 实/虚部（hann 窗, n_fft=1024, hop=240；在端上用 FFT 计算）
  mic_pos      (1, M, 3)      麦克风坐标 (m)
  mic_mask     (1, M)         1=有效
  E_re, E_im   (1, F, 4, M)   LS-FOA 编码矩阵：只依赖麦克风坐标，设备上算一次即可。
                              本脚本同时导出 <out>_E.npz（给定几何的 E），以及可在端上复现的 numpy 实现 compute_E_numpy()
输出：
  out (1, T/10, 3, 4, C) multi-ACCDOA+距离
复数线性代数 (linalg.solve) 不进 ONNX 图，这是刻意的设计：ONNX 对复数支持差。

用法：python export/export_onnx.py --config configs/distill_small.yaml --ckpt best.pt --out small.onnx --n-mics 4
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.full_model import ArrayAgnosticSELD, ONNXWrapper  # noqa: E402
from seld.engine import load_model_weights  # noqa: E402
from seld.utils import SPEED_OF_SOUND, fibonacci_sphere, load_config, starss23_mic_coords  # noqa: E402


def compute_E_numpy(mic_pos, fs=24000, n_fft=1024, n_dirs=240, reg=1e-2, c=SPEED_OF_SOUND):
    """与 LSFOAEncoder.encoding_matrix 等价的 numpy 版本（全部麦克风有效）。mic_pos (M,3) -> E (F,4,M) complex"""
    pos = mic_pos - mic_pos.mean(0, keepdims=True)
    dirs = fibonacci_sphere(n_dirs)
    Y = np.stack([np.ones(n_dirs), dirs[:, 1], dirs[:, 2], dirs[:, 0]], 0)
    f = np.linspace(0, fs / 2, n_fft // 2 + 1)
    A = np.exp(1j * 2 * np.pi * f[:, None, None] / c * (pos @ dirs.T)[None])  # (F,M,Q)
    G = A @ A.conj().transpose(0, 2, 1)
    M = pos.shape[0]
    lam = reg * np.trace(G, axis1=1, axis2=2).real / M + 1e-6
    G = G + lam[:, None, None] * np.eye(M)[None]
    EH = np.linalg.solve(G, A @ Y.T)
    return EH.conj().transpose(0, 2, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-mics", type=int, default=4, help="导出时示例麦克风数（M 维为动态轴）")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    model = ArrayAgnosticSELD(cfg)
    if a.ckpt:
        load_model_weights(model, a.ckpt)
    model.eval()
    w = ONNXWrapper(model).eval()
    au = cfg.audio
    M, F = a.n_mics, au.n_fft // 2 + 1
    T = int(a.seconds / au.label_hop) * (int(au.label_hop * au.fs) // au.hop)
    pos = starss23_mic_coords()[:M] if M <= 4 else np.random.randn(M, 3).astype(np.float32) * 0.04
    E = compute_E_numpy(pos.astype(np.float64), au.fs, au.n_fft, cfg.model.ls.n_dirs, cfg.model.ls.reg)
    E_re = torch.from_numpy(E.real.astype(np.float32))[None]
    E_im = torch.from_numpy(E.imag.astype(np.float32))[None]
    X_re, X_im = torch.randn(1, M, T, F) * 0.1, torch.randn(1, M, T, F) * 0.1
    pos_t, mask = torch.from_numpy(pos)[None], torch.ones(1, M)
    # 校验 numpy 版 E 与 torch 版一致
    Er_t, Ei_t = model.ls.encoding_matrix(pos_t, mask > 0.5)
    print("E numpy vs torch max abs diff:", float((Er_t - E_re).abs().max()), float((Ei_t - E_im).abs().max()))
    inputs = (X_re, X_im, pos_t, mask, E_re, E_im)
    names = ["X_re", "X_im", "mic_pos", "mic_mask", "E_re", "E_im"]
    dyn = {"X_re": {1: "M", 2: "T"}, "X_im": {1: "M", 2: "T"}, "mic_pos": {1: "M"}, "mic_mask": {1: "M"},
           "E_re": {3: "M"}, "E_im": {3: "M"}, "out": {1: "T_out"}}
    with torch.no_grad():
        ref = w(*inputs).numpy()
    torch.onnx.export(w, inputs, a.out, input_names=names, output_names=["out"], dynamic_axes=dyn,
                      opset_version=a.opset, dynamo=False)
    np.savez(a.out.replace(".onnx", "_E.npz"), E_re=E_re.numpy(), E_im=E_im.numpy(), mic_pos=pos)
    print("exported", a.out, f"({os.path.getsize(a.out)/1e6:.2f} MB)")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        o = sess.run(None, {n: t.numpy() for n, t in zip(names, inputs)})[0]
        print("onnxruntime vs torch max abs diff:", float(np.abs(o - ref).max()), "out shape", o.shape)
        # 换一个麦克风数 / 时长测试动态轴
        M2, T2 = max(2, M - 1), T * 2
        E2 = compute_E_numpy(pos[:M2].astype(np.float64), au.fs, au.n_fft, cfg.model.ls.n_dirs, cfg.model.ls.reg)
        feed = dict(X_re=np.random.randn(1, M2, T2, F).astype(np.float32) * 0.1, X_im=np.random.randn(1, M2, T2, F).astype(np.float32) * 0.1,
                    mic_pos=pos[None, :M2], mic_mask=np.ones((1, M2), np.float32),
                    E_re=E2.real[None].astype(np.float32), E_im=E2.imag[None].astype(np.float32))
        o2 = sess.run(None, feed)[0]
        with torch.no_grad():
            r2 = w(*[torch.from_numpy(feed[n]) for n in names]).numpy()
        print(f"dynamic test M={M2} T={T2}: out {o2.shape}, max diff {float(np.abs(o2 - r2).max()):.2e}")
    except ImportError:
        print("onnxruntime 未安装，跳过数值校验")


if __name__ == "__main__":
    main()
