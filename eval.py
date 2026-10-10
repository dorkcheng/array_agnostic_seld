# -*- coding: utf-8 -*-
"""评测：DCASE 风格指标 + “阵列互换 (array swap)” 评测。

  python eval.py --config configs/smoke.yaml --ckpt /path/best.pt \
      --h5 seen=/data/h5/test.h5 unseen_geom=/data/h5/test_unseen.h5 \
      --mic-subsets all,2,3,4 --out results.json

  * --h5 可给多个 name=path：例如训练中见过的几何 vs 未见过的几何（仿真时用不同 --array-types 生成）
  * --mic-subsets：all = 全部麦克风；整数 k = 每段随机（可复现）选 k 个麦克风（k 大于实际数则用全部）
  * 结果按数据集 × 子集 × 阵列类型分别给出 F20 / DOAE / RDE / SELD
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.full_model import ArrayAgnosticSELD  # noqa: E402
from seld.dataset import SELDH5Dataset, collate_pad  # noqa: E402
from seld.engine import evaluate_predictions, load_model_weights, run_inference  # noqa: E402
from seld.utils import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--h5", nargs="+", required=True, help="name=path 或 path")
    ap.add_argument("--mic-subsets", default="all")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--use-foa", action="store_true", help="骨干直接吃数据中的理想 FOA（评估教师/上限）")
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    cfg = load_config(a.config, a.overrides)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_weights(ArrayAgnosticSELD(cfg), a.ckpt).to(dev).eval()
    thr = a.thr if a.thr is not None else cfg.eval.thr
    d = cfg.data
    results = {}
    for spec in a.h5:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        for sub in a.mic_subsets.split(","):
            subset = None if sub == "all" else int(sub)
            ds = SELDH5Dataset(path, d.n_classes, d.chunk_sec, cfg.audio.fs, cfg.audio.label_hop, mode="eval",
                               dist_scale=d.dist_scale, mic_subset=subset, use_foa=a.use_foa, use_foa_feat=False)
            dl = DataLoader(ds, batch_size=a.batch_size, shuffle=False, collate_fn=lambda b: collate_pad(b, d.max_mics))
            preds = run_inference(model, dl, dev, amp=cfg.train.amp, use_foa=a.use_foa)
            key = f"{name}/mics={sub}"
            results[key] = {"all": evaluate_predictions(preds, d.n_classes, thr, d.dist_scale)}
            by_type = defaultdict(dict)
            for sid, v in preds.items():
                by_type[ds.array_type[sid]][sid] = v
            for t, pv in by_type.items():
                results[key][t] = evaluate_predictions(pv, d.n_classes, thr, d.dist_scale)
            print(key, json.dumps({k: {m: round(x, 3) for m, x in v.items()} for k, v in results[key].items()}, ensure_ascii=False))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1, ensure_ascii=False)
        print("saved", a.out)


if __name__ == "__main__":
    main()
