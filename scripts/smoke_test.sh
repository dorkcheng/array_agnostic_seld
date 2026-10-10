#!/usr/bin/env bash
# CPU 端到端冒烟测试：仿真 -> 预处理 -> 训练 2 epoch(+续训) -> DDP(gloo) -> 评测(array swap) -> ONNX 导出
set -euo pipefail
cd "$(dirname "$0")/.."
W=${SMOKE_DIR:-/workspace/smoke}
rm -rf "$W"; mkdir -p "$W"
OV="out_dir=$W/ckpt data.train_h5=$W/h5/train.h5 data.val_h5=$W/h5/test.h5"
python -m sim.dummy_sources --out $W/dummy_sources --n-per-class 6
python -m sim.simulate --source-csv $W/dummy_sources/sources.csv --out $W/sim_train --n-scenes 16 --duration 3 --max-order 6
python -m sim.simulate --source-csv $W/dummy_sources/sources.csv --out $W/sim_test --n-scenes 6 --duration 3 --max-order 6 \
       --array-types random,tetra --seed 7 --prefix test_      # 测试集用训练中少见/未见的几何
python -m sim.precompute --root $W/sim_train --out $W/h5/train.h5
python -m sim.precompute --root $W/sim_test --out $W/h5/test.h5
python scripts/check_ls_foa.py --h5 $W/h5/test.h5 --n 3
python scripts/check_metrics_oracle.py $W/h5/test.h5 5
python train.py --config configs/smoke.yaml $OV
python train.py --config configs/smoke.yaml $OV train.epochs=3          # 测试自动续训
torchrun --nproc_per_node=2 train.py --config configs/smoke.yaml $OV exp_name=smoke_ddp
python eval.py --config configs/smoke.yaml --ckpt $W/ckpt/smoke/best.pt --h5 unseen_geom=$W/h5/test.h5 train=$W/h5/train.h5 \
       --mic-subsets all,2,3 --out $W/eval_results.json $OV
python export/export_onnx.py --config configs/smoke.yaml --ckpt $W/ckpt/smoke/best.pt --out $W/small.onnx --n-mics 4 $OV
echo "SMOKE TEST OK"
