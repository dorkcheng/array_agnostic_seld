# -*- coding: utf-8 -*-
"""训练入口（支持 torchrun DDP、AMP fp16、每 epoch 断点 + 自动续训、12h 会话时间保护）。

单卡/CPU:  python train.py --config configs/smoke.yaml
Kaggle 2×T4: torchrun --nproc_per_node=2 train.py --config configs/sim_pretrain.yaml
命令行覆盖: python train.py --config x.yaml train.epochs=3 data.chunk_sec=4
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seld.dataset import SELDH5Dataset, collate_pad  # noqa: E402
from seld.engine import TrainWrapper, evaluate_predictions, run_inference  # noqa: E402
from seld.utils import load_config, set_seed  # noqa: E402

T_START = time.time()


def setup_dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        rank, local = dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        return rank, dist.get_world_size(), local
    return 0, 1, 0


def log(rank, *a):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def save_ckpt(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # 原子替换，避免会话中断时写坏断点


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    rank, world, local = setup_dist()
    set_seed(cfg.seed + rank)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg.train.amp) and device.type == "cuda"
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    if rank == 0:
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)

    d = cfg.data
    ds_kw = dict(n_classes=d.n_classes, chunk_sec=d.chunk_sec, fs=cfg.audio.fs, label_hop=cfg.audio.label_hop,
                 dist_scale=d.dist_scale, feat_hop=cfg.audio.hop)
    train_ds = SELDH5Dataset(d.train_h5, mode="train", mic_drop_prob=d.mic_drop_prob, min_mics=d.min_mics, **ds_kw)
    sampler = DistributedSampler(train_ds, shuffle=True) if world > 1 else None
    collate = lambda b: collate_pad(b, d.max_mics)
    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=sampler is None, sampler=sampler,
                          num_workers=d.num_workers, collate_fn=collate, drop_last=len(train_ds) >= cfg.train.batch_size * world,
                          pin_memory=device.type == "cuda", persistent_workers=d.num_workers > 0)
    val_dl, foa_only = None, False
    if d.get("val_h5"):
        foa_only = bool(cfg.train.get("foa_only", False))
        val_ds = SELDH5Dataset(d.val_h5, mode="eval", use_foa=foa_only, use_foa_feat=False, **ds_kw)
        val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=d.num_workers, collate_fn=collate)

    wrapper = TrainWrapper(cfg).to(device)
    n_tr = sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad)
    log(rank, f"model params (trainable) {n_tr/1e6:.2f}M | world={world} device={device} amp={amp}")
    bb_params = [p for n, p in wrapper.model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    fe_params = [p for n, p in wrapper.model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": fe_params, "lr": cfg.train.lr},
                             {"params": bb_params, "lr": cfg.train.get("lr_backbone", cfg.train.lr)}],
                            weight_decay=cfg.train.weight_decay)
    steps_per_epoch = max(1, len(train_dl))
    total_steps = cfg.train.epochs * steps_per_epoch
    warm = int(cfg.train.get("warmup_epochs", 0) * steps_per_epoch)
    lr_lambda = lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))) * 0.99 + 0.01
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # ---- 自动续训 ----
    start_epoch, best, step = 0, float("inf"), 0
    last = os.path.join(exp_dir, "last.pt")
    # Kaggle 新会话里 /kaggle/working 是空的：可把上个会话的输出挂为 input，并设 train.resume_from 指向其中的 last.pt
    src = last if os.path.exists(last) else cfg.train.get("resume_from")
    if cfg.train.get("auto_resume", True) and src and os.path.exists(src):
        ck = torch.load(src, map_location="cpu", weights_only=False)
        wrapper.model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best, step = ck["epoch"], ck["best"], ck["step"]
        log(rank, f"resumed from {src}: epoch={start_epoch} step={step} best={best:.4f}")

    model_ddp = DDP(wrapper, device_ids=[local] if device.type == "cuda" else None,
                    find_unused_parameters=cfg.train.get("find_unused_parameters", False)) if world > 1 else wrapper
    max_sec = cfg.train.get("max_hours", 11.3) * 3600

    def state(epoch):
        return dict(model=wrapper.model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, best=best, step=step, cfg=dict(cfg))

    for epoch in range(start_epoch, cfg.train.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model_ddp.train()
        agg, n, t0 = {}, 0, time.time()
        for it, b in enumerate(train_dl):
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                L = model_ddp(b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(L["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(wrapper.model.parameters(), cfg.train.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
            if it % cfg.train.log_every == 0:
                log(rank, f"ep {epoch} it {it}/{steps_per_epoch} " + " ".join(f"{k}={float(v.detach()):.4f}" for k, v in L.items()) + f" lr={sched.get_last_lr()[0]:.2e}")
            if time.time() - T_START > max_sec:  # Kaggle 12h 会话保护：保存“本 epoch 未完成”的断点后退出
                if rank == 0:
                    save_ckpt(last, **state(epoch))
                log(rank, "time budget reached, checkpoint saved, exiting; rerun the same command to resume")
                if world > 1:
                    dist.barrier()
                    dist.destroy_process_group()
                return
        log(rank, f"epoch {epoch} done in {time.time()-t0:.1f}s | " + " ".join(f"{k}={v/n:.4f}" for k, v in agg.items()))
        metrics = None
        if rank == 0 and val_dl is not None and (epoch + 1) % cfg.train.get("val_every", 1) == 0:
            preds = run_inference(wrapper.model, val_dl, device, amp, use_foa=foa_only)
            metrics = evaluate_predictions(preds, d.n_classes, cfg.eval.thr, d.dist_scale)
            log(rank, f"val epoch {epoch}: " + json.dumps({k: round(v, 4) for k, v in metrics.items()}))
            if metrics["SELD"] < best:
                best = metrics["SELD"]
                save_ckpt(os.path.join(exp_dir, "best.pt"), **state(epoch + 1))
        if rank == 0:
            save_ckpt(last, **state(epoch + 1))
            with open(os.path.join(exp_dir, "log.jsonl"), "a") as f:
                f.write(json.dumps(dict(epoch=epoch, train={k: v / n for k, v in agg.items()}, val=metrics)) + "\n")
        if world > 1:
            dist.barrier()
    if rank == 0 and not os.path.exists(os.path.join(exp_dir, "best.pt")):
        save_ckpt(os.path.join(exp_dir, "best.pt"), **state(cfg.train.epochs))
    log(rank, "training finished")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
