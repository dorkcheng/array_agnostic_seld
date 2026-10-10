# -*- coding: utf-8 -*-
"""HDF5 数据集：可变麦克风数 (2..max_mics)，collate 时补零并给出 mask。"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import rows_to_adpit


class SELDH5Dataset(Dataset):
    """
    mode='train': 随机裁剪 chunk_sec，随机麦克风子集增强
    mode='eval' : 按 chunk_sec 不重叠切片覆盖整段（最后一片补零），返回 (sid, start_frame) 用于拼接
    mic_subset: eval 时固定只用前 k 个（或给定索引列表）麦克风，用于 array-swap 评测
    """

    def __init__(self, h5_path, n_classes, chunk_sec=10.0, fs=24000, label_hop=0.1, mode="train",
                 dist_scale=5.0, mic_drop_prob=0.0, min_mics=2, mic_subset=None, subset_seed=0,
                 use_foa=True, use_foa_feat=True, feat_hop=240):
        self.path, self.mode = h5_path, mode
        self.C, self.fs, self.hop_s = n_classes, fs, int(round(label_hop * fs))
        self.chunk_frames = int(round(chunk_sec / label_hop))
        self.chunk_samples = self.chunk_frames * self.hop_s
        self.feat_per_label = self.hop_s // feat_hop
        self.dist_scale, self.mic_drop_prob, self.min_mics = dist_scale, mic_drop_prob, min_mics
        self.mic_subset, self.subset_seed = mic_subset, subset_seed
        self.use_foa, self.use_foa_feat = use_foa, use_foa_feat
        self.h5 = None
        with h5py.File(h5_path, "r") as h5:
            self.sids = sorted(h5.keys())
            self.lengths = {s: int(h5[s].attrs["n_samples"]) for s in self.sids}
            self.has_foa = all("foa" in h5[s] for s in self.sids)
            self.has_feat = all("foa_feat" in h5[s] for s in self.sids)
            self.array_type = {s: str(h5[s].attrs["array_type"]) for s in self.sids}
        if mode == "eval":
            self.items = []
            for s in self.sids:
                nf = int(np.ceil(self.lengths[s] / self.hop_s))
                for st in range(0, max(nf, 1), self.chunk_frames):
                    self.items.append((s, st))
        else:
            self.items = [(s, None) for s in self.sids]

    def __len__(self):
        return len(self.items)

    def _open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")

    def _choose_mics(self, M, idx):
        if self.mic_subset is not None:
            if isinstance(self.mic_subset, int):  # 随机（但可复现）选 k 个
                k = min(self.mic_subset, M)
                rng = np.random.default_rng(self.subset_seed + idx)
                return np.sort(rng.choice(M, k, replace=False))
            return np.array([m for m in self.mic_subset if m < M])
        if self.mode == "train" and M > self.min_mics and np.random.rand() < self.mic_drop_prob:
            k = np.random.randint(self.min_mics, M + 1)
            return np.sort(np.random.choice(M, k, replace=False))
        return np.arange(M)

    def __getitem__(self, idx):
        self._open()
        sid, st = self.items[idx]
        g = self.h5[sid]
        n = self.lengths[sid]
        n_frames_total = int(np.ceil(n / self.hop_s))
        if st is None:
            st = np.random.randint(0, max(1, n_frames_total - self.chunk_frames + 1))
        s0, s1 = st * self.hop_s, min(n, st * self.hop_s + self.chunk_samples)

        def crop(x):
            out = np.zeros(x.shape[:-1] + (self.chunk_samples,), np.float32)
            out[..., : x.shape[-1]] = x  # x 已是 [s0:s1] 切片
            return out

        mic = g["mic"][:, s0:s1].astype(np.float32)
        mics = self._choose_mics(mic.shape[0], idx)
        item = dict(mic=crop(mic[mics]), mic_pos=g["mic_pos"][()][mics].astype(np.float32))
        rows = g["labels"][()]
        if len(rows):
            rows = rows[(rows[:, 0] >= st) & (rows[:, 0] < st + self.chunk_frames)].copy()
            rows[:, 0] -= st
        item["target"] = rows_to_adpit(rows, self.chunk_frames, self.C, self.dist_scale)
        if self.use_foa and self.has_foa:
            item["foa"] = crop(g["foa"][:, s0:s1].astype(np.float32))
        if self.use_foa_feat and self.has_feat:
            Tf = self.chunk_frames * self.feat_per_label
            f = g["foa_feat"][:, st * self.feat_per_label: st * self.feat_per_label + Tf].astype(np.float32)
            ff = np.zeros((f.shape[0], Tf, f.shape[2]), np.float32)
            ff[:, : f.shape[1]] = f
            item["foa_feat"] = ff
        item["sid"], item["start"] = sid, st
        return item


def collate_pad(batch, max_mics=8):
    M = max(max_mics, max(b["mic"].shape[0] for b in batch))
    B, N = len(batch), batch[0]["mic"].shape[-1]
    mic = torch.zeros(B, M, N)
    pos = torch.zeros(B, M, 3)
    mask = torch.zeros(B, M, dtype=torch.bool)
    for i, b in enumerate(batch):
        m = b["mic"].shape[0]
        mic[i, :m] = torch.from_numpy(b["mic"])
        pos[i, :m] = torch.from_numpy(b["mic_pos"])
        mask[i, :m] = True
    out = dict(mic=mic, mic_pos=pos, mic_mask=mask, target=torch.from_numpy(np.stack([b["target"] for b in batch])),
               sid=[b["sid"] for b in batch], start=torch.tensor([b["start"] for b in batch]))
    for k in ("foa", "foa_feat"):
        if all(k in b for b in batch):
            out[k] = torch.from_numpy(np.stack([b[k] for b in batch]))
    return out
