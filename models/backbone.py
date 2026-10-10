# -*- coding: utf-8 -*-
"""SELD 骨干网络封装。

统一接口: forward(feat (B,7,Tf,n_mels), n_frames) -> dict(emb (B,T,D), out (B,T,3,4,C))
  out = multi-ACCDOA + 距离: 3 个轨道 × [x,y,z,dist] × C 类；xyz 用 tanh，dist 用 ReLU（单位 dist_scale 米）

两种实现：
  * PSELDNetsBackbone: 官方 PSELDNets (https://github.com/Jinbo-Hu/PSELDNets) 的 CNN14-Conformer
      (= 其 multi_accdoa.CRNN, encoder=CNN12, num_features=[64..2048], decoder=conformer×1)
      输入为 4ch FOA 的 64 维 log-mel + 3ch 强度向量（共 7 通道），24kHz, nfft 1024, hop 240。
      权重：HuggingFace datasets/Jinbo-HU/PSELDNets/model/mACCDOA-CNN14-Conformer-0.582.ckpt
      (Lightning ckpt，键为 'net.xxx')。我们复用其 scalar/convs/decoder 权重，丢弃 170 类的 fc，换成本任务的头。
  * StandInBackbone: 轻量 CNN+BiGRU，接口一致，随机初始化。用于冒烟测试 / 手机端蒸馏小模型 /
      无法下载权重时的替身。
"""
import os
import sys
import types
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


class SELDHead(nn.Module):
    def __init__(self, d, n_classes, n_tracks=3):
        super().__init__()
        self.C, self.K = n_classes, n_tracks
        self.fc = nn.Linear(d, n_tracks * 4 * n_classes)

    def forward(self, emb):
        B, T, _ = emb.shape
        o = self.fc(emb).view(B, T, self.K, 4, self.C)
        return torch.cat([torch.tanh(o[:, :, :, :3]), F.relu(o[:, :, :, 3:])], 3)


class ChannelScalar(nn.Module):
    """与 PSELDNets 相同：每个特征通道一个 BatchNorm2d(n_mels)，对 mel 维做归一化。"""

    def __init__(self, in_ch, n_mels):
        super().__init__()
        self.scalar = nn.ModuleList([nn.BatchNorm2d(n_mels) for _ in range(in_ch)])

    def forward(self, x):  # (B,C,T,F)
        return torch.cat([self.scalar[c](x[:, c:c + 1].transpose(1, 3)).transpose(1, 3) for c in range(x.shape[1])], 1)


def _pool_to_frames(x, n_frames):
    """x (B,T',D) -> (B,n_frames,D)（时间轴自适应平均/插值）"""
    x = x.transpose(1, 2)
    x = F.adaptive_avg_pool1d(x, n_frames) if x.shape[-1] >= n_frames else F.interpolate(x, size=n_frames, mode="linear", align_corners=False)
    return x.transpose(1, 2)


class StandInBackbone(nn.Module):
    """时间轴池化因子乘积 = 10 = 每个标签帧的特征帧数，因此输出帧率天然为 100ms，
    不需要插值/自适应池化（便于 ONNX 动态长度导出）。"""

    def __init__(self, n_classes, in_ch=7, n_mels=64, channels=(32, 64, 128), d_model=128, n_layers=2,
                 time_pools=(2, 5, 1), **kw):
        super().__init__()
        self.scalar = ChannelScalar(in_ch, n_mels)
        layers, c0 = [], in_ch
        for i, c in enumerate(channels):
            layers += [nn.Conv2d(c0, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                       nn.AvgPool2d((time_pools[i], 2))]
            c0 = c
        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(c0, d_model)
        self.rnn = nn.GRU(d_model, d_model // 2, num_layers=n_layers, batch_first=True, bidirectional=True)
        self.head = SELDHead(d_model, n_classes)
        self.emb_dim = d_model

    def forward(self, feat, n_frames):
        x = self.cnn(self.scalar(feat)).mean(-1).transpose(1, 2)  # (B,T',C)
        x = self.rnn(self.proj(x))[0]
        emb = x if x.shape[1] == n_frames else _pool_to_frames(x, n_frames)
        return dict(emb=emb, out=self.head(emb))


def _import_pseldnets(repo):
    """把 PSELDNets/src 加入路径；用空壳 'models' 包避免执行其 __init__（会导入 HTS-AT/PaSST 等重依赖）。"""
    src = os.path.join(repo, "src")
    if not os.path.isdir(os.path.join(src, "models", "components")):
        raise FileNotFoundError(f"PSELDNets repo not found at {repo}")
    if src not in sys.path:
        sys.path.insert(0, src)
    if "models" not in sys.modules or not hasattr(sys.modules["models"], "__pseldnets_stub__"):
        pkg = types.ModuleType("models")
        pkg.__path__ = [os.path.join(src, "models")]
        pkg.__pseldnets_stub__ = True
        sys.modules["models"] = pkg
    from models.components.backbone import CNN8, CNN12  # noqa
    from models.components.model_utilities import Decoder  # noqa
    return CNN8, CNN12, Decoder


class PSELDNetsBackbone(nn.Module):
    """PSELDNets CNN14-Conformer（mACCDOA 版本）封装。

    注意：本仓库的 'models' 包名与 PSELDNets 的 'models' 冲突，因此在导入 PSELDNets 时会临时替换
    sys.modules['models']；我们自己的模块已在此之前导入完毕，不受影响。
    """

    def __init__(self, n_classes, repo_path="third_party/PSELDNets", ckpt_path=None, in_ch=7, n_mels=64,
                 encoder="CNN12", num_features=(64, 128, 256, 512, 1024, 2048), decoder="conformer",
                 num_decoder_layers=1, trainable="all", **kw):
        super().__init__()
        own_models = sys.modules.get("models")
        try:
            CNN8, CNN12, Decoder = _import_pseldnets(repo_path)
        finally:
            # 恢复我们自己的 models 包
            if own_models is not None:
                sys.modules["models"] = own_models
        self.scalar = ChannelScalar(in_ch, n_mels)
        self.convs = {"CNN8": CNN8, "CNN12": CNN12}[encoder](in_ch, list(num_features))
        self.decoder = Decoder(decoder, num_features[-1], num_layers=num_decoder_layers)
        self.head = SELDHead(num_features[-1], n_classes)
        self.emb_dim = num_features[-1]
        self.loaded = False
        if ckpt_path and os.path.exists(ckpt_path):
            self.load_pseldnets_ckpt(ckpt_path)
        else:
            warnings.warn(f"[PSELDNets] 未找到权重 {ckpt_path}，使用随机初始化（TODO: 设置 backbone.ckpt_path）")
        self.set_trainable(trainable)

    def load_pseldnets_ckpt(self, path):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        sd = {k[4:] if k.startswith("net.") else k: v for k, v in sd.items()}
        new = {}
        for k, v in sd.items():
            if k.startswith("fc."):
                continue  # 170 类 mACCDOA 头丢弃
            nk = k.replace("scalar.", "scalar.scalar.", 1) if k.startswith("scalar.") else k
            new[nk] = v
        missing, unexpected = self.load_state_dict(new, strict=False)
        missing = [m for m in missing if not m.startswith("head.")]
        print(f"[PSELDNets] loaded {path}: {len(new)} tensors, missing={missing[:5]}({len(missing)}), unexpected={unexpected[:5]}({len(unexpected)})")
        self.loaded = len(missing) == 0

    def set_trainable(self, mode):
        """all: 全部微调；decoder_head: 冻结 CNN；head: 只训练头。
        TODO: PSELDNets 的 AdapterBit（src/models/components/model_utilities_adapt.py）可在此接入。"""
        for n, p in self.named_parameters():
            if mode == "all":
                p.requires_grad = True
            elif mode == "decoder_head":
                p.requires_grad = n.startswith(("decoder.", "head."))
            elif mode == "head":
                p.requires_grad = n.startswith("head.")

    def forward(self, feat, n_frames):
        x = self.convs(self.scalar(feat))  # (B,C,T/8,F/8)
        x = x.mean(dim=3).transpose(1, 2)
        x = self.decoder(x)
        emb = _pool_to_frames(x, n_frames)
        return dict(emb=emb, out=self.head(emb))


def build_backbone(cfg, n_classes):
    kind = cfg.get("type", "standin")
    kw = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "pseldnets":
        try:
            return PSELDNetsBackbone(n_classes, **kw)
        except FileNotFoundError as e:
            warnings.warn(f"{e}; 回退到 StandInBackbone（随机初始化）")
            return StandInBackbone(n_classes, **kw.get("standin", {}))
    return StandInBackbone(n_classes, **kw)
