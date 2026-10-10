# 阵列无关 SELD（Array-Agnostic Sound Event Localization and Detection）

面向 **Kaggle 2×T4（16GB, fp16, DDP, 12h 会话）** 的研究代码：任意 2~8 麦克风阵列 → 与 FOA 兼容的 4 通道表示 → 预训练 FOA SELD 骨干（PSELDNets）。

```
阵列波形 (B,M,N) + 麦克风坐标 (B,M,3) + mask
   │ STFT (hann 1024 / hop 240 @ 24kHz)
   ▼
[解析 LS-FOA 编码 E(f)]  ──┐                   （只依赖几何，逐频点正则最小二乘）
[神经残差 ΔE(t,f)]        ──┤  FOA = Σ_m (E_m + ΔE_m) X_m      （ACN: W,Y,Z,X）
   (共享 CNN + rMPE + TAC，对麦克风数量/顺序不变)
   ▼
log-mel×4 + 强度向量×3 (7×T×64，与 PSELDNets 完全一致)
   ▼
PSELDNets CNN14-Conformer 骨干（或 StandIn 小骨干）→ multi-ACCDOA + 距离 (3 轨道 × [x,y,z,d] × C)
```

训练损失 = ADPIT multi-ACCDOA(+距离) + 帧级置换不变蒸馏（教师 = 骨干吃 **理想 FOA**）+ 小权重 FOA 重建损失，权重均可配置。

---

## 1. 目录结构

```
array_agnostic_seld/
├── README.md / requirements.txt
├── configs/
│   ├── base.yaml               # 所有默认超参（含注释）
│   ├── teacher_foa.yaml        # 阶段零：理想 FOA 上训练教师骨干
│   ├── sim_pretrain.yaml       # 阶段一：仿真预训练
│   ├── starss23_finetune.yaml  # 阶段二：STARSS23 MIC 微调
│   ├── distill_small.yaml      # 阶段三：蒸馏手机端小模型
│   └── smoke.yaml              # CPU 冒烟测试
├── sim/
│   ├── dummy_sources.py        # 合成音调/噪声 dummy 事件 + sources.csv
│   ├── arrays.py               # 随机阵列：linear / circular / phone / random / tetra
│   ├── simulate.py             # pyroomacoustics 场景仿真（多进程）：阵列 wav + 理想 FOA + 几何 json + DCASE csv
│   ├── precompute.py           # 打包 HDF5（+ 理想 FOA 的 logmel-IV 特征）
│   └── prepare_starss23.py     # STARSS23 → 同一目录格式（MIC 四面体坐标）
├── seld/
│   ├── utils.py                # 配置、球谐、STARSS23 MIC 坐标
│   ├── features.py             # STFT / log-mel / IV（实数实现，可导出 ONNX）
│   ├── labels.py               # DCASE csv <-> ADPIT 目标 (T,6,5,C)
│   ├── dataset.py              # HDF5 数据集，可变麦克风数 + mask，随机麦克风子集增强
│   ├── losses.py               # ADPIT、置换不变蒸馏、FOA 重建
│   ├── metrics.py              # DCASE 风格 F20 / DOAE / RDE（帧级近似实现）
│   └── engine.py               # 训练包装（DDP 一次 forward 算完全部损失）、推理拼接、评测
├── models/
│   ├── ls_foa_encoder.py       # 解析 LS FOA 编码器
│   ├── residual_tac.py         # 残差前端：共享 CNN + rMPE + TAC
│   ├── backbone.py             # PSELDNets 封装 / StandIn 骨干 / 输出头
│   └── full_model.py           # 完整模型 + ONNX 包装
├── train.py                    # torchrun DDP + AMP + 每 epoch 断点 + 自动续训 + 12h 时间保护
├── eval.py                     # DCASE 指标 + array swap（麦克风子集 / 未见几何 / 按阵列类型）
├── export/export_onnx.py       # 小模型 ONNX 导出 + onnxruntime 数值校验（动态 M、T）
├── kaggle/array_agnostic_seld_kaggle.ipynb
├── scripts/
│   ├── smoke_test.sh               # CPU 端到端冒烟测试
│   ├── check_ls_foa.py             # LS-FOA vs 理想 FOA 分频带 NMSE、IV 方向检查
│   ├── check_metrics_oracle.py     # 指标单元检查（完美预测 → F=1）
│   └── check_pseldnets_order.py    # 用官方权重核对 mel 滤波器与 IV 通道顺序
└── third_party/PSELDNets       # git clone https://github.com/Jinbo-Hu/PSELDNets （需自行 clone）
```

## 2. 安装

```bash
pip install -r requirements.txt
git clone --depth 1 https://github.com/Jinbo-Hu/PSELDNets third_party/PSELDNets
# PSELDNets 预训练权重（~800MB，HuggingFace 数据集仓库）
python -c "from huggingface_hub import hf_hub_download as d; print(d('Jinbo-HU/PSELDNets','model/mACCDOA-CNN14-Conformer-0.582.ckpt',repo_type='dataset',local_dir='ckpts'))"
```
然后把 `configs/base.yaml` 的 `model.backbone.ckpt_path` 改为实际路径（**TODO**）。找不到权重时会告警并随机初始化；找不到 PSELDNets 仓库时自动回退到 StandIn 骨干，代码仍可运行。

### PSELDNets 对接要点（已在本机用官方权重核对）
* 官方仓库：<https://github.com/Jinbo-Hu/PSELDNets>；权重：<https://huggingface.co/datasets/Jinbo-HU/PSELDNets/tree/main/model>
* CNN14-Conformer = 其 `multi_accdoa.CRNN`（`configs/model/crnn.yaml`：encoder=CNN12, num_features=[64…2048], decoder=conformer×1），输入 7 通道 `logmelIV`，24kHz、nfft 1024、hop 240、64 mel（torchaudio MelScale, norm='slaney'）、label 100ms；170 类 mACCDOA 输出（3 轨道×3×170）。
* ckpt 是 Lightning 格式，键为 `net._orig_mod.xxx`（torch.compile 过）；我们去掉前缀后加载 `scalar/convs/decoder`，丢弃 `fc`（170 类），换成本任务的 3×4×C 头。**实测 0 个缺失键**。
* 我们的 mel 滤波器与 ckpt 内保存的 `af_extractor.mel_scale.fb` 最大误差 3.8e-8。
* **IV 通道顺序**：PSELDNets 直接用通道 1/2/3 作 IV 的三个分量。用官方权重 + 原 170 类头在我们仿真的 ACN FOA 上测 DOA：`iv_xyz_index=[1,2,3]` 中位误差 31°，`[3,1,2]` 为 112°，所以默认 `[1,2,3]`（`scripts/check_pseldnets_order.py` 可复现）。
* 为避免 import 其重依赖（lightning、HTS-AT 等），封装时用空壳 `models` 包只导入 `models.components.{backbone,model_utilities,conformer}`。
* 时间轴：官方做 ×8 上采样再按 10 帧平均；我们直接把 conformer 输出自适应池化到标签帧数（只影响时间聚合方式，不影响加载的权重）。
* HTS-AT 版本与 AdapterBit 尚未接入（TODO，见 `PSELDNetsBackbone.set_trainable`）。

## 3. 数据仿真（sim/）

```bash
# (a) dummy 源（测试用）；真实实验请准备 FSD50K 子集 csv：filename,class_idx,class_name
python -m sim.dummy_sources --out data/dummy_sources --n-per-class 40
# (b) 仿真：随机鞋盒房间 (3-10m × 3-10m × 2.5-4m, RT60 0.2-0.8s)、2-8 麦克风随机阵列、1-3 个静止声源
python -m sim.simulate --source-csv data/dummy_sources/sources.csv --out data/sim_train --n-scenes 2000 --duration 10 --workers 8
python -m sim.simulate ... --out data/sim_unseen --array-types tetra --seed 2 --prefix unseen_   # 未见几何测试集
# (c) 打包 HDF5
python -m sim.precompute --root data/sim_train --out data/h5/sim_train.h5
python scripts/check_ls_foa.py --h5 data/h5/sim_train.h5    # 诊断
```
每个场景输出 `mic/*.wav`（M 通道）、`foa/*.wav`（理想 FOA）、`geometry/*.json`（麦克风坐标，相对阵列中心）、`metadata/*.csv`（`frame,class,track,azi,ele,dist_cm`，100ms 帧，DCASE2024 格式）。

**理想 FOA 的渲染方法**：对每个声源调用 pyroomacoustics 的镜像源模型（与阵列 RIR 同一次 `compute_rir`），取全部镜像源位置与反射衰减，在阵列中心放一个“理想一阶球谐虚拟麦克风”：每个镜像源按 `damping/d` 的幅度（与 pra 的 1/d 约定一致，不含 4π）、`d/c` 的分数延迟（与 pra 相同的 81 点加窗 sinc 及 40 样点固定延迟，保证与阵列录音时间对齐）、乘以方向 `u` 的 ACN/SN3D 球谐 `[1, u_y, u_z, u_x]` 后叠加。阵列与 FOA 来自同一 ISM，描述同一个声场；FOA 不加传感器噪声（教师输入是“干净的理想 FOA”）。

注意：
* ISM 阶数被 `--max-order` 截断（默认 12），没有随机晚期混响尾部；阵列与 FOA 两边一致截断。
* 声源静止（无运动轨迹），事件起止由片段长度决定（未做静音裁剪）。
* `class_idx` 要映射到 STARSS23 的 13 类，才能与阶段二无缝衔接。
* **SpatialScaper**（arXiv 2401.12238）可作为替代/补充：它用真实测量的 SRIR 库（TAU-SRIR、METU 等）生成 DCASE 格式的移动声源场景，但 SRIR 只对 FOA/MIC(四面体) 格式测量，**无法直接给任意阵列**。建议用法：SpatialScaper 生成的 FOA/MIC 数据作为阶段一里“真实房间”的补充；任意阵列的部分仍用本仿真器。

## 4. 训练（train.py）

```bash
# 单卡 / CPU
python train.py --config configs/smoke.yaml
# Kaggle 2×T4
# 阶段零：PSELDNets 骨干 + 新 13 类头，只在理想 FOA 上训练 → 冻结教师（原 170 类头无法直接当教师）
torchrun --nproc_per_node=2 train.py --config configs/teacher_foa.yaml model.backbone.ckpt_path=/kaggle/working/ckpts/model/mACCDOA-CNN14-Conformer-0.582.ckpt
# 阶段一：前端 + 学生骨干（从阶段零初始化），教师 = 阶段零 best.pt
torchrun --nproc_per_node=2 train.py --config configs/sim_pretrain.yaml
torchrun --nproc_per_node=2 train.py --config configs/starss23_finetune.yaml   # init_from 阶段一 best.pt
torchrun --nproc_per_node=2 train.py --config configs/distill_small.yaml       # 教师 = 阶段二完整模型
```
* 任意配置项可在命令行用 `a.b.c=value` 覆盖。
* 每个 epoch 原子写入 `{out_dir}/{exp_name}/last.pt`（模型/优化器/调度器/GradScaler/epoch），验证更优时写 `best.pt`；**同一命令重跑即自动续训**。
* 会话剩余时间保护：超过 `train.max_hours`（默认 11.3h）立即存断点退出。Kaggle 新会话中 `/kaggle/working` 为空：把上个会话输出挂为 Input，并设置 `train.resume_from=/kaggle/input/.../last.pt`。
* AMP fp16：LS 编码矩阵（复数求解）、STFT、log-mel/IV 强制 fp32（STFT 功率会超过 fp16 上限）；残差网络与骨干在 autocast 下运行。
* 教师模式 `teacher.mode`：
  * `frozen_backbone`（默认）：深拷贝骨干并冻结，吃理想 FOA（与 arXiv 2609.18040 一致：教师与学生结构相同、参数独立、教师冻结）。**必须**用 `teacher.ckpt` 指向阶段零的 best.pt（从中取 `backbone.*`），否则教师的任务头是随机的，会给出告警。
  * `shared`：同一骨干吃理想 FOA（教师输出 detach），可再加 `loss.ideal_seld`；适合骨干没有预训练权重时（冒烟测试即用此模式）。
  * `full_model`：教师是另一个完整模型（吃同一阵列输入），用于蒸馏小模型。
* 显存提示：CNN14-Conformer 约 1.76 亿参数（含前端），`frozen_backbone` 再多一份冻结副本。T4 上建议先试 `batch_size=4~8, chunk_sec=5`，不够时降 `model.residual.hidden` 或调大 `freq_stride`。**我们没有在 T4 上实测显存/速度**。

### STARSS23 阶段
```bash
python -m sim.prepare_starss23 --starss /kaggle/input/starss23 --out data/starss23_train --split dev-train
python -m sim.prepare_starss23 --starss /kaggle/input/starss23 --out data/starss23_test  --split dev-test
python -m sim.precompute --root data/starss23_train --out data/h5/starss23_train.h5
```
STARSS23 MIC 格式坐标（DCASE2023/2024 Task3 文档：Eigenmike em32 的 6/10/26/22 号振膜，半径 4.2cm）：

| 通道 | 方位角 | 仰角 | r |
|---|---|---|---|
| M1 | 45° | 35° | 4.2 cm |
| M2 | −45° | −35° | 4.2 cm |
| M3 | 135° | −35° | 4.2 cm |
| M4 | −135° | 35° | 4.2 cm |

注意这些振膜在刚性球上，平面波自由场 LS 编码在高频有系统偏差（理论上应乘刚性球模态强度），残差网络会学习部分补偿；刚性球模型留作 TODO。教师输入使用 STARSS23 同步录制的 FOA。

## 5. 评测（eval.py）

```bash
python eval.py --config configs/sim_pretrain.yaml --ckpt .../best.pt \
   --h5 seen=data/h5/sim_val.h5 unseen_geom=data/h5/sim_unseen.h5 --mic-subsets all,2,3,4 --out results.json
```
* 指标：位置感知 F-score（类别正确且角度误差 ≤ 20°）、DOA 误差（度）、相对距离误差 |d̂−d|/d，以及 SELD = (1−F + DOAE/180 + min(RDE,1))/3。帧级（100ms）计算，同类多声源用匈牙利匹配，按类宏平均。这是 **近似实现**，与 DCASE 官方 `SELD_evaluation_metrics.py`（含 1s 片段/距离阈值等细节）数值不完全相同，正式对比请用官方代码。
* array swap：`--mic-subsets k` 每段随机（可复现）抽 k 个麦克风；多个 `--h5` 对比已见/未见几何；结果再按阵列类型细分。

## 6. ONNX 导出（export/）

```bash
python export/export_onnx.py --config configs/distill_small.yaml --ckpt .../stage3_distill_small/best.pt --out seld_small.onnx --n-mics 4
```
图输入：阵列 STFT 实/虚部、麦克风坐标、mask、**预先计算的 LS 编码矩阵 E**（只依赖几何，端上算一次；脚本提供等价的 numpy 实现 `compute_E_numpy` 并同时导出 `_E.npz`）。复数线性代数与 STFT 不进 ONNX 图（ONNX 对复数支持差），M 与 T 为动态轴，并用 onnxruntime 校验（M、T 改变后仍一致）。

## 7. 冒烟测试

```bash
SMOKE_DIR=/tmp/smoke bash scripts/smoke_test.sh
```
仿真 16+6 个 3s 场景 → HDF5 → 训练 2 epoch → 续训到第 3 epoch → 2 进程 DDP (gloo) → 评测（全部/2/3 麦克风）→ ONNX 导出 + 校验。

## 8. 设计说明与已知局限

* **LS 编码器**：我们用的是“方向网格上的正则最小二乘拟合球谐响应”（更接近 2609.18040 中的 ASM 基线，Gayer et al. 2024），而不是论文的“虚拟球 + 模态强度反演”常规编码器；两者都可以作为残差的初始化。
* **残差网络**：论文用 ConvGLU 编解码 + TS-Mamba；我们按需求改为 **每通道共享 CNN + rMPE + TAC**，以支持可变麦克风数/顺序。输出层零初始化，起点等于 LS 编码。频率轴内部下采样（`freq_stride`）后插值回去以省显存。
* **rMPE**：参考 AGG-RL，但我们以有效麦克风质心为参考点（AGG-RL 用离中心最近的麦克风作参考通道），并额外加入频率相关的 `cos/sin(2πf/c·r·e_xyz)` 几何特征。
* **蒸馏**：帧级置换不变（3! 轨道排列取最小）的 MSE；mACCDOA 没有独立 SED logit，因此没有逐字复现论文的 BCE + 置信加权 DOA 形式（提供 `distill_conf_thr` 的简化版）。论文也指出更低的 FOA 重建误差 ≠ 更好的 SELD，所以重建损失权重默认很小。
* 仿真无运动声源、无扩散噪声场、无麦克风增益/相位失配、无设备遮挡（手机外壳散射）——这些都是迁移到真实设备时的主要差距。
* 冒烟测试只验证“能跑通”，模型极小、数据极少，指标没有意义。

## 9. 参考文献

1. J. Hu et al., **PSELDNets**: Pre-trained Neural Networks on Large-scale Synthetic Datasets for SELD, IEEE TASLP 2025. [arXiv:2411.06399](https://arxiv.org/abs/2411.06399) · [代码](https://github.com/Jinbo-Hu/PSELDNets) · [权重/数据](https://huggingface.co/datasets/Jinbo-HU/PSELDNets)
2. J. Liu, Y. Cao, M. Wu, J. Yang, **Task-oriented neural FOA encoding for SELD from irregular microphone arrays**. [arXiv:2609.18040](https://arxiv.org/abs/2609.18040)
3. M.-S. Baek et al., **Physics-Informed Audio-Geometry-Grid Representation Learning for Universal Sound Source Localization (AGG-RL)**, ICLR 2026. [论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/ee860a9fa65a55a335754c557a5211de-Paper-Conference.pdf) · [代码](https://github.com/BaekMS/Audio-Geometry-Grid_Representation-Learning)
4. Y. Luo et al., **FaSNet: End-to-end microphone permutation and number invariant multi-channel speech separation (TAC)**. [arXiv:1910.14104](https://arxiv.org/abs/1910.14104)
5. I. Roman et al., **Spatial Scaper**: a library to simulate and augment soundscapes for SELD in realistic rooms. [arXiv:2401.12238](https://arxiv.org/abs/2401.12238)
6. A. Dementyev et al., **PhaseCoder**: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs. [arXiv:2601.21124](https://arxiv.org/abs/2601.21124)
7. K. Shimada et al., Multi-ACCDOA & ADPIT, ICASSP 2022. [arXiv:2110.07124](https://arxiv.org/abs/2110.07124)
8. DCASE2024 Task 3 (STARSS23, 含距离估计)：<https://dcase.community/challenge2024/task-audio-and-visual-sound-event-localization-and-detection-with-source-distance-estimation>
9. R. Scheibler et al., pyroomacoustics, ICASSP 2018. <https://github.com/LCAV/pyroomacoustics>
