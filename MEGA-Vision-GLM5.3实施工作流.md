# 「慧眼 MEGA-Vision」项目实施工作流
## 供 ZCode GLM5.3 在 Linux + NVIDIA GPU 环境逐阶段执行的完整方案

---

## 修订记录

| 版本 | 变更 |
|---|---|
| v1.0 | 初版：九阶段工作流 |
| v1.1 | 第一轮审查：修正组件/数据链接与验收阈值（见技术可行性验证报告） |
| v1.2 | 第二轮审查：10 项原理纠错 + 步骤优化 + 学习效率（见原理分析与工作流纠错报告） |
| v2.0 | 第三轮：并入前瞻性优化报告全部建议。P0 强制：可执行 Gate、基线金丝雀、按国家域外评估、医学双通道、gap 方差-偏差分解与协方差原型；P1 标准：掩码蒸馏、自适应切片、密度图计数、特征缓存/coreset/渐进分辨率、细粒度指标、数据飞轮；P2 可选：底座分档、扩散增广、部署硬化。正文以【P0 强制】【P1 标准】【P2 可选】标注 |
| v2.1 | 环境授权模型：三环授权——硬约束不可改 / 软约束钉死默认、备案偏离 / 自由区全权；执行纪律第 4 条与失败决策树同步 |

---

## 第 0 部分：给 GLM5.3 的执行总则（必读，先读这里）

**你的角色**：你是一名资深计算机视觉算法工程师。你的唯一任务是把本文件描述的「慧眼 MEGA-Vision」项目从零实现到最终验收。本文件是自包含的：所有算法设计、环境要求、数据方案、实现步骤、验收标准都在本文件内，不要依赖任何外部对话上下文。

**硬性环境前提（不满足必须先停下报告，不得继续）**：
- 操作系统：Linux（Ubuntu 20.04/22.04，x86_64）
- GPU：NVIDIA 显卡，显存 ≥ 24 GB（RTX 3090 / 4090 / A100 / L40S 均可）
- 不得在无 GPU、无 CUDA 的环境下"假装训练"——任何训练步骤必须在 GPU 上实际执行并记录显存占用与耗时

**执行纪律（每一条都必须遵守）**：
1. 本文件 = 项目宪法。每个 Phase 开始前，先重读本文件对应小节，再把该节内容复制为你的工作清单。
2. **门禁机制（Gate，代码裁决）**【P0 强制】：每个 Gate 都对应一个验收测试文件 `tests/gates/test_gate_N.py`（内含指标计算、阈值断言、坏例画廊生成）。通过标准 = **运行 `pytest tests/gates/ -v` 全部通过**——达标与否由代码裁决，禁止用模型自述代替；不通过 → 按第 8 部分失败决策树处理，最多修复重试 3 轮。
3. **基线金丝雀（Canary）**【P0 强制】：每个 Phase 开始前先运行 `scripts/baseline_canary.py --phase N`，复现该 Phase 的已知基线数字（如 Phase 3：YOLOv8 在 GWHD 验证集 AP@50 达到文献值 ±1 pp；Phase 4：SAM 2.1 零样本在 SIIM 的 Dice 达到文献值 ±2 pp）。基线复现失败 = 环境/数据/解析存在 bug，**禁止开始本 Phase**，先修环境。
4. **禁止跳步、禁止静默降级**：任何验收标准未达成就继续下一阶段的行为都是违规。若客观条件（显存、网络、数据集）导致无法达标，必须在阶段报告中显式记录"未达标 + 原因 + 已采用的降级方案"。对钉死版本/默认配置的任何偏离（含环境版本）同样必须按 3.2 授权规则备案，静默更换视为违规。
5. **Git 管理**：每个 Phase 通过后提交一次 git commit（信息格式：`phase-N: <本阶段完成内容>`）。
6. **可复现与断点续训**：所有训练脚本必须带固定随机种子、支持 `--resume` 与 `--logdir`；一切中间产物（缓存特征、伪掩码、few-shot 划分、检查点）统一放 `artifacts/` 并在 `artifacts/manifest.json` 登记来源与哈希——中断后从 manifest 恢复，禁止重跑已完成步骤；关键实验记录训练命令、显存峰值、耗时、最终指标。
7. **时间预算**：各 Phase 有单卡 GPU 时间预算（见第 6.1 节）；实际耗时超过预算 2 倍仍未达 Gate 的，强制触发备选路线或降配方案，不得无限重试。
8. **汇报**：每个 Gate 通过后，输出一份简短阶段报告（做了什么 / 指标 vs 通过标准 / 遗留问题），最终汇总为总验收报告。

---

## 第 1 部分：项目命名与总体目标

### 1.1 项目命名

| 项 | 名称 |
|---|---|
| 项目中文名 | **「慧眼」多专家协同智能图像识别平台**（寓意：洞察复杂场景中的微小关键信息） |
| 项目代号 | **MEGA-Vision**（Multi-Expert Grounded Adaptive Vision Platform） |
| 核心算法 | **MEGA-Net**（Multi-Expert Grounded Adaptive Network，多专家协同自适应识别网络） |
| 代码仓库名 | `mega-vision` |
| 项目口号 | 一眼见微，万物可识 |

### 1.2 八大目标（本工作流的"北极星"，全部必须达成）

| # | 目标 | 一句话验收标准 |
|---|---|---|
| 1 | 创新图像识别算法 | MEGA-Net 六项创新点（见 2.2）在代码中全部落地 |
| 2 | 适用特殊领域 | 农业（菜园航拍逐株识别）与医学（胸部 X 光）两个示范域跑通 |
| 3 | 参考国际最新设计 | 采用 DINOv2/DINOv3、DEIM(CVPR2025)、SAM 2.1 等 2023–2025 SOTA 组件 |
| 4 | 高精度/高鲁棒/多目标检测+分类+分析 | 检测、分类、分析三级结果齐全，指标达第 7 部分阈值，鲁棒性压测达标 |
| 5 | 小样本与海量样本结果基本一致 | 5-shot 与全量训练性能差 ≤ 5 个百分点（Phase 7 协议强制验证） |
| 6 | 复杂场景识别关键信息 | 航拍图逐株检出、X 光微小异常检出（含 SAHI 切片 + 细节支路） |
| 7 | 模型功能取决于样本类别 | 同一套代码，只换数据集与任务头即可在农业/医学间切换 |
| 8 | 多模型协同 | 级联路由 + 异构集成 + 知识蒸馏三机制全部实现并验证 |

---

## 第 2 部分：算法设计摘要（自包含，直接照此实现）

### 2.1 总体架构（四级流水线 + 协同层）

```
                    ┌──────────────────────────────────┐
 输入：可见光影像     │  视觉底座（冻结）                 │
 （+可选高光谱立方体）│  DINOv2 ViT-L/14 主干            │
                    │  + 高分辨率细节支路（ConvNeXt 浅层）│
                    └────────────────┬─────────────────┘
                                     │ 跨尺度注意力融合 → 特征金字塔 {P2,P3,P4,P5}
                    ┌────────────────▼─────────────────┐
  Phase 3 检测专家   │  DEIM/DINO 风格 query-based DETR   │
  （多目标检测）      │  + 对比去噪训练 + 密集一对一匹配    │
                    │  + SAHI 高分辨率切片推理            │
                    └────────────────┬─────────────────┘
                                     │ 检测框 B + 置信度 s
                    ┌────────────────▼─────────────────┐
  Phase 4 实例专家   │  SAM 2.1 框提示分割                │
  （实例级理解）      │  → 像素级掩码 M + 掩码质量自评头    │
                    └────────────────┬─────────────────┘
                                     │ 逐实例 ROI 特征 + 掩码统计
                    ┌────────────────▼─────────────────┐
  Phase 5 识别专家   │  原型度量分类头 + 领域分析头        │
  （分类与分析）      │  （农业：株数/出苗率/生长阶段；      │
                    │   医学：异常评分/热力图/报告要素）    │
                    └────────────────┬─────────────────┘
                                     │ 类别/属性/分析结论
                    ┌────────────────▼─────────────────┐
  Phase 6 协同层     │  难度路由（轻/重专家链）            │
  （多模型协同）      │  + 异构专家温度标定集成             │
                    │  + 知识蒸馏到边缘小模型              │
                    └──────────────────────────────────┘
```

### 2.2 六个创新点（代码中必须全部体现）

1. **冻结底座 + 可插拔任务头**：同一框架只换任务头与数据集即可切换领域（目标 7 的架构化实现）。
2. **原型度量 + 基础模型特征的小样本—海量一致性机制**：低参度量头 + 冻结底座压缩过拟合，配合原型 EMA 在线更新与主动学习，并用"5/10/50-shot vs 全量"验证协议作硬性闸门（目标 5）。
3. **检测框"接地"到 SAM 2.1 的级联实例理解**：DETR 框输出直接作为 SAM prompt 生成像素级掩码，附加掩码 IoU 自评头切断错误级联（目标 6 的核心手段）。
4. **不确定性门控的多专家路由协同**：场景复杂度路由器动态调度轻/重专家链，异构专家温度标定加权集成（目标 8）。
5. **细节支路 + SAHI 切片推理的微小目标专项增强**：CNN 高分辨率细节特征与 ViT 全局语义跨尺度注意力融合（目标 6）。
6. **光谱—可见光跨模态交叉注意力融合**（Phase 9 可选）：RGB 特征与光谱 Transformer 谱特征实例级对齐融合，支撑作物生长/胁迫分析（目标 2 的农业扩展）。

### 2.3 技术选型总表（全部为成熟开源组件）

| 模块 | 选型 | 来源 |
|---|---|---|
| 视觉底座 | DINOv2 ViT-L/14（备选 DINOv3） | `facebookresearch/dinov2`、`transformers` |
| 检测头 | DEIM（CVPR 2025，DETR 系 SOTA）；备选 Ultralytics RT-DETR | `https://github.com/Intellindust-AI-Lab/DEIM` |
| 实例分割 | SAM 2.1（Hiera-Large） | `https://github.com/facebookresearch/sam2` |
| 切片推理 | SAHI | `pip install sahi` |
| 细节支路 | ConvNeXt-Tiny 前 3 stage | `torchvision` |
| 参数高效微调 | LoRA | `pip install peft` |
| 光谱支路 | SpectralFormer 简化版 + 交叉注意力 | 自研（参考 UniHSFormer-X 思路） |

---

## 第 3 部分：环境与基础设施（Phase 0 的目标）

### 3.1 硬件与系统要求

- Linux x86_64（Ubuntu 20.04/22.04），NVIDIA 驱动 ≥ 535，CUDA ≥ 12.1，cuDNN ≥ 8.9
- GPU 显存 ≥ 24 GB；RAM ≥ 64 GB；磁盘 ≥ 1 TB（数据集 + 权重）
- 网络：可访问 GitHub / Hugging Face / Kaggle（国内环境需先配置镜像，见 3.4）

### 3.2 环境安装命令（默认组合，允许备案偏离）

**授权规则（三环）**：
- **硬约束（不可改）**：Linux x86_64 + NVIDIA 显存 ≥ 24GB、Python 3.10、驱动 ≥ 535 / CUDA ≥ 12.1、Gate 0 必须全绿、权重必须使用下方官方直链并校验哈希——违反即停。
- **软约束（默认钉死，可备案偏离）**：torch 2.4.1+cu121 / torchvision 0.19.1、依赖包清单、conda 环境名 `mega`。允许偏离，但必须：① 在阶段报告登记"偏离原因 + 新版本 + 复验结果"；② 重跑 Gate 0 全绿；③ 更换 torch/sahi/transformers/timm 任一关键件后，重跑对应 Phase 的基线金丝雀。
- **自由区（全权，无需备案）**：镜像源选择（国内网络可用 hf-mirror.com / ghproxy 等）、安装目录、conda/uv 工具、下载工具、CUDA 小版本——产物仍须登记 `artifacts/manifest.json`。

下方命令是**已实测验证的默认组合：默认照此执行；偏离按上述规则备案，静默更换视为违规**。

```bash
# 1) 创建环境
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

conda create -n mega python=3.10 -y
conda activate mega

# 2) PyTorch（CUDA 12.1 版）
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

# 3) 核心依赖
pip install timm transformers peft sahi pycocotools albumentations \
            opencv-python-headless scikit-learn scipy pandas matplotlib \
            tensorboard einops pyyaml tqdm

# 4) 拉取上游仓库（Phase 3/4 使用）
git clone https://github.com/Intellindust-AI-Lab/DEIM.git
git clone https://github.com/facebookresearch/sam2.git

# 5) 预训练权重（官方直链，已验证可下载）
wget -P weights/ https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget -P weights/ https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 3.3 Gate 0：环境验收门

| 验证命令 | 通过标准 |
|---|---|
| `nvidia-smi` | 显示 NVIDIA GPU，驱动 ≥ 535 |
| `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `True`，CUDA 12.x，GPU 名称正确 |
| `python -c "import torch; x=torch.randn(1024,1024,device='cuda'); print((x@x).sum().item())"` | 无报错，矩阵乘法完成 |
| `python -c "import timm, transformers, sahi, peft; print('ok')"` | 输出 `ok` |

---

## 第 4 部分：仓库结构（创建以下目录树）

```
mega-vision/
├── README.md                 # 项目说明、命名、架构图、快速开始
├── docs/
│   ├── DESIGN.md             # 本文件完整副本（作为仓库内宪法）
│   └── reports/              # 各阶段报告 phase-0.md ... phase-8.md
├── configs/
│   ├── base.yml              # 公共配置（路径、种子、设备）
│   ├── mega_gwhd.yml         # 农业检测（Phase 3）
│   ├── mega_cxr14.yml        # 医学检测（Phase 5）
│   ├── mega_fewshot.yml      # 小样本协议（Phase 7）
│   └── mega_hsi.yml          # 高光谱（Phase 9，可选）
├── src/
│   ├── backbone/             # Phase 2：底座
│   │   ├── mega_base.py      #   DINOv2 + 细节支路 + 跨尺度融合 → 特征金字塔
│   │   └── detail_branch.py  #   ConvNeXt 细节支路
│   ├── detector/             # Phase 3：检测专家
│   │   ├── train_det.py      #   训练入口（DEIM 风格，含对比去噪开关）
│   │   ├── infer_det.py      #   单图推理
│   │   ├── infer_sahi.py     #   密度引导自适应切片推理（A2）
│   │   ├── density_branch.py #   密度图计数分支（A4）
│   │   └── mask_branch.py    #   内置掩码分支（A1，Phase 4 蒸馏回灌）
│   ├── segmenter/            # Phase 4：SAM 2.1
│   │   ├── sam_prompt.py     #   框提示 → 掩码
│   │   └── mask_iou_head.py  #   掩码质量自评头
│   ├── heads/                # Phase 5：识别与分析
│   │   ├── prototype_head.py #   协方差加权原型分类头（EMA 更新，Mahalanobis）
│   │   ├── global_probe.py   #   整图全局探测通道（医学弥漫性病种）
│   │   ├── agri_analysis.py  #   农业分析头（密度融合计数/出苗率/生长阶段）
│   │   └── med_analysis.py   #   医学分析头（双通道融合/异常评分/热力图/报告要素）
│   ├── collab/               # Phase 6：协同层
│   │   ├── router.py         #   难度路由器（轻/重路径）
│   │   ├── ensemble.py       #   温度标定异构集成
│   │   └── distill.py        #   知识蒸馏
│   ├── data/
│   │   ├── registry.py       #   数据集注册与缓存
│   │   ├── fewshot_split.py  #   5/10/50-shot 划分器（种子固定）
│   │   └── augment.py        #   增强（Copy-Paste、光照扰动等）
│   ├── eval/
│   │   ├── metrics.py        #   AP/mAP/AUC/ECE/计数误差
│   │   ├── robustness.py     #   鲁棒性压测套件
│   │   └── consistency.py    #   小样本一致性协议（Phase 7）
│   ├── hsi/                  # Phase 9：光谱支路
│   │   ├── spectralformer.py
│   │   └── fusion.py         #   光谱-可见光交叉注意力
│   └── pipeline.py           # 端到端串联：检测→分割→分类→分析→协同
├── scripts/
│   ├── download_datasets.sh  # 数据集下载（见 5.1）
│   ├── baseline_canary.py    # 基线金丝雀（每 Phase 前置健康检查，D2）
│   ├── country_split.py      # GWHD 按国家域外划分（B1）
│   ├── export_onnx.py        # 边缘导出
│   └── demo.py               # 演示入口（单图/目录/视频流）
├── tests/                    # 单元测试（每个模块至少 1 个）
│   └── gates/                # Gate 验收测试 test_gate_0.py ~ test_gate_8.py（代码裁决达标，D1）
└── requirements.txt
```

---

## 第 5 部分：九阶段实施工作流（每个 Phase = 目标 + 步骤 + Gate）

### Phase 0：环境搭建与仓库初始化
- **目标**：通过 Gate 0，创建仓库骨架，`README.md` 写入命名与架构图。
- **步骤**：执行 3.2 命令 → 创建 3.3 目录树 → `git init` 并首次提交。
- **Gate 0**：见 3.3 表格。通过后写 `docs/reports/phase-0.md`。

### Phase 1：数据管线
- **数据集（按序获取，每个都必须能实际下载解压）**：
  1. **GWHD（Global Wheat Head Detection 2021）**——麦穗头俯视航拍密集小目标数据集（6,522 张 1024×1024 图 / 275,187 个麦穗头标注），作为"菜园逐株检测"的代理基准：官方站 `https://www.global-wheat.com/`（Zenodo 分发）；备用镜像 Hugging Face `caoying/GlobalWheatHeadDataset2021`；AIcrowd 挑战站亦有分发。
  2. **ChestX-ray14**——NIH 胸部 X 光 14 病种（112,120 张）：官方 `https://nihcc.app.box.com/v/ChestXray-NIHCC`；网络受限时用 GCS 公开桶 `gs://gcs-public-data--healthcare-nih-chest-xray/png/`；标签表 `Data_Entry_2017_v2020.csv`。
  3. **SIIM-ACR Pneumothorax**（Kaggle 竞赛 `siim-acr-pneumothorax-segmentation`，气胸分割，验证微小异常分割）
  4. **TBX11K**（肺结核，稀有病种 few-shot 用）：官方仓库 `https://github.com/yun-liu/Tuberculosis`（内含 Google Drive / 百度网盘链接）；备用 Kaggle 镜像 `usmanshams/tbx-11`。注意：官方只公开 train/val 标注（test 集标注需 CodaLab 挑战注册），本项目只用 train+val 自划分。
  5. **WHU-Hi 高光谱**（Phase 9 用）：`http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm`
  6. **菜园自建集**（若用户提供标注）：按 `datasets/orchard/{images,labels,masks}/` 规范放置，标注格式 COCO JSON。
- **数据目录规范**：全部放 `datasets/<NAME>/`；`src/data/registry.py` 提供统一读取接口；所有图像登记到 `datasets/index.csv`（路径/来源/用途）。
- **标注完整性抽查**：每个数据集随机 50 图，把框/掩码渲染到图上存入 `docs/reports/phase-1-viz/`，人工目检确认无错位后才能过 Gate 1（标注解析错误会污染后续所有 Phase）。
- **自建菜园标注规范（供掩码子集与生长阶段使用）**：掩码标注 ≥ 100–200 株（COCO 格式）；生长阶段四分类（苗期/营养期/开花期/成熟期）标注 ≥ 500 张、每图只取主体阶段标签；两类标注均登记到 `datasets/orchard/`。
- **按国家域外划分**【P0 强制】：用 GWHD 自带 `Metadata.csv`（16 机构/12 国家）生成"留出国家"划分：`scripts/country_split.py` 以固定种子留出 2–3 个国家作为**域外测试集**，训练/验证用其余国家；该划分供 Phase 8 跨国家泛化评估使用，划分清单写入 `artifacts/manifest.json`。
- **few-shot 划分器**：`fewshot_split.py --k 5|10|50` 输出固定种子（42）的支持集/查询集/剩余集清单，保证可复现。
- **增强集**：`augment.py` 实现 Mosaic、Copy-Paste、亮度/对比度/模糊/噪声扰动（Phase 8 鲁棒性压测复用）。
- **Gate 1**：`python -m tests.test_data` 全部通过；输出每数据集统计报告（图像数/标注数/类别分布/目标尺寸直方图，写入 `docs/reports/phase-1.md`）。**通过标准：GWHD 6,522 图与 CXR14 ≥ 5 万图实际就位，且小目标（<16px）占比 > 10% 的统计被明确写出。**

### Phase 2：视觉底座（创新点 1、5 的基座）
- **实现**（`src/backbone/`）：
  1. `mega_base.py`：加载 DINOv2 ViT-L/14 官方权重（`weights/dinov2_vitl14_pretrain.pth`，torch.hub 的 `dinov2_vitl14` 入口）并**冻结**。**输入分辨率固定 518×518（DINOv2 原生训练分辨率：位置编码零插值、零损失）**——严禁直接喂 1024 原图（需要插值位置编码，质量下降且前向成本高约 4 倍）；底座前向时把 1024 图 resize 到 518、得到 37×37 patch token，再用转置卷积/双线性插值把 token 网格上采样对齐到 1024 域（1/4 = 256² 网格），按 ViTDet/Frozen-DETR 方式构建特征金字塔 `{P2(1/4), P3(1/8), P4(1/16), P5(1/32)}`（`F.interpolate` 到目标网格即可，无需整数倍；可直接参考 Frozen-DETR 源码）；
  2. `detail_branch.py`：`torchvision.convnext_tiny` 的 **stem + stage1（输出 stride = 4，即 1/4 分辨率）**（冻结），直接吃 1024 原图输出 256² 细节特征——注意 ConvNeXt stage2/stage3 的 stride 分别是 8/16，"前 3 个 stage"会把细节特征降成 1/16，与"1/4 细节"的设计意图矛盾，只取 stem+stage1；
  3. 跨尺度注意力融合：细节特征与底座 1/4 特征做交叉注意力后拼接，与 `{P3,P4,P5}` 共同输出特征金字塔。
- **验证实验**：在 CXR14 子集做 5-shot 线性探测（正常/异常二分类）：特征 = 底座全局 token 平均池化；分类器 = 逻辑回归（sklearn）。
- **Gate 2**：线性探测 5-shot 二分类 AUC ≥ 0.70；融合金字塔各尺度特征形状自检通过（写死预期形状断言）。不达标 → 先检查底座权重加载与冻结是否正确；仍不达标 → 在 CXR14 未标注图像上做一轮继续自监督预训练后重测——文献证明该操作能让 DINOv2 在 CXR 上反超 CheXzero，是医学域达标的关键增强。
- **医学底座域适配（本 Phase 完成，不留给 Phase 5）**：在 CXR14 未标注图像（约 11 万张）上对底座做**轻量继续自监督预训练**：输入 224、MAE 或 DINO 自蒸馏协议、仅 1 个 epoch（约 875 iters）、编码器用 LoRA 或全量小学习率更新，单卡约 4–6 小时。产出医学适配底座 `weights/mega_dinov2_cxr_adapted.pth`，Phase 5 直接使用。
- **底座策略分档**【P2 可选，默认仍为冻结档】：`configs/base.yml` 增加 `backbone.regime: frozen|lora|full` 三档，按标注规模自动选择——<1k 标注 → `frozen`；1k–10k → `lora`；>10k → `full`（full 档建议改用 DINOv3 权重）；人工可覆盖。Phase 7 报告中必须给出"档位选择 vs 性能"对照表。

### Phase 3：检测专家（目标 3、4、6 的核心）
- **主路线**：DEIM 官方仓库 `https://github.com/Intellindust-AI-Lab/DEIM`（CVPR 2025，基于 RT-DETR 代码体系，config 驱动，支持注册自定义 backbone）——在其中注册 MEGA 底座（`mega_base.py`），开启**对比去噪训练**与**密集一对一匹配/匹配感知损失**；底座冻结、检测头与融合层可训练。**直接参考实现**：Frozen-DETR（NeurIPS 2024，`https://github.com/iSEE-Laboratory/Frozen-DETR`，Apache-2.0）——它就是"冻结 DINOv2 ViT-L + DETR 头"的现成范式（COCO 上为 DINO 检测器带来 +4.8 AP），其 patch-token 融合代码可直接照抄。
- **备选路线**（主路线受阻时启用，须在报告中注明）：Ultralytics RT-DETR 用 GWHD 微调作为检测专家。
- **密度图计数分支**【P1 标准】：（`density_branch.py`）检测头旁增加轻量密度图回归分支（TasselNet/CSRNet 式，点标注监督由 GWHD 框中心自动生成，高斯核 σ 按框尺寸自适应）；输出密度图服务两件事：① 农业株数统计的密度积分（Phase 5 融合）；② 自适应切片策略（见下）。与检测头联合训练，无需单独数据。
- **内置掩码分支（预留）**【P1 标准，训练在 Phase 4 收尾执行】：`mask_branch.py` 为每个 DETR query 增加轻量掩码分支（Mask DINO 式 query→mask）；其训练依赖 Phase 4 产出的误差感知微调 SAM 2.1 教师，因此掩码蒸馏在 Phase 4 收尾完成（见 Phase 4），本 Phase 只预留分支代码与损失接口。
- **SAHI 接入注意**：sahi 不原生支持 DEIM 模型，按其自定义模型接口（DetectionModel 子类封装 DEIM 推理）接入切片预测，或直接用 sahi 的切片工具函数自行拼接。
- **训练（两段式，防收敛不足）**：DETR 系检测头从零训练在小数据集上收敛慢（GWHD 训练集仅 3,422 图），30 epochs 从零训头大概率不达 0.90——采用两段式：
  1. **通用头初始化**：先在 COCO train2017（约 118k 图，约 20GB）上用冻结底座 + 可训练检测头训 12 epochs（Frozen-DETR 协议，其工作已证明冻结 DINOv2 特征 + DETR 头 12 epochs 即可收敛到可用水平），4090 单卡约 1 天；通过标准：COCO val AP ≥ 35（宽松门槛，只为获得通用检测先验）。【P1 标准】可采用渐进分辨率（640 训 8 epochs → 1024 训 4 epochs，省约 40% 时间），但必须先做 A/B 确认精度无损失后才采用；
  2. **GWHD 域微调**：加载上述检测头，GWHD 30 epochs 微调。
  COCO 下载/时间不可行时的替代方案：GWHD 直接训 90 epochs（`epochs=90`），报告中必须记录该偏离。
  超参：`lr=1e-4(head)/1e-3(neck)`、`batch=16`（冻结底座 + AMP 下 24GB 卡可开）、`ema=true`、分辨率 1024×1024、AMP。记录显存峰值与每 epoch 耗时。
- **自适应切片推理**【P1 标准】：`infer_sahi.py` 实现**密度引导的稀疏切片**——先用密度图分支前向一次得到密度图，密集区 slice 1024/overlap 0.2，稀疏区 slice 2048/overlap 0；逐片推理 → NMS 合并输出全图结果 JSON。slice 尺寸与训练输入同构（1024）；底座按 Phase 2 决议在 518 分辨率前向、特征插值回 1024 网格。验收：自适应 vs 均匀切片的 AP@50 差 ≤ 0.5 pp 且计算量节省 ≥ 25%。
- **Gate 3**：GWHD 验证集 **AP@50 ≥ 0.90**；SAHI 全图推理与单图推理结果一致性测试通过（IoU ≥ 0.95 的框占比 > 90%）；小目标（<16px）召回率相比"未开 SAHI 单尺度推理"**提升 ≥ 10%**（量化创新点 5 的收益）。文献上 GWHD top 成绩 AP@50 ≈ 93.7%，0.90 属可达但偏紧的目标；若冻结底座差 1–2 个百分点，允许对底座开启 LoRA（rank=16）重训，报告中必须记录"冻结 vs LoRA"两种设置的对比。【P1 标准】另报告 mAP@50:95、AR@50 与检测置信度校准（detection ECE）；每轮评估生成坏例画廊（FP/FN top-20 渲染图）存档 `docs/reports/phase-3-badcases/`。

### Phase 4：实例分割专家（创新点 3）
- **并行性**：本 Phase 只依赖 Phase 2 与数据集（误差感知微调用 GT 框 + SIIM 掩码，不依赖 Phase 3 的检测框），双卡环境可与 Phase 3 并行执行；单卡则先 Phase 3 后 Phase 4。
- **实现**：`sam_prompt.py` 加载 `sam2.1_hiera_large`（官方权重直链 `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`），以 Phase 3 检测框为 box prompt 输出掩码；`mask_iou_head.py` 实现轻量 IoU 自评头（输入 = ROI 图像 + 掩码，输出预测 IoU，L1 损失）。
- **误差感知微调（必做步骤，非可选）**：文献（WheatSAM，CRV 2025）证明 SAM 对"检测器生成的框"很敏感，直接零样本在麦穗类密集小目标上会退化——必须对 SAM 2.1 做 LoRA 微调，且微调时对 box prompt 注入受控扰动（随机平移 ±20–50px、缩放 ±20–50%、重叠 0.1–0.3，扰动概率 0.2–0.8）模拟检测误差。
- **微调数据（重要更正：GWHD 没有掩码标注）**：GWHD 2021 只提供检测框（CSV），没有 GT 掩码——掩码监督数据必须来自：① SIIM-ACR 气胸真实掩码（约 2,600 张含掩码）；② 自建菜园掩码子集（Phase 1 人工标注 100–200 株）；③ 用 SAM 2.1 对 GWHD GT 框生成的伪掩码（自训练，可加扰动）。IoU 自评头的训练数据用 ①+②。
- **掩码蒸馏回灌**【P1 标准】：以本 Phase 微调好的 SAM 2.1 为教师，对 Phase 3 预留的内置掩码分支（`mask_branch.py`）做蒸馏训练（监督：GWHD 伪掩码 + SIIM 真掩码；损失：掩码 BCE/Dice + 特征对齐），产出 `weights/mask_branch_distilled.pth`。此步完成后 SAM 2.1 在推理中退居"困难样本专家"（Phase 6 路由决定），常规样本走内置掩码分支。
- **Gate 4**：掩码 IoU ≥ 0.80（在 **SIIM 验证集 + 自建菜园掩码子集**上评估；文献基准 Dice≈89.7% 折合 IoU≈0.81）；自评头预测 IoU 与真实 IoU 的 **MAE ≤ 0.10**；GWHD 侧用代理指标验证：分割后检测 AP@50 相比纯检测不下降（±1 pp 内）、掩码框内面积比均值落在 0.6–1.0；演示图（检测框 + 掩码 + 自评分数叠加渲染）保存到 `docs/reports/phase-4-viz.png`。【P1 标准】内置掩码分支（蒸馏后）与 SAM 2.1 教师在 SIIM/自建子集上的掩码 IoU 差距 ≤ 2 pp。

### Phase 5：识别与分析头（目标 4、7）
- **原型分类头**（`prototype_head.py`）：特征 = 底座 ROI 特征 + 掩码统计（面积/长宽比/光谱均值）；类原型 = 支持集特征均值，**在线 EMA 更新**；预测 = 特征与各类原型的归一化距离 softmax。支持 1–10 shot 建原型与全量建原型两种模式。**适用边界：该 softmax 公式只适用于单标签任务（农业品种/杂草分类等）。CXR14 是多标签数据集（一张片可同时有多种病灶），互斥 softmax 在原理上不成立——医学头必须实现为 14 个独立 one-vs-rest 二分类（每病种一个二元原型/线性探测 + 阈值校准），输出多标签概率向量。**【P1 标准】原型本体升级为协方差加权**：类原型 = 特征精度加权均值 + 每类特征协方差（Mahalanobis 距离代替余弦距离），并输出原型置信区间；低置信实例自动进入主动学习队列。
- **农业分析头**【P1 标准】（`agri_analysis.py`）：株数统计 = **密度图积分与检测计数的一致性融合**（两者差值超过阈值时输出"计数不可信"标记）、出苗率（株数/穴数）、生长阶段四分类（苗期/营养期/开花期/成熟期，回归 + 分类双输出）、缺苗/杂草标记。
- **医学分析头（全局 + ROI 双通道）**【P0 强制】（`med_analysis.py` + `global_probe.py`）：**全局通道** = 整图 CLS token 的 14 个 one-vs-rest 探测（负责弥漫性表现：胸腔积液/心脏肥大/肺气肿等，这些病种没有边界清晰的病灶 ROI）；**ROI 通道** = 检出病灶区域的 14 个 one-vs-rest 探测（负责局灶性表现：结节/肿块/气胸等）；两通道概率温度标定后融合。异常评分 = 融合后病种概率最大值，整体异常判定 = 任一病种概率超过校准阈值；病灶热力图（ROI 上 Grad-CAM，按病种生成）；结构化报告要素 JSON（病灶位置/大小/病种/置信度/建议复核标记）。
- **训练**：CXR14 多病种分类（14 病种 + 正常，多标签 one-vs-rest，全局 + ROI 双通道）；分别跑"few-shot 头"（每病种 10 张支持集）与"全量头"两套配置。底座使用 Phase 2 产出的医学适配权重 `weights/mega_dinov2_cxr_adapted.pth`——文献显示纯自然图 DINOv2 线性探测在 CXR 上约 0.75–0.79 AUC，CXR 继续预训练后可达 RAD-DINO 级（NIH CXR8 约 0.85），这是 Gate 5 达标的推荐路径；"全量头"应为可训练 MLP（原型头仅用于 few-shot 档）。【P1 标准】**特征缓存全面化**：底座对全数据集只前向一次，光度级增强在特征域完成，分类/原型/分析头/主动学习全部在缓存特征上训练（几何级增强仍须在线前向，此边界写入代码注释）。
- **稀有类增广**【P2 可选】：对 CXR14 稀有类（如 Hernia）尝试扩散模型（med-DDPM 类）生成式增广；仅当"生成样本加入支持集前后 AUC 对比"显示改善才采用，否则弃用并记录结论。
- **农业分析头验证（更正：GWHD 没有生长阶段标签）**：GWHD 只有检测框 GT，无法标注穗生长阶段——农业验证改用可直接从 GWHD GT 计算的任务：**株数统计误差 < 5%**（掩码连通域计数 vs GT 框数）；生长阶段四分类改用自建菜园数据集（Phase 1 按规范标注 ≥ 500 张）或公开作物阶段数据集（如 DiaMOS/CGSP），数据就位后再验证。
- **Gate 5**：CXR14 多病种 mean AUC ≥ 0.80，其中**弥漫性病种子集（积液/心脏肥大/肺气肿）AUC ≥ 0.75**【P0 强制】、**稀有类（Hernia 等）单独报告、不参与平均**；输出端到端样例报告（对 3 张样例 X 光片生成 JSON 诊断要素）+ 坏例画廊（FN/FP top-20）；农业侧 GWHD 株数统计误差 < 5%（**启用密度图融合后收紧为 < 3%，密集子集单独报告**【P1 标准】）；生长阶段分类（自建/公开数据）准确率 ≥ 90%。

### Phase 6：多模型协同层（目标 8，创新点 4）
- **难度路由器**（`router.py`）：输入 = 目标密度、平均掩码自评分、检测置信度熵、图像复杂度（边缘密度），小 MLP 输出轻/重路径决策；阈值在验证集上校准（约束：重路径样本召回 ≥ 99% 的"最终被确认为困难"样本）。【P1 标准】轻路径 = 内置掩码分支（A1 蒸馏产物，单模型单前向），重路径 = SAM 2.1 级联——常规样本不触发 SAM，困难样本才外挂。
- **异构集成**（`ensemble.py`）：集成成员必须是**架构异构**的三链——① MEGA 专家链（DINOv2 底座 + DEIM 头）、② RT-DETR 域微调（备选路线产物）、③ DEIM 仓库 D-FINE 官方配置域微调；**不同随机种子的同架构模型是同质成员，不能称为"异构"**。检测框用 **WBF（Weighted Box Fusion，按置信度加权合并，优于简单 NMS）**融合；分类概率温度标定后用预测熵倒数加权；输出校准置信度并报告 ECE。
- **知识蒸馏**（`distill.py`）：teacher = MEGA 检测链；student = DEIM 仓库 D-FINE-S 配置（注意：DEIM 仓库没有叫"DEIM-small"的模型，其小模型即 D-FINE-S/DEIMv2-S 配置）+ 原型头；logit 蒸馏 + 特征蒸馏；导出 ONNX/TensorRT（`export_onnx.py`）并测吞吐。【P2 可选】INT8/FP8 量化导出 + 量化后精度复验（复验门槛同标准精度 Gate）；编写模型卡（Model Card）与数据卡记录用途边界与医学免责声明。
- **Gate 6**：验证集 **ECE ≤ 0.05**；蒸馏学生 GWHD AP@50 ≥ teacher 的 95%；ONNX/TensorRT 导出成功且单图延迟 ≤ 50 ms（服务器）或 ≥ 15 FPS（边缘，以实际部署机为准，报告中注明机型）；路由路径对比实验：重路径占比 < 30% 时整体精度下降 ≤ 1 pp（证明路由省算力不伤精度）。【P1 标准】端到端（含内置掩码分支）相比全量 SAM 级联的延迟下降 ≥ 40%，且 SIIM/自建掩码 IoU 差距 ≤ 2 pp。

### Phase 7：小样本—海量一致性验证（目标 5，创新点 2）
- **协议**（`eval/consistency.py` 强制执行）：
  1. 对 CXR14 分类任务跑 **5 / 10 / 50-shot 与全量**四档（固定种子，few-shot 用 `fewshot_split.py` 划分；50-shot 档用 **coreset 选择**【P1 标准】：按特征空间覆盖度/不确定度挑选 50 张，并附"随机抽样 vs coreset"对照表）；
  2. 对 GWHD 检测任务跑 10-shot 与全量两档（GWHD 是单类别检测，"10-shot" = 随机 10 张训练图、约 400+ 个实例；报告中必须同时给出"按图"与"按实例"两种口径，避免夸大 few-shot 含义）；
  3. 绘制"性能—样本量"曲线，计算 `gap = 全量指标 − 5(10)-shot 指标`，并对 gap 做**方差/偏差分解**【P0 强制】：用"同支持集多次重采样"估计方差项——方差主导 → 增加支持样本/启用协方差加权原型；偏差主导 → 走 continue-pretrain/LoRA；
  4. **通过标准：分类 AUC 与检测 AP@50 的 gap ≤ 5 个百分点**。
- **不通过的处理（按诊断对因下药，替代盲目按序尝试）**：
  a. 方差主导：增加支持样本或启用协方差加权原型（Mahalanobis），或走主动学习——对未标注样本按预测熵采样 top-1%，伪标注 + 人工抽查后加入支持集，重测 gap；
  b. 偏差主导：对底座加 rank=16 的 LoRA 微调（或启用 Phase 2 的 continue-pretrain 权重）后重测；
  c. **数据飞轮**【P1 标准】：主动学习正式化为闭环——模型 → 挖掘低置信/困难样本 → SAM 2.1 自动标注 + 人工抽查 → 入库重训；**停止条件：连续两轮 gap 改善 < 0.5 pp 即停**，飞轮轮次与增量写入报告。
- **Gate 7**：gap ≤ 5pp 达成；曲线图、四档指标表、方差-偏差分解结论、策略对照表（含 A5 分档实验，若启用）写入 `docs/reports/phase-7.md`。

### Phase 8：鲁棒性压测与最终验收（目标 4、6）
- **压测套件**（`eval/robustness.py`）：
  - 光度扰动（亮度 ±30%、对比度 ±30%）；
  - 几何扰动（高斯模糊 σ=1.5 模拟航拍抖动、旋转 ±15°）；
  - 噪声注入（高斯噪声 σ=25、X 光胶片伪影）；
  - 遮挡（随机擦除 10% 面积）；
  - 域漂移（CXR14 训练 → SIIM/TBX11K 跨机构评测）；
  - **真实域外评估（跨国家）**【P0 强制】：用 Phase 1 `country_split.py` 的留出国家集评测 GWHD——这是比合成扰动更真实的域泛化测试。
- **通过标准**：各类扰动下 AP@50 / AUC 降幅 ≤ 5 个百分点；TTA 开启后扰动降幅进一步收窄；**跨国家 AP@50 降幅 ≤ 8 pp**（合成扰动阈值仍为 ≤ 5 pp）。
- **Gate 8（最终验收）**：逐项核对第 7 部分验收总表（八大目标），全部满足 → 输出总验收报告；任何一项不满足 → 回退对应 Phase 修复。

### Phase 9（可选加分）：光谱支路（目标 2、7 的农业扩展）
- **实现**：`hsi/spectralformer.py`（简化 SpectralFormer：谱 token 编码 + 空间卷积 + 分类头）+ `hsi/fusion.py`（RGB 特征与谱特征交叉注意力融合）。
- **数据**：WHU-Hi-LongKou / HanChuan / HongHu。
- **Gate 9**：WHU-Hi 三基准总体精度 **≥ 97%**；融合支路相对"仅光谱"精度提升 ≥ 1 pp 的消融结论。

---

## 第 6 部分：关键配置与超参数（默认值，写入 configs/base.yml）

```yaml
seed: 42
device: cuda
fp16: true
backbone:
  name: dinov2_vitl14        # 冻结；备选 dinov3
  freeze: true
  lora: {enabled: false, rank: 16}   # Phase 7 不达标时开启
detail_branch: {name: convnext_tiny, freeze: true, out_stride: 4}
detector:
  name: deim                 # 对比去噪默认开启
  lr: 1.0e-4
  lr_backbone: 0
  batch_size: 8
  epochs: 30
  img_size: 1024
  ema: true
sahi: {slice: 1024, overlap_ratio: 0.2, postprocess_nms: true}
sam2:
  checkpoint: sam2.1_hiera_large.pt
  prompt: box
mask_iou_head: {lr: 1.0e-3, epochs: 10}
prototype_head: {tau: 0.1, ema_momentum: 0.9, dist: cosine}
router: {hidden: [64, 32], heavy_ratio_target: 0.3}
ensemble: {members: 3, tta: true, temperature: 1.5}
distill: {alpha_logit: 0.5, alpha_feat: 0.5}
fewshot: {k_values: [5, 10, 50], seed: 42}
consistency_gap_max: 5.0     # 百分点
robustness: {max_drop_pp: 5.0}
```

### 6.1 各阶段 GPU 时间预算（RTX 4090 单卡为基准；超预算 2 倍触发备选路线）

| Phase | 任务 | 预算 | 说明 |
|---|---|---|---|
| 全 | 基线金丝雀复现（每 Phase 前置） | 每项 ≤ 1 h | 环境健康检查；失败禁止进入该 Phase |
| 2 | CXR14 5-shot 探测 | 2 h | 冻结底座纯特征提取 + sklearn 逻辑回归 |
| 2 | 医学底座继续预训练 | 6 h | 224 输入、1 epoch（约 875 iters） |
| 3 | COCO 头初始化 | 1.5 d | 12 epochs、约 11k iters（冻结底座 + batch 16；渐进分辨率需先 A/B）；跳过则走 GWHD 90-epoch 直训路线（约 0.5 d） |
| 3 | GWHD 域微调（含密度图分支） | 8 h | 30 epochs × 214 iters ≈ 6.4k iters + 密度分支联合训练 |
| 4 | SAM 2.1 LoRA 误差感知微调 + 掩码蒸馏回灌 | 12 h | GWHD 伪掩码 + SIIM 子集 + 内置掩码分支蒸馏 |
| 5 | 医学双通道头 + 农业分析头 | 6 h | 全局+ROI 通道；底座特征缓存后分类头训练近乎秒级/epoch |
| 6 | 蒸馏 + 集成 + 路由 | 8 h | 学生微调为主 |
| 7 | 四档一致性协议 | 1 d | 大头在 50-shot/全量档（coreset 可省时） |
| 8 | 鲁棒性压测 | 4 h | 纯推理，无训练 |

---

## 第 7 部分：最终验收对照总表（Phase 8 逐项打勾）

| 目标 | 验证方法 | 通过标准 |
|---|---|---|
| 1 创新算法 | 代码审查清单：六创新点对应模块存在且有单测 | 6/6 落地 |
| 2 特殊领域 | 农业（GWHD/自建）+ 医学（CXR14）双域跑通 | 双域 Gate 通过 |
| 3 国际最新设计 | 依赖清单：DINOv2/DINOv3、DEIM、SAM 2.1、SAHI 实际被调用（非仅引用） | 全部实际使用 |
| 4 高精度/鲁棒/检测+分类+分析 | Gate 3/5/8 | AP@50 ≥ 0.90（GWHD）、mean AUC ≥ 0.80（CXR14）、扰动降幅 ≤ 5pp |
| 5 小样本一致性 | Phase 7 协议 | gap ≤ 5pp（含方差/偏差分解归档） |
| 6 复杂场景关键信息 | SAHI 对比实验 + 微小目标召回提升 | 小目标召回相对提升 ≥ 10%；自适应切片省算力 ≥ 25%；X 光微小病灶检出 demo |
| 7 功能取决于样本类别 | 同一 `pipeline.py` 跑通两个域，仅替换 config 与权重 | 双域 demo 输出正确格式结果 |
| 8 多模型协同 | Gate 6 | ECE ≤ 0.05、蒸馏 ≥ 95% teacher、路由省算力不伤精度、内置掩码端到端延迟 −40% |
| 增 前瞻优化项（v2.0） | Gate 3/4/5/8 | 跨国家 AP@50 降幅 ≤ 8pp；内置掩码 IoU 差距 ≤ 2pp；密度融合计数 < 3%；弥漫性子集 AUC ≥ 0.75 |

---

## 第 8 部分：失败处理决策树（GLM5.3 遇到问题按此处理）

| 情况 | 处理 |
|---|---|
| 无 GPU / 非 Linux | **立即停止**，输出环境现状报告，不执行任何训练 |
| 环境与钉死版本冲突（如驱动过新导致 cu121 轮子报错） | 按 3.2 授权规则处理：优先微调安装方式（换 cu124/cu126 轮子并保持 torch 主版本 2.4.x）；仍冲突则备案偏离并重跑 Gate 0 + 对应金丝雀；关键件更换必须出现在阶段报告"偏离备案"小节 |
| 显存不足（OOM） | 依次降配：batch 减半 → 图像 1024→768 → 底座换 dinov2-small、SAM 换 sam2.1_small、检测头换 D-FINE-S 配置；每步记录显存峰值，报告注明降配影响 |
| GitHub / Hugging Face 下载失败 | 换镜像：HF 用 `https://hf-mirror.com`（设置 `HF_ENDPOINT`），GitHub 用 ghproxy；仍失败则记录并改用备选来源 |
| Kaggle 数据集无法获取 | 用等价公开集替代并报告（如检测可用 VisDrone 小目标集；医学可用 ChestX-det10） |
| Gate 未通过 | 分析失败原因（日志/曲线/坏例可视化）→ 修复重试 ≤ 3 轮 → 仍失败：写入阶段报告"未达标+原因+降级方案"，并在总验收报告中显式标记该项风险，**不得谎报通过** |
| 医学数据合规问题 | 只使用公开数据集；所有医学结果加免责声明"仅供研究辅助，不作临床诊断依据" |
| 训练超出时间预算 2 倍 | 按第 6.1 节触发备选路线（COCO 头初始化 ↔ GWHD 90-epoch 直训、LoRA/降分辨率、镜像源），切换原因写入报告 |

---

## 第 9 部分：最终交付物清单

1. `mega-vision` 代码仓库（含全部 Phase 代码、单测、configs）
2. 每阶段报告 `docs/reports/phase-0..8.md`
3. 总验收报告：八大目标逐项打勾 + 指标表 + 性能-样本量曲线 + 鲁棒性压测结果
4. 训练好的权重（农业检测、医学分类、蒸馏学生、可选高光谱）与下载说明
5. 演示脚本 `scripts/demo.py`（单图/目录推理，输出检测+分割+分类+分析可视化结果）
6. 边缘部署包（ONNX/TensorRT + 部署说明）
7. 设计文档 `docs/DESIGN.md`（本文件副本 + 算法设计报告全文）

---

**给 GLM5.3 的最后一条指令**：从第 3 部分开始逐 Phase 执行，每通过一个 Gate 就输出阶段报告，全程遵守第 0 部分纪律；全部完成后，输出总验收报告并对照第 7 部分表格逐项确认。
