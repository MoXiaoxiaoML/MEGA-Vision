# MEGA-Vision 技术可行性验证报告（v3.0：Astra 目标 + 2026-11 技术栈刷新版）

**版本记录**

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-09-21 | 组件/依赖/数据集/验收阈值实证核验（HTTP 探测 + GitHub API + 文献交叉） |
| v2.0 | 2026-10-04 | 重新可行性论证：融合 ScienceDirect 18 篇调研中采纳的 7 项文献方法，逐项给出可行性证据与资源代价 |
| **v3.0** | **2026-11-10** | **随目标执行模型迁移 GPT-6 Astra 与技术栈刷新重做核验**：①逐项核验 2026-11 技术栈（DINOv3/DEIMv2/SAM 2.1/SAM 3.1/GWFSS/环境版本/许可证/链接）；②逐条核验 24 条引用（无虚构，8 处标题/年份/出处更正、2 条无出处断言删除）；③修正 v2.0 及以前的环境与许可结论（DINOv2 权重许可、518 原生分辨率断言、SAM 2.1 年份、DEIMv2 NC 许可、GWHD 图像数口径）；④更新风险表与预算 |

**核验方法（2026-11-10）**：逐项抓取官方页面（PyPI / GitHub / Hugging Face / arXiv / 期刊 DOI / 数据集官方站），核对标题、版本号、许可、权重直链与可达性；无法直接验证的项明确标注"未验证/无法确证"，不以推测替代。**状态标记**：✅ 页面可访问且含对应内容；⚠️ 部分核验（有出入或需条件）；❌ 未能核验/不成立。

---

## 一、总体结论

1. **目标模型迁移可行**：GPT-6 Astra（2026-09-03 发布，`gpt-6-astra`，上下文 1,050,000 / 最大输入 922,000 / 最大输出 128,000，知识截止 2026-04-30）具备执行本工作流所需的工具调用（Responses API）、computer-use 与长上下文能力；适配要点（身份注入、知识截止规则、状态外部化、越界白名单、证据链）已写入工作流 v3.0 第 0 部分。
2. **2026-11 技术栈成立**：DINOv3 ViT-L/16 + DEIMv2 训练配方 + SAM 2.1（默认）/SAM 3.1（P1）组合在许可、权重获取与显存/算力上均可落地；环境钉死 torch 2.10.0+cu128 / torchvision 0.25.0 / Python 3.12（有官方兼容表与上游安装说明证据），最新版（torch 2.15.0 / torchvision 0.29.2）列为备案升级项。
3. **引用无虚构**：设计报告 §1.2.10 与 §5 的 24 条引用全部可溯源，但 8 条标题/年份/出处不准确已更正、2 条无出处技术断言已删除；**未发现需要撤回的整篇虚构文献**。
4. **风险可控**：主要新增风险为 gated 权重获取（DINOv3/SAM 3.1）、DEIMv2 的 NC 许可、GWHD 官方划分口径不确定、CXR14 GCS 桶计费；均已在工作流 v3.0 第 2.4/8 节给出降级或替代路线。

---

## 二、2026-11 技术栈刷新核验

### 2.1 运行环境

| 项 | 核验结论 | 状态 | 对工作流的处置 |
|---|---|---|---|
| PyPI 最新 torch | **2.15.0**（要求 Python ≥3.11；默认 wheel CUDA 13.2，另有 13.4） | ✅ | 仅列为备案升级项，不作默认 |
| PyPI 最新 torchvision | **0.29.2**（要求 torch ≥2.14.0） | ✅ | 同上；torch 2.15 与 0.29 配对官方未确认 |
| 官方兼容表 | **torch 2.10 ↔ torchvision 0.25**（另有 2.11↔0.26、2.12↔0.27、2.13↔0.28） | ✅ | **默认钉死 torch 2.10.0+cu128 / torchvision 0.25.0** |
| CUDA wheel 索引 | cu128 索引含 2.7–2.11+；cu130 索引含 2.10–2.14 | ✅ | 默认用 cu128（与 SAM 3 官方安装说明一致） |
| Python | 最新 3.15；torch 2.15 要求 ≥3.11；3.10 已于 2026-10 EOL | ✅ | 默认 Python 3.12（SAM 3 要求 ≥3.12） |
| Ubuntu | 26.04 LTS 已发布；24.04 LTS 为推荐基线 | ✅ | 默认 24.04；22.04 可用 |
| NVIDIA 驱动 | 生产分支 620.70.06；CUDA 12.8 需 ≥570 | ✅ | 默认要求 ≥580 |
| TensorRT | 11.4.0.106 | ✅ | FP16 推理必须 ≥10.6（DEIMv2 已知 10.4 问题） |

### 2.2 模型与组件

| 组件 | 核验结论 | 状态 | 对工作流的处置 |
|---|---|---|---|
| **DINOv3 ViT-L/16** | 2025-08 发布；300M 参数、1024 维、24 层、16 头、4 register、patch 16、**RoPE**；224 输入→201 token；输入只需为 16 的整数倍，**512 输入→32×32=1024 patch token**（+1 CLS+4 reg=1029）；LICENSE 为 **DINOv3 License（非 NC，可商用附条件）**；权重 **gated**（HF `facebook/dinov3-vitl16-pretrain-lvd1689m`，或接受许可后 Meta 邮件直链；约 1.21 GB）；timm ≥1.0.20、transformers ≥4.56 支持 | ✅ | 作为默认视觉底座；gated 下载步骤与降级 DINOv2 路线写入 3.2/8 节；**"512 原生分辨率、零位置编码插值"旧断言不成立，已改写** |
| DINOv2 ViT-L/14（备选） | 代码 Apache-2.0；HF 模型卡 `license: apache-2.0`（2026-11-10 核验）；300M 参数、patch 14、RoPE 无（绝对位置编码） | ✅ | 备选底座；工作流旧版"权重 CC-BY-NC-4.0"表述更正 |
| **DEIMv2** | 2025-09-26 发布（`Intellindust-AI-Lab/DEIMv2`）；DETR 系实时 SOTA，DINOv3 原生集成 + STA；COCO：S 50.9 AP/9.7M、M 53.0/18.1M、L 56.0/32.2M、X 57.8/50.3M；权重 HF `Intellindust/DEIMv2_DINOv3_*_COCO`；**LICENSE = DEIMv2 License（NC，禁止商用，商用需联系授权）**；2026 演进为 EdgeCrafter（TMLR 2026，同样 NC） | ✅ | 作为检测训练配方来源与对照基线；**明确 NC 边界**，商用/闭源换 D-FINE/RF-DETR |
| D-FINE（合规备选） | ICLR 2025 Spotlight，Apache-2.0 | ✅ | 检测备选与商用替代路线 |
| RF-DETR（合规备选） | Roboflow，Apache-2.0 | ✅ | 检测备选路线 |
| **SAM 2.1** | 实际发布于 **2024-09-30**（旧文档写 2025，已更正）；Apache-2.0；权重直链 `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt` 有效；要求 python ≥3.10、torch ≥2.5.1、torchvision ≥0.20.1 | ✅ | 默认分割教师，24GB 可 LoRA 微调 |
| **SAM 3.1** | SAM 3 于 2025-11-19 发布；**SAM 3.1（Object Multiplex）于 2026-03-27 发布**；848M 参数；权重 `sam3.1_multiplex.pt` 约 3.5 GB；**SAM License（自定义、非 Apache）**，权重 **gated**（HF `facebook/sam3.1`）；要求 Python ≥3.12、torch ≥2.7、CUDA ≥12.6；Sam3Tracker 兼容 SAM2 API 并支持 box prompt | ✅ | P1 升级项：仅零样本教师推理；不参与训练/最终推理；显存不足或未获批回退 SAM 2.1 |
| SAHI | 0.14.0，MIT | ✅ | 切片推理仍保留；DEIMv2 需自定义 DetectionModel 封装 |
| Ultralytics（YOLO26/RT-DETR 系列） | AGPL-3.0（商用需企业授权） | ✅ | 仅对照/非交付；不进入商用路径 |

### 2.3 数据集

| 数据集 | 核验结论 | 状态 | 对工作流的处置 |
|---|---|---|---|
| **GWHD 2021** | Zenodo `10.5281/zenodo.5092309`：`gwhd_2021.zip` 约 **10.2 GB**，md5 `22b4b542c9ae7e056d7fcdeae9ecaed5`，CC-BY-4.0；权威标注量 **275,187 头 / 16 机构 / 12 国**（Plant Phenomics 2021, DOI 10.34133/2021/9846158）；官方划分 CSV 在包内。**图像总数论文口径约 6,500 张**（官方站"超过 6000"，WILDS 镜像 6,515）；旧文档"6,522 张 / 训练集 3,422"**无法确证**；HF 镜像（3,655/1,476/1,381）为二次重划分 | ✅（总数⚠️） | 改为"以官方 CSV 实测行数为准"；迭代数与预算按实测重算；禁用 HF 重划分当官方划分 |
| **GWFSS 2025（新增）** | Plant Phenomics 2025，DOI 10.1016/j.plaphe.2025.100084；HF `GlobalWheat/GWFSS_v1.0`：**1,096 张像素级语义器官掩码（叶/茎/穗），CC-BY-4.0，约 611 MB**；另有 52,078 张无标注图（论文口径）；**语义非实例**，与 GWHD 配对性未证实 | ✅（配对性⚠️） | 仅作 Phase 4 掩码预训练/教师信号，不作实例监督；使用前 50 图验证口径 |
| ChestX-ray14 | 官方 Box 页有效；GCS 桶 `gs://gcs-public-data--healthcare-nih-chest-xray/png/` **Requester Pays（读取方付费）**；112,120 张、14 病种 | ⚠️ | 工作流补充计费说明与标签表名；致谢 NIH |
| SIIM-ACR | Kaggle 竞赛页（需登录接受规则）；本次抓取限流，**未验证下载**；含掩码数量以实际下载统计为准 | ⚠️ | 删除旧文档"约 2,600 张"未核实数字 |
| TBX11K | `github.com/yun-liu/Tuberculosis`（TPAMI 2023）：11,200 张，Train 6,600 / Val 1,800 / Test 2,800；测试标签需挑战注册；Kaggle 镜像存在 | ✅ | 只用 train+val 自划分 |
| WHU-Hi | `rsidea.whu.edu.cn` 可访问；LongKou / HanChuan / HongHu（270/274 波段） | ✅ | Phase 9 使用 |

### 2.4 许可证判定（可商用？）

| 组件 | 许可证 | 可商用 | 处置 |
|---|---|---|---|
| DINOv3 | DINOv3 License | ✅ 附条件（附协议/致谢/禁军事；权重 gated） | 接受记录入库 |
| DINOv2 | Apache-2.0（HF 模型卡） | ✅ | 旧 NC 表述作废 |
| DEIMv2 / EdgeCrafter | 自定义 NC | ❌ | 仅非商业研究；交付换 D-FINE/RF-DETR |
| SAM 2.1 | Apache-2.0 | ✅ | — |
| SAM 3/3.1 | SAM License | ✅ 附条件（gated） | 交付申报 |
| D-FINE / RF-DETR / SAHI | Apache-2.0 / MIT | ✅ | 商用替代链 |
| Ultralytics | AGPL-3.0 | ⚠️ | 仅对照 |
| TensorRT | NVIDIA 专有 | ✅ 免费工具 | 分发遵守条款 |
| 数据集 | CC-BY-4.0 / 学术条款 | ✅ 研究用途 | 署名 + 条款登记 |

---

## 三、引用真伪核验（设计报告 §1.2.10 / §5 / 工作流）

**结论：无整篇虚构引用**；8 条"部分准确"已更正，2 条无出处技术断言已删除。

| # | 原引用 | 判定 | 更正 |
|---|---|---|---|
| 1 | CheXFound: A Visual Foundation Model for Chest X-Ray, IEEE TMI 2025 | ⚠️ 模型名当标题 | 标题应为 *Chest X-ray Foundation Model with Global and Local Representations Integration*（arXiv 2502.05142） |
| 2 | Ark+: A Foundation Model for Chest X-Rays, Nature 2025 | ⚠️ 标题不准确 | 实际 *A fully open AI foundation model applied to chest radiography*（DOI 10.1038/s41586-025-09079-8） |
| 3 | UniHSFormer-X: Unified Transformer for Hyperspectral Crop Monitoring | ⚠️ 标题/作者不符 | 实际标题含 "Prototype-Routed Semantic Structuring"（Agriculture 15(13):1427） |
| 4 | SAM 2.1, 2025 | ⚠️ 年份错 | **2024-09-30** |
| 5 | IFAC 2024 "Towards Minimizing Domain Gap Using Synthetic Data" | ⚠️ 标题不完整 | 补全 "…in Automotive Vision Control Applications" |
| 6 | LF-GAN "for CT image synthesis" CMPB 2024 | ⚠️ 标题不完整 | 补全 "high-fidelity … of liver tumor" |
| 7 | AOR 2025 海上内容感知合成 | ⚠️ 标题不完整 | 补全 "…with diffusion models" |
| 8 | KNNOR-Reg "MIT 许可" | ⚠️ 许可未核实 | 改为"开源；许可以仓库 LICENSE 为准" |
| 9 | §1.2.3 "SAM2 在太阳能板航拍分割、SAM2CT 在 CT 病灶分割优于前代" | ❌ 无出处 | **删除** |
| 10 | §1.2.6 "2025 光学流配准误差 2–4 像素" | ❌ 无出处 | **删除**，改为"配准误差须实测并写入报告" |
| 11 | DINOv2 权重 CC-BY-NC-4.0 | ⚠️ 过时 | 现 HF 模型卡为 Apache-2.0 |
| 12 | 工作流"518×518 为 DINOv2 原生训练分辨率、零插值" | ⚠️ 表述不成立 | 改写为 DINOv3 RoPE 的可变分辨率事实；不再承诺"零损失" |

其余引用（DINOv2、DINO、DEIM、DEIMv2、SAM、SAM 2、RT-DETR、SAHI、Prototypical Nets、LoRA、SpectralFormer、EVA-X、WheatSAM、Frozen-DETR、L1-L4/L7 文献、GWHD/GWFSS/CXR14）经核验为**真实且出处准确**（个别补全卷期/DOI）。

---

## 四、历史结论继承与修订

| v1.0/v2.0 结论 | v3.0 状态 |
|---|---|
| DINOv2 ViT-L/14、SAM 2.1 Hiera-L 官方直链可下载 | ✅ 仍有效（SAM 2.1 直链已复核；DINOv2 作为备选底座） |
| DINOv2 权重 CC-BY-NC-4.0 | ❌ **修订**：现模型卡 Apache-2.0 |
| 518 为 DINOv2 原生分辨率、零位置编码插值 | ❌ **修订**：旧断言不成立；DINOv3 用 RoPE，512 为推理分辨率 |
| DEIM 仓库许可证"待核验" | ✅ **已核**：DEIM（原版）Apache-2.0；**DEIMv2 为 NC**，二者不可混用 |
| GWHD 2021：6,522 图 / 275,187 头 | ⚠️ **部分修订**：275,187 头/12 国/16 机构可证；图像总数以官方 CSV 实测为准（约 6,500） |
| CXR14 / SIIM / TBX11K / WHU-Hi 可获取 | ⚠️ 基本有效；CXR14 GCS 为 Requester Pays，SIIM 需登录且掩码数量未核实 |
| Gate 3 AP@50≥0.90、Gate 5 mean AUC≥0.80、Gate 4 IoU≥0.80 | ✅ 维持（阈值仍成立，实测以 Gate 为准） |
| Dice↔IoU 换算（0.897→0.813） | ✅ 复核通过 |
| Frozen-DETR 冻结 DINOv2-L + DETR 头 +4.8 AP（Apache-2.0） | ✅ 复核通过 |
| 文献融合 L1–L7 方法 | ✅ 方法可行性与证据仍成立；仅 L2/L3/L5/L6 的标题/许可表述按第三节更正 |

---

## 五、L1–L7 文献融合方法可行性（继承 v2.0，许可证与标题按 v3.0 更正）

| 项 | 方法 | 证据强度 | 落地等级与代价 | v3.0 备注 |
|---|---|---|---|---|
| L1 | Focal Tversky 掩码损失 + OTR 最优阈值（Eng. Struct. 2023） | 4 个失衡分割基准系统对比，FTversky 综合最优 | 必做；零训练时间成本 | 需在 SAM LoRA 上复验，无增益回退 Dice+BCE |
| L2 | 扰动影响分排序（IFAC 2024） | 受控随机化显著收窄合成-真实域差，并量化各扰动影响分 | 必做；每组扰动一次重评 | 标题已补全；结论域内使用，不跨域外推 |
| L3 | KNNOR-Reg 特征级回归过采样（Software Impacts 2025） | 多回归器对比 MAE/RMSE 多数最优 | 标准；半天接入 | 许可证以仓库为准（旧"MIT"未证实） |
| L4 | 稀有类增广三重门控（ASOC 2024） | 20 数据集基准：小样本下 SMOTE 族优于深度生成 | 标准；离线步骤近零成本 | ENN 过滤 + 入池前后 AUC 验收 |
| L5 | 轻量无预训练 GAN + GLCM 纹理门（CMPB 2024） | 100 张 CT 从零训练，下游分类显著提升；RTX3060 可训 | 备选；约 1–2 周工程量 | 标题已补全（liver tumor） |
| L6 | 内容感知合成（AOR 2025，开源） | 海上任务 mAP50 0.35→0.44 | 备选；约 1–2 天生成千张级 | 医学禁用姿态变换；标题已补全 |
| L7 | 协议级（DSS 2017）：预训练→微调、balanced accuracy、时序切分 | 协议证据 | 必做；零代码成本 | 时序防泄漏已写入 Phase 1 |

**排除项维持 v2.0 结论**：纯数学/期权定价/无 ML 的骨科病例/与端到端训练不兼容的随机权重激活，共 9 篇排除；NCU/tanhLU 激活函数仅做一次限定消融。

---

## 六、更新后的风险表

| 风险 | 影响 | 对策（v3.0） |
|---|---|---|
| DINOv3 / SAM 3.1 权重 gated 未获批 | 底座或升级分割教师不可用 | 提前在 Phase 0 申请并登记；DINOv3→DINOv2 备选、SAM 3.1→SAM 2.1 回退；均写入报告 |
| DEIMv2/EdgeCrafter 为 NC 许可 | 商用/闭源交付受限 | 仅非商业研究使用；交付切 D-FINE/RF-DETR（Apache-2.0）；`licenses/LICENSES.md` 标注 |
| torch 2.10 与某上游组件不兼容 | 环境搭不起来 | 三环软约束偏离：在官方兼容表内换 2.11/0.26 或 2.14/0.29，重跑 Gate 0 + 金丝雀 |
| Torch 2.15 与 torchvision 0.29 配对未确认 | 盲目升级风险 | 默认不升级；升级必须备案 + 复验 |
| GWHD 官方划分口径不确定 | 迭代数/预算/指标口径偏差 | 以官方 CSV 实测行数为唯一口径；重算并记录 |
| CXR14 GCS 桶 Requester Pays | 下载产生费用 | 改用官方 Box 或本地镜像；记录费用来源 |
| GWFSS 语义掩码与 GWHD 配对性未证实 | 掩码监督无效 | 仅作预训练/教师信号，先 50 图验证；不用于实例监督 |
| SAM 3.1 848M/3.5GB 显存压力 | OOM | 仅推理；降分辨率/AMP/回退 SAM 2.1 |
| TensorRT FP16 版本问题 | 推理结果错误 | 强制 TensorRT ≥10.6；否则改 FP32/ONNX Runtime |
| 医学预训练权重（RETFound/RAD-DINO 等）不可用 | 违规风险 | 仅作文献对比，不引入权重；底座只用非医学预训练 DINOv3/DINOv2 |
| 扩散/生成增广在小样本反噬精度 | 稀有类更差 | L4 规模 Gate + ENN + AUC 对比，不改善即弃用（继承 v2.0） |
| 小样本与全量仍有差距 | 不满足目标 5 | Phase 7 协议 + 方差-偏差分解 + 主动学习/LoRA（继承 v2.0） |

---

## 七、时间与资源预算（v3.0）

| 项 | v2.0/v2.4 预算 | v3.0 | 说明 |
|---|---|---|---|
| Phase 2 医学继续预训练 | 6 h | 4–6 h（实测后修正） | 224 输入、1 epoch = 876 iters（112,120÷128）；须只用官方 train split |
| Phase 3 COCO 头初始化 | 1.5 d | 1.5 d（wall-clock） | 12 epochs ≈ 17.7 万 iters（batch=8；若登记 batch=16 则 ≈8.9 万）；可跳过走 GWHD 直训（≈1 d） |
| Phase 3 GWHD 微调 | 8 h | ≤10 h（wall-clock） | 30 epochs × ceil(N_train/8)，N_train 以官方 CSV 实测为准（约 3,655–4,000，旧 4,700/3,422 无依据） |
| Phase 4 SAM LoRA + 蒸馏 | 12 h | 12 h + 2–4 h（SAM 3.1 教师可选） | SAM 3.1 仅推理 |
| Phase 5/6/7/8 | 6/8/24/8 h | 6–8 / 24（8 蒸馏 + 16 成员训练）/ 24 / 8 h | Phase 6 含两成员域微调 +16 h（工作流 v3.0 补齐集成闭环） |
| 单卡总预算 | 4–6 天 | **5.5–7.5 天**（含 COCO 与集成成员；跳过 COCO 并收敛成员约 4–5.5 天） | RTX 4090 基准 |

---

## 八、结论

**2026-11 技术栈刷新后的方案可行且更强**：
1. 底座换 DINOv3 后，许可证从"旧文档误写的 NC"变为**可商用附条件**（仍 gated），且 patch 16 + RoPE 与 DEIMv2 的 DINOv3 集成天然匹配；
2. DEIMv2 提供 2025 最成熟的 DETR 实时训练配方与 STA，但其 **NC 许可**已如实标注并提供 Apache-2.0 替代链；
3. SAM 2.1 保持 24GB 可训的默认教师，SAM 3.1 作为 2026 最新可选的零样本教师；
4. 所有引用与链接经 2026-11-10 核验，8 处更正、2 条删除、无虚构；
5. 迁移 GPT-6 Astra 的执行风险（知识截止、状态漂移、越界倾向、输出上限）已由工作流 v3.0 第 0 部分与第 8 节逐条覆盖。

**未验证项（如实声明）**：DINOv3 与 SAM 3.1 的 gated 审批时长、GWHD 官方 CSV 的确切行数、SIIM 含掩码图像数、GWFSS 与 GWHD 的配对性、torch 2.15/torchvision 0.29 与全部上游组件的兼容性——这些须在执行时按工作流 Gate 0/Phase 1 实测记录。
