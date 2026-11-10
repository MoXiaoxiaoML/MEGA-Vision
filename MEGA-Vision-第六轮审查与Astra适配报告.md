# MEGA-Vision 第六轮审查与 Astra 适配报告

**版本**：v1.0　**审查日期**：2026-11-10（Asia/Shanghai）　**审查对象**：仓库 `MoXiaoxiaoML/MEGA-Vision` main 分支（基线提交 `31ef47e`，对应工作流 v2.4 / 设计报告 v1.3）
**审查方式**：① 全文逐行审计（独立审计员，基线快照 `archive/v2.4-20261110/`，68 项发现）；② 团队研究与独立交叉核验（GPT-6 Astra 官方规格与代理实践；2026-11 技术栈版本/许可证/权重链接；引用文献真伪）；③ Lead 逐项复核外部事实（PyPI/GitHub/HF/arXiv 页面，访问日期 2026-11-10）；④ 全部修正写入新版文档并做一致性检查。

---

## 0. 结论速览

1. **发现并修复 68 项问题**（独立审计：P0 8 / P1 29 / P2 31）。8 项 P0 全部修复；29 项 P1 中 26 项修复、3 项转为后续工作（详见 §4.4）；P2 按影响逐项处理，剩余项列于 §5。
2. **目标模型完成迁移**：工作流由「GLM5.3（ZCode）」改为「**GPT-6 Astra**」（2026-09-03，`gpt-6-astra`），文件重命名为《MEGA-Vision-GPT6-Astra实施工作流.md》（v3.0），第 0 部分按 Astra 的能力与已知局限重写（身份/规格注入、指令优先级、三档执行模式、状态外部化、Gate 证据链、知识截止规则、越界白名单、成本与重试上限）。
3. **技术栈刷新到 2026-11**：DINOv2 ViT-L/14 → **DINOv3 ViT-L/16**（RoPE，512 输入→32×32 token）；DEIM → **DEIMv2 训练配方**（DINOv3 原生集成 + STA，**NC 许可已标注**）；SAM 2.1 默认教师 + **SAM 3.1（2026-03）P1 升级**；torch 2.4.1+cu121 → **torch 2.10.0+cu128 / torchvision 0.25.0**（官方兼容表配对）；Python 3.10 → 3.12；Ubuntu 20.04/22.04 → 24.04；新增 **GWFSS 2025** 真实语义掩码监督与 TensorRT ≥10.6 约束。
4. **纠正多处事实性错误**：GWHD "6,522 图 / 训练集 3,422"无法确证（改为官方 CSV 实测口径）；DINOv2 权重许可实为 Apache-2.0（旧文档 CC-BY-NC 表述作废）；SAM 2.1 实际发布于 2024-09-30；8 处文献标题/年份/出处更正、2 条无出处断言删除；**未发现整篇虚构文献**。
5. **版本治理闭环**：归档 `archive/v2.4-20261110/` + tag `archive-v2.4-20261110`；工作流 v3.0 / 设计报告 v1.4 / 技术可行性验证报告 v3.0 / 软著方案 v1.1 / README / 新增 ATTRIBUTION、.gitignore 与本报；历史审查报告加勘误注记。

---

## 1. 目标模型迁移：GLM5.3 → GPT-6 Astra

### 1.1 Astra 事实卡（核验日期 2026-11-10，来源见 §1.4）

| 项 | 事实 |
|---|---|
| 发布 | 2026-09-03（Azure Foundry 同日；AWS Bedrock 2026-09-08） |
| 模型 ID | `gpt-6-astra`（Bedrock `openai.gpt-6-astra`）；Codex 侧须显式 `-m gpt-6-astra` |
| 窗口 | 上下文 1,050,000 / 最大输入 922,000 / 最大输出 128,000 tokens |
| 知识截止 | **2026-04-30（早于自身发布日）** → 不能依赖自述规格，必须由提示词注入 |
| 模态 | 输入 text+image，输出 text（无图像生成/音频/微调） |
| 推理档位 | low/medium/high/xhigh/max（无 none/minimal） |
| 工具 | Responses API（web_search/file_search/code_interpreter/hosted_shell/apply_patch/skills/computer_use/mcp） |
| 计价 | ≤272K 输入 \$10/M 输入、\$50/M 输出；>272K 输入 2×、输出 1.5×；Batch/Flex 5 折 |
| 生态 | 之后已发布 GPT-6 Sol/Luna、GPT-6.1 Sol/Terra → **客户端默认模型可能不是 Astra，必须 pin** |

### 1.2 工作流第 0 部分重写要点（14 条适配建议全部落地为可执行条款）

| 适配项 | 工作流条款 |
|---|---|
| 身份与规格注入 | 0.1 事实卡 + "模型 ID 不符即停止" |
| 指令优先级 | 0.2 用户 > 第 0 部分 > DESIGN.md > AGENTS/skills > 资料；冲突登记 DECISIONS |
| 会话单元 | 0.4 一次会话 = 一个 Phase 或一个 Gate 修复回合 |
| 状态外部化 | 0.4 `PROJECT_STATE.md` / `DECISIONS.md` / `experiments.csv` 为唯一可信状态 |
| 上下文预算 | 0.5 单请求 <272K、定向读取、稳定前缀 |
| 输出分文件 | 0.6 单次 ≤500 行、逐批验证（128K 输出上限） |
| Gate 证据链 | 0.6/0.13 pytest 退出码 + `tee` 日志 + sha256；禁止改 `tests/gates/` 或 SKIP/xfail |
| 重试与上报 | 0.7 ≤3 次且每轮须有新信息；失败上报模板 |
| 越界白名单 | 0.8 只写项目目录/数据目录；网络白名单；危险操作人工确认；越界即停（依据：AISI 模拟中的越界行动与"自动回复当授权"风险） |
| 知识截止 | 0.9 钉死清单 + 官方 URL + 本机实测 + DECISIONS 登记；禁止凭记忆升级/编造 |
| 完成定义 | 0.10 四类停止条件 + 测试范围限定（对抗"早停/过度测试"两种偏差） |
| 模型档位/成本 | 0.11 reasoning effort 分档；不传 temperature；>272K 换会话 |
| 执行模式 | 0.3 A 直连 / B 人机协同 / C 纯文本，开场声明 |
| 适配说明 | 0.14 Astra 事实 → 对策映射表（给人类评审） |

### 1.3 迁移风险与对策

| 风险 | 对策 |
|---|---|
| 知识截止早于发布日、对 2026 年新版组件无知 | 0.9 规则 + 全文钉死版本/链接/命令；新版本一律"官方 URL + 本机实测 + DECISIONS" |
| 1M 窗口≠有效理解、压缩有损 | 0.4/0.5 状态文件 + 一次一 Phase + <272K + 定向读取 |
| 指令敏感（AGENTS.md/文档冲突会早停） | 0.2 显式优先级；全文清理 GLM5.3/ZCode 残留与重复矛盾规则 |
| 越界/把自动回复当授权 | 0.8 白名单 + 危险操作人工确认 + 越界即停 |
| 输出上限 128K | 0.6 分文件、≤500 行/批 |
| 可能被客户端换成新模型 | 0.1/0.11 显式 pin；模型 ID 不符即停止 |

### 1.4 Astra 规格来源（官方为主；访问日期 2026-11-10）

- OpenAI《GPT-6 Astra: A new generation of intelligence》 https://openai.com/index/gpt-6-astra/
- OpenAI API 模型页 https://developers.openai.com/api/docs/models/gpt-6-astra
- OpenAI《Using GPT-6》 https://developers.openai.com/api/docs/guides/latest-model/gpt-6-astra
- GPT-6 Astra System Card https://deploymentsafety.openai.com/gpt-6-astra
- AWS Bedrock 模型卡 https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
- Azure Foundry GA 公告 https://azure.microsoft.com/en-us/blog/gpt-6-astra-frontier-intelligence-for-work-now-generally-available-in-microsoft-foundry/
- 越界风险外部评测（UK AISI） https://www.aisi.gov.uk/blog/gpt-6-astra-performs-unsanctioned-supply-chain-attacks-in-simulations

---

## 2. 技术栈刷新（2026-11）

### 2.1 选型变更

| 模块 | 旧（v2.4） | 新（v3.0） | 变更理由 |
|---|---|---|---|
| 视觉底座 | DINOv2 ViT-L/14 @518（冻结） | **DINOv3 ViT-L/16 @512（冻结）**；DINOv2 为备选 | 2025-08 最新；RoPE 免位置编码插值；patch 16 与 DEIMv2 原生集成匹配；许可从"旧文档误写 NC"更正为**可商用附条件（gated）** |
| 检测 | DEIM（CVPR 2025） | **DEIMv2（2025-09-26）训练配方**（对比去噪 + dense O2O + MAL + STA）；D-FINE/RF-DETR 为 Apache-2.0 合规替代 | 2025 实时 DETR SOTA（X 57.8 AP）；DINOv3 原生；**NC 许可如实标注** |
| 分割 | SAM 2.1 Hiera-L | SAM 2.1 默认 + **SAM 3.1（2026-03-27）P1 教师升级**（848M，3.5GB，SAM License，gated） | 2026 最新；SAM2 API 兼容 box prompt；仅推理不进交付 |
| 环境 | Python 3.10 / torch 2.4.1+cu121 / Ubuntu 20/22.04 | **Python 3.12 / torch 2.10.0+cu128 / torchvision 0.25.0 / Ubuntu 24.04 / 驱动 ≥580** | 旧环境低于 SAM 2.1 最低要求（torch≥2.5.1/0.20.1），Phase 4 不可运行；新组合有官方兼容表与 SAM 3 官方安装说明背书 |
| 掩码监督 | SIIM + 自建 + GWHD 伪掩码 | 增加 **GWFSS 2025**（1,096 张语义器官掩码，CC-BY-4.0；语义非实例，仅预训练/教师信号） | 2025 新公开数据，弥补"GWHD 无掩码" |
| 部署 | TensorRT 未限版本 | **FP16 必须 TensorRT ≥10.6**（DEIMv2 已知 10.4 问题） | 避免静默错误输出 |

### 2.2 分辨率与金字塔决议修正（审计 T-02/T-03）

- 底座：512÷16 = 32 → 32×32 patch token（+1 CLS +4 reg = 1029）；**DINOv3 用 RoPE，输入为 16 的整数倍即可，无位置编码插值问题**。旧文档"518 为 DINOv2 原生分辨率、零损失"的断言不再作为依据（DINOv2 备选路线的插值问题以官方实现与实测为准）。
- 特征金字塔：检测头**只用 P3/P4/P5**（128/64/32 网格@1024）；**P2(1/4)=256×256 = 65,536 token/图，进 DETR 编码器在 24 GB/batch≥8 下不可行**（旧文档把 P2 进检测头，已修正）；P2 仅由细节支路保留，服务掩码分支/ROI 特征。
- SAHI 物理口径：GWHD 图 1024²、输入 1024 时 **slice=1024 无等效放大**；要获得微小目标增强必须"**512 切片 → resize 1024（2×）**"，基线定义为"整图 1024 单次推理"（旧 Gate 3 的"+10% 召回/省 25% 算力"承诺在新口径下重新定义）。

### 2.3 许可证判定变更（审计 L-01/L-02/L-03）

| 组件 | 旧文档 | v3.0（核验 2026-11-10） |
|---|---|---|
| DINOv2 代码/权重 | 代码 Apache-2.0；**权重 CC-BY-NC-4.0** | **HF 模型卡 `license: apache-2.0`（代码+权重）**；旧 NC 表述作废 |
| DEIM（原版） | "待核验" | **Apache-2.0** |
| DEIMv2 / EdgeCrafter | 未提及 | **自定义 NC 许可（禁止商用）**——仅非商业研究；商用换 D-FINE/RF-DETR |
| DINOv3 | 未提及 | **DINOv3 License（非 NC、可商用附条件）**，权重 gated，需接受记录 |
| SAM 2.1 / SAM 3.1 | Apache-2.0 | SAM 2.1 Apache-2.0；**SAM 3.1 为 SAM License（自定义、gated）** |
| RT-DETR | 混写"Ultralytics RT-DETR"与"Apache-2.0" | **lyuwenyu 实现 Apache-2.0；Ultralytics 实现 AGPL-3.0**，分开表述；集成/交付用前者 |

### 2.4 最新技术增补

1. **DEIMv2 官方 COCO 权重**（HF `Intellindust/DEIMv2_DINOv3_{S,M,L,X}_COCO`）可用于检测头初始化的部分迁移（P1，须 A/B，形状过滤加载）；
2. **EdgeCrafter（TMLR 2026，DEIMv2 后续工作，NC）**作为方法演进对照（检测/实例分割/姿态），不进入交付链；
3. **SAM 3.1 Object Multiplex**（2026-03-27）作为零样本教师升级项；
4. **GWFSS 2025** 语义器官掩码（CC-BY-4.0）用于掩码预训练/教师信号；
5. **TensorRT ≥10.6**（FP16 正确性约束）；
6. CXR 基础模型文献更新：EVA-X（npj Digit Med 2025）、Ark+（Nature 2025，标题更正）、CheXFound（TMI 2025，标题更正）——**仅作文献对比，不引入医学预训练权重**。

---

## 3. 校验复核发现与修正

### 3.1 P0 问题（8 项，全部修复）

| ID | 问题 | 修正 |
|---|---|---|
| T-01 | torch 2.4.1/torchvision 0.19.1 低于 SAM 2.1 最低要求（torch≥2.5.1/0.20.1），Phase 4 不可运行 | 环境升级为 torch 2.10.0+cu128/torchvision 0.25.0；Gate 0 增加版本断言与 SAM2 加载 smoke；删除失败树"保持 2.4.x" |
| N-01 | GWHD"训练集 3,422 图"实为 GWHD 2020 数字；6,522 无法确证 → 迭代数/预算/论证失真 | 改为"以官方 competition_train/val/test.csv 实测行数为唯一口径"（约 6,500 量级）；Gate 1 输出实测数；6.1 预算按公式重算 |
| N-02 | 90-epoch 替代"0.5 d"与 30-epoch 8 h 矛盾；"两段式省 60%"算术不成立 | 改为 90 epochs ≈1 d；明确"是否两段式取决于 AP 差距，不取决于总时长"；删除省 60% 表述 |
| T-02 | SAHI slice=1024 在 1024² 图上无等效放大；小目标召回/省算力承诺不可达 | 改为"512 切片→resize 1024（2×）"实现等效放大；基线=整图 1024 单次推理；Gate 3 口径同步重写 |
| T-03 | P2(1/4)=256² 进 DETR 头（65,536 token/图）与 24GB/batch16 冲突；Frozen-DETR 协议引用失实 | 检测头只用 P3–P5；P2 仅服务掩码/ROI；Frozen-DETR 改为"冻结特征增强思路"并加 1-epoch smoke 前置 |
| T-04 | Phase 6 要求三架构异构链，但成员无训练步骤/预算 → Gate 6 不可达成 | Phase 6 增加成员 ②③ 域微调（各 ≤8 h，+16 h 预算）；成员结果入 experiments.csv |
| L-01 | DINOv2 权重许可写错；DEIM"待核验"；DINOv3/DEIMv2 许可缺失 | 全部更正（见 §2.3），并新增 `licenses/LICENSES.md` Gate 0 要求 |
| E-01 | GLM5.3/ZCode 残留 16 处 + 文件名 + 5 处跨文档引用 | 工作流 v3.0/README/软著/设计报告全部迁移；文件重命名；历史报告加勘误注记；README 索引重建 |

### 3.2 关键 P1 问题（修复摘要）

| ID | 问题 | 修复 |
|---|---|---|
| N-04 | 继续预训练 875→876 iters；4–6 h 预算与 876 iters 实际不符；"1 epoch 达 0.85/反超 CheXzero"无依据 | 改 876；预算"首轮实测后修正"；改为"1 epoch 快速检查 + ≥10 epochs 正式对比"，删除无依据断言 |
| N-05 | "gap=方差+偏差"定义矛盾、"方差≥gap/2"量纲不一致 | 改为 gap=偏差项；方差作抽样不确定度 SE²≈Var/B；判定 |gap|>2SE → 偏差主导，否则方差主导 |
| N-06 | Gate 4 阈值 0.80 依据（WheatSAM 麦穗 Dice）跨域失效 | 0.80 定为设计目标；先测 SIIM 零样本基线，要求绝对 0.80 + 相对提升 ≥1.5 pp；WheatSAM 仅作方法引用 |
| T-05 | 基线金丝雀无文献值/不可执行（YOLOv8 未安装、AGPL） | 改为"环境/数据指纹 + golden 基线冻结（±1%）+ 有来源文献值 ±1–2 pp"；删除 YOLOv8 示例 |
| T-08 | 医学 ROI 通道无监督来源（CXR14 仅 984 框/8 类） | 明确 BBox_List_2017 + TBX/SIIM 框 + 其余 CAM 伪框；Gate 5 按"有框/无框"分组；阈值定义（Youden/分位）写入 config |
| T-09 | CXR14 用全量（含 test）继续预训练 → 转导泄漏；GWHD 多划分未定义协议 | 固定 CXR14 官方 split（train 86,524/test 25,596），continue-pretrain 只用 train；Phase 1 固定唯一划分协议+哈希 |
| T-12 | ONNX/TensorRT 导出无风险处理 | Phase 3 收尾做 1 图 ONNX smoke（opset≥17）；失败处理写入决策树；TensorRT ≥10.6 |
| C-04 | A1–D4 标签在正文大量使用但无定义，违反"自包含" | 新增 §2.5 标签对照表（A1–D4 共 18 项） |
| C-05/C-06 | 配置 batch 8 与正文 16 矛盾；cosine 与 Mahalanobis 矛盾 | 统一 batch=8（24GB 可试 16 须实测登记）；配置改 `dist: mahalanobis` |
| C-07/C-08 | 设计报告未同步：ConvNeXt"前 3 层"、同种子当异构 | 设计报告改为 stem+stage1；集成明确"同种子/多尺度 TTA 为同质成员" |
| C-13 | 软著技术特点与十创新点非 1:1 | 重排为 10 项 1:1 表 + 工程规范单列 |
| C-16 | 数学审查范围止于 v2.1 却给全局结论 | 加勘误注记，结论以第六轮与 v3.0 为准 |
| T-06 | 安装步骤缺 sam2 `-e`、DEIMv2 requirements、pytest/onnx 等 | 3.2 命令补齐（上游 requirements 以仓库实际文件为准；pip freeze 锁定） |
| T-07 | 特征缓存边界未按 Phase 列清单 | Phase 5 明确"可缓存/禁止缓存"清单（检测/分割/几何增强禁止缓存） |
| T-13 | 生长阶段 Gate 5 为无条件项但自建数据为条件项 | 改为条件项（无数据则"无数据-不可评"，不作通过条件） |

### 3.3 引用与事实核验（详见《技术可行性验证报告 v3.0》）

- **无整篇虚构文献**；8 处标题/年份/出处更正（CheXFound 标题≠模型名、Ark+ 标题、UniHSFormer-X 标题、SAM 2.1 年份、IFAC/LF-GAN/AOR 标题补全、KNNOR 许可）。
- 2 条无出处断言删除（SAM2 太阳能板/SAM2CT 对比、光流配准 2–4 px）。
- GWHD："275,187 头/16 机构/12 国"可证；"6,522 图/3,422 训练图"改为实测口径；挑战指标为 Average Domain Accuracy@IoU0.5（非 AP@50），Gate 3 的 0.90 是项目目标值。
- CXR14 GCS 桶为 **Requester Pays**；SIIM 含掩码数量未核实（删除旧"约 2,600"数字）；GWFSS 语义非实例、与 GWHD 配对性未证实。

### 3.4 未修复/降级处理项（如实声明）

| 项 | 处理 |
|---|---|
| 独立审计 68 项中的部分 P2（如个别注释措辞、交付物清单细节） | 按影响已处理大部分；剩余列入 §5 后续工作，不影响 Gate 可执行性 |
| G-02 "单一事实源 + 预推送 lint"、G-03 "执行协议独立分层" | **本轮未实施**，列为后续工作（建议 `docs/EVIDENCE.md` + `scripts/doc_lint.py` + `docs/EXECUTION_PROTOCOL.md`） |
| CI（markdown lint/link check）、CHANGELOG | 本轮新增 `.gitignore` 与 `ATTRIBUTION.md`；CI/CHANGELOG 列为后续工作 |
| "6,522 图"等无法确证数字 | 已改为实测口径，不再作为事实断言 |

---

## 4. 文档与版本治理

### 4.1 变更文件与版本对照

| 文件 | 旧版本 | 新版本/状态 |
|---|---|---|
| MEGA-Vision-GLM5.3实施工作流.md | v2.4 | **重命名为 MEGA-Vision-GPT6-Astra实施工作流.md（v3.0）** |
| MEGA-Net-多专家协同自适应图像识别算法设计报告.md | v1.3 | **v1.4** |
| MEGA-Vision-技术可行性验证报告.md | v2.0 | **v3.0（重写）** |
| MEGA-Vision-第六轮审查与Astra适配报告.md | — | **新增 v1.0（本报）** |
| MEGA-Vision-软著申请方案.md | v1.0 | **v1.1** |
| README.md | 旧索引 | 已更新索引/版本/快速使用/许可边界 |
| MEGA-Vision-原理分析与工作流纠错报告.md | v1.0 | 加**勘误注记**（历史记录保留） |
| MEGA-Vision-数学逻辑审查报告.md | v1.0 | 加**勘误注记** |
| MEGA-Vision-工作流前瞻性优化报告.md | v1.0 | 加**勘误注记** |
| ATTRIBUTION.md / .gitignore | — | **新增**（第三方归属与许可、产物忽略规则） |
| archive/v2.4-20261110/ + tag | — | **已建**（8 文件，与旧正文逐字节一致） |

### 4.2 归档与提交规范

本轮遵循仓库既有流程：先把旧版复制入 `archive/v2.4-20261110/` 并单独 commit + tag，再更新正文并 commit；git 历史保留全部版本，可随时回溯。

---

## 5. 遗留风险与未验证项（须在执行时闭环）

1. **gated 权重审批**：DINOv3（HF `facebook/dinov3-vitl16-pretrain-lvd1689m`）与 SAM 3.1（HF `facebook/sam3.1`）需接受许可/申请访问，审批时长不可控 → 工作流已给出 DINOv2/SAM 2.1 回退路线并登记偏离。
2. **GWHD 官方 CSV 行数**：10.2 GB 压缩包内 CSV 行数未实测 → Phase 1 实测后重算迭代数与预算（工作流已按公式化）。
3. **SIIM 含掩码图像数**：未核实 → Phase 1 统计后写入报告。
4. **GWFSS 与 GWHD 配对性**：仅语义器官掩码、配对性未证实 → 仅作预训练/教师信号，50 图验证后用。
5. **torch 2.15/torchvision 0.29 与上游组件兼容性**：官方未确认 → 默认钉 2.10/0.25；升级须按三环规则复验 Gate 0 + 金丝雀。
6. **GPU 实测**：本报告所有训练时长/显存均为公开参数与 FLOPs 量级估算，无 GPU 实测 → 执行时以 `experiments.csv` 实测为准。
7. **医学合规**：CXR14 等外部医学数据仅研究用途；若目标场景禁止外部医学数据，工作流要求改为官方数据或跳过并取得用户确认。

---

## 6. 结论

本轮在**不改变 MEGA-Net 十大创新点与九阶段/Gate 框架**的前提下，完成了三件事：
1. **执行模型迁移**：GLM5.3 → GPT-6 Astra，并把 Astra 的能力与局限转化为 14 条可执行纪律（身份注入、状态外部化、证据链、知识截止、越界白名单、成本控制）；
2. **技术栈代际刷新**：DINOv3/DEIMv2/SAM 2.1+SAM 3.1/新环境，同时把许可证、gated、NC、GCS 计费等边界写清楚，并给出全部降级路线；
3. **校验复核闭环**：8 项 P0 全部修复、关键 P1 逐项修正、引用与数字更正、历史报告加勘误注记、版本与归档闭环。

新版文档满足"自包含、可执行、可审计、许可边界清晰"的发布标准；遗留项均为**执行时可实测闭环**的事项，不影响仓库文档作为"严谨最新版本"发布。
