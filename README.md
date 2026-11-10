# MEGA-Vision 项目文档

「慧眼」MEGA-Vision：多专家协同自适应图像识别平台（核心算法 MEGA-Net，Multi-Expert Grounded Adaptive Network）。口号：一眼见微，万物可识。

> **v3.0 / 2026-11-10**：实施工作流的目标执行模型已由 **GLM5.3（ZCode）** 迁移为 **GPT-6 Astra**（OpenAI，2026-09-03 发布，`gpt-6-astra`），并按 2026-11 最新技术栈刷新（DINOv3、DEIMv2、SAM 2.1/SAM 3.1、PyTorch 2.10.0+cu128、Python 3.12、Ubuntu 24.04）。变更详情见《MEGA-Vision-第六轮审查与Astra适配报告.md》。

## 文档索引

| 文档 | 版本 | 说明 |
|---|---|---|
| MEGA-Net-多专家协同自适应图像识别算法设计报告.md | v1.4 | 算法原理、十大创新点、关键数学定义、§1.2.10 文献融合增强、可行性、预估效果 |
| MEGA-Vision-GPT6-Astra实施工作流.md | **v3.0** | **执行文档**：自包含、九阶段、Gate 验收体系，供 **GPT-6 Astra** 在 Linux + NVIDIA GPU 环境逐阶段实施 |
| MEGA-Vision-技术可行性验证报告.md | **v3.0** | 2026-11 技术栈刷新核验 + 7 项文献融合方法可行性论证 + 引用真伪核验 |
| MEGA-Vision-第六轮审查与Astra适配报告.md | v1.0 | 本轮校验复核：目标模型迁移、技术栈刷新、审计问题清单与修正、Astra 适配风险与对策 |
| MEGA-Vision-原理分析与工作流纠错报告.md | v1.0 | 第二轮审查：10 项原理纠错 + 步骤优化 + 学习效率提升 |
| MEGA-Vision-工作流前瞻性优化报告.md | v1.0 | 第三轮：高阶模型视角的进阶修改建议（已并入工作流 v2.0） |
| MEGA-Vision-数学逻辑审查报告.md | v1.0 | 第四轮审查：公式数值复核、8 项修正、逻辑闭环核验 |
| MEGA-Vision-软著申请方案.md | v1.1 | 软件著作权登记：名称、源程序/文档材料规划、技术特点、合规清单、流程 |

## 版本归档规范

- `archive/` 按版本归档：`v2.1-20261004/`、`v2.2-20261004/`、`v2.3-20261004/`、`v2.4-20261110/`（对应同名 git tag `archive-v<版本>-<日期>`）；
- 更新流程：本地修改文档 → 旧版复制入 `archive/<版本>-<日期>/` 并提交 → 更新正文 → commit + push；
- git 提交历史本身保留全部历史版本，可随时 `git checkout` 回溯。

## 快速使用

1. **目标环境**：Linux x86_64（Ubuntu 24.04 LTS）+ NVIDIA GPU ≥ 24GB（驱动 ≥ 580 / CUDA 运行时 12.8）+ Python 3.12 + PyTorch 2.10.0+cu128 + torchvision 0.25.0。
2. **把《MEGA-Vision-GPT6-Astra实施工作流.md》全文交给 GPT-6 Astra**（Codex/Agent 会话中显式 `-m gpt-6-astra`；模型 ID 不符时工作流要求 Astra 停止并声明）。
3. **三种执行模式**：A 直连执行（Astra 有终端/文件工具，默认）、B 人机协同（Astra 给命令、人执行回贴）、C 纯文本（输出文件内容）。开始前声明档位。
4. **九阶段逐 Phase 实施**：每个 Phase 先跑基线金丝雀，Gate 由 `pytest tests/gates/` 代码裁决（退出码 0 + 日志落盘 + 哈希），不接受"模型自述达标"。
5. **状态与证据**：`PROJECT_STATE.md` / `DECISIONS.md` / `experiments.csv` / `logs/` / `artifacts/manifest.json` 是唯一可信状态；报告中的数字必须有日志路径。

## 命名与合规

- 项目中文名：「慧眼」多专家协同智能图像识别平台；代号 MEGA-Vision；核心算法 MEGA-Net。
- 软著登记名建议：慧眼多专家协同自适应图像识别平台软件 V1.0（见软著申请方案）。
- 第三方归属与许可证汇总见 `ATTRIBUTION.md`；训练产物忽略规则见 `.gitignore`。
- **第三方许可证边界（v3.0 核验）**：
  - **DINOv3**：DINOv3 License（非 NC，可商用但附条件：附协议/致谢/禁军事用途），权重 **gated**，需接受协议并登记；
  - **DINOv2**：代码与权重 Apache-2.0（HF 模型卡 2026-11-10 核验；旧版文档"CC-BY-NC-4.0"表述已过时）；
  - **DEIMv2 / EdgeCrafter**：**NC 许可，禁止商用**——仅限非商业研究/竞赛；商用前必须替换为 D-FINE（Apache-2.0）或 RF-DETR（Apache-2.0）；
  - **SAM 2.1**：Apache-2.0；**SAM 3.1**：SAM License（自定义、gated，可商用附条件）；
  - **Ultralytics（YOLO/RT-DETR）**：AGPL-3.0（传染性，仅作对照/非交付用途）；
  - 数据集：GWHD/GWFSS/WHU-Hi 为 CC-BY-4.0（需署名）；ChestX-ray14 需引用论文并致谢 NIH；SIIM-ACR/TBX11K 按各自科研条款。
- 医学输出免责声明：本项目结果仅供研究辅助，不作临床诊断依据；医学数据仅按各自许可用于研究。
