# MEGA-Vision 项目文档

「慧眼」MEGA-Vision：多专家协同自适应图像识别平台（核心算法 MEGA-Net，Multi-Expert Grounded Adaptive Network）。口号：一眼见微，万物可识。

## 文档索引

| 文档 | 版本 | 说明 |
|---|---|---|
| MEGA-Net-多专家协同自适应图像识别算法设计报告.md | v1.1 | 算法原理、十大创新点、关键数学定义、可行性、预估效果 |
| MEGA-Vision-GLM5.3实施工作流.md | **v2.2** | **执行文档**：自包含、九阶段、Gate 验收体系，供 GLM5.3 在 Linux + NVIDIA GPU 环境逐阶段实施 |
| MEGA-Vision-技术可行性验证报告.md | v1.0 | 组件/依赖/数据集/验收阈值的实证核验结论 |
| MEGA-Vision-原理分析与工作流纠错报告.md | v1.0 | 10 项原理纠错 + 步骤优化 + 学习效率提升 |
| MEGA-Vision-工作流前瞻性优化报告.md | v1.0 | 高阶模型视角的进阶修改建议（已并入工作流 v2.0） |
| MEGA-Vision-数学逻辑审查报告.md | v1.0 | 第四轮审查：公式数值复核、8 项修正、逻辑闭环核验 |
| MEGA-Vision-软著申请方案.md | v1.0 | 软件著作权登记：名称、源程序/文档材料规划、技术特点、合规清单、流程 |

## 版本归档规范

- `archive/v2.1-20261004/`：v2.1 升级为 v2.2 前的完整存档，对应 git tag `archive-v2.1-20261004`；
- 更新流程：本地修改文档 → 旧版复制入 `archive/<版本>-<日期>/` 并提交 → 更新正文 → commit + push；
- git 提交历史本身保留全部历史版本，可随时 `git checkout` 回溯。

## 快速使用

1. 目标环境：Linux x86_64 + NVIDIA GPU ≥ 24GB（CUDA ≥ 12.1，PyTorch 2.4.1）。
2. 将《MEGA-Vision-GLM5.3实施工作流.md》全文交给 GLM5.3（ZCode 会话），按工作流第 0 部分总则执行。
3. 九阶段逐 Phase 实施，Gate 由 pytest 代码裁决（`tests/gates/`），环境授权按 3.2 节三环规则。

## 命名与合规

- 项目中文名：「慧眼」多专家协同智能图像识别平台；代号 MEGA-Vision；核心算法 MEGA-Net。
- 软著登记名建议：慧眼多专家协同自适应图像识别平台软件 V1.0（见软著申请方案）。
- 开源合规：DINOv2 权重为 CC-BY-NC（非商业），Ultralytics 为 AGPL-3.0——商用前需替换相关路径（详见软著申请方案第五节）。
