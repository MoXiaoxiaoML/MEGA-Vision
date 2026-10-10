# MEGA-Vision 项目文档

「慧眼」MEGA-Vision：多专家协同自适应图像识别平台（核心算法 MEGA-Net，Multi-Expert Grounded Adaptive Network）。口号：一眼见微，万物可识。

## 文档索引

| 文档 | 说明 |
|---|---|
| MEGA-Net-多专家协同自适应图像识别算法设计报告.md | 算法原理、可行性报告、创新点、预估效果 |
| MEGA-Vision-GLM5.3实施工作流.md | **执行文档（v2.1）**：自包含、九阶段、Gate 验收体系，供 GLM5.3 在 Linux + NVIDIA GPU 环境逐阶段实施 |
| MEGA-Vision-技术可行性验证报告.md | 组件/依赖/数据集/验收阈值的实证核验结论 |
| MEGA-Vision-原理分析与工作流纠错报告.md | 10 项原理纠错 + 步骤优化 + 学习效率提升 |
| MEGA-Vision-工作流前瞻性优化报告.md | 高阶模型视角的进阶修改建议（已全部并入工作流 v2.0） |

## 快速使用

1. 目标环境：Linux x86_64 + NVIDIA GPU ≥ 24GB（CUDA ≥ 12.1，PyTorch 2.4.1）。
2. 将《MEGA-Vision-GLM5.3实施工作流.md》全文交给 GLM5.3（ZCode 会话），并附启动提示词（按工作流第 0 部分总则执行）。
3. 按九阶段逐 Phase 实施，每个 Gate 由 pytest 代码裁决（`tests/gates/`），环境授权按 3.2 节三环规则执行。

## 命名

- 项目中文名：「慧眼」多专家协同智能图像识别平台
- 项目代号：MEGA-Vision；核心算法：MEGA-Net
- 技术栈：DINOv2 冻结底座 + DEIM 检测头 + SAM 2.1 实例分割 + 原型/双通道分类头 + 多模型协同层（可选高光谱支路）
