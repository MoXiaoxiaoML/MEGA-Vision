# 第三方归属与许可证声明（ATTRIBUTION）

本仓库为 MEGA-Vision 项目的**文档仓库**（设计报告、实施工作流、审查报告），不包含第三方代码或模型权重。文中引用/规划使用的第三方组件、论文与数据集归属如下（许可证以各上游官方发布页面为准，核验日期 2026-11-10；执行项目时须按工作流第 2.4 节登记到项目内 `licenses/LICENSES.md`）。

## 一、模型与代码组件

| 组件 | 来源 | 许可证 | 使用边界 |
|---|---|---|---|
| DINOv3（含 ViT-L/16 权重） | Meta AI Research — https://github.com/facebookresearch/dinov3 | DINOv3 License（自定义，非 NC，可商用附条件）；权重 **gated** | 需接受协议并登记；分发衍生品附协议；禁军事/ITAR 用途 |
| DINOv2（备选底座） | Meta AI Research — https://github.com/facebookresearch/dinov2 | Apache-2.0（代码与 HF 权重模型卡） | 可商用，保留声明 |
| DEIMv2（检测训练配方） | Intellindust AI Lab — https://github.com/Intellindust-AI-Lab/DEIMv2 | **DEIMv2 License（NC，禁止商用）** | 仅非商业研究；商用/闭源交付前替换 |
| EdgeCrafter（DEIMv2 后续工作，对比参考） | Intellindust AI Lab — https://intellindust-ai-lab.github.io/projects/EdgeCrafter/ | EdgeCrafter License（NC） | 同上 |
| DEIM（原版，方法参考） | Intellindust AI Lab — https://github.com/Intellindust-AI-Lab/DEIM | Apache-2.0 | 可商用 |
| SAM 2.1 | Meta FAIR — https://github.com/facebookresearch/sam2 | Apache-2.0 | 可商用，保留声明 |
| SAM 3 / SAM 3.1 | Meta Superintelligence Labs — https://github.com/facebookresearch/sam3 | SAM License（自定义、gated、可商用附条件） | 需申请访问；交付申报 |
| D-FINE（合规检测备选） | https://github.com/Peterande/D-FINE | Apache-2.0 | 可商用 |
| RF-DETR（合规检测备选） | Roboflow — https://github.com/roboflow/rf-detr | Apache-2.0 | 可商用 |
| SAHI | https://github.com/obss/sahi | MIT | 可商用 |
| Frozen-DETR（方法参考） | https://github.com/iSEE-Laboratory/Frozen-DETR | Apache-2.0 | 可商用 |
| Ultralytics（YOLO/RT-DETR，对照用途） | https://github.com/ultralytics/ultralytics | AGPL-3.0 | 传染性；仅对照/非交付，商用需企业授权 |
| timm / transformers / peft / torchvision / PyTorch | Hugging Face / Meta / PyTorch Foundation | Apache-2.0 / BSD | 可商用 |
| TensorRT | NVIDIA | 专有（免费工具） | 分发遵守 NVIDIA 条款 |

## 二、数据集

| 数据集 | 来源 | 许可证/条款 | 说明 |
|---|---|---|---|
| GWHD 2021（Global Wheat Head Dataset） | Zenodo 10.5281/zenodo.5092309；https://www.global-wheat.com/ | CC-BY-4.0 | 需署名；10.2 GB；官方划分 CSV 在包内 |
| GWFSS 2025（Global Wheat Full Semantic Organ Segmentation） | Hugging Face `GlobalWheat/GWFSS_v1.0`；Plant Phenomics 2025 DOI 10.1016/j.plaphe.2025.100084 | CC-BY-4.0 | 1,096 张语义器官掩码；需署名 |
| ChestX-ray14 | NIH Clinical Center（CVPR 2017 论文） | 公开研究用途 | 需引用论文并致谢 NIH；GCS 桶 Requester Pays |
| SIIM-ACR Pneumothorax | Kaggle 竞赛 | 竞赛/科研条款 | 需登录并接受规则；仅科研 |
| TBX11K | https://github.com/yun-liu/Tuberculosis（TPAMI 2023） | 科研用途（以仓库/论文为准） | 测试集标注需挑战注册 |
| WHU-Hi 高光谱 | 武汉大学 RSIDEA — http://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm | 学术研究用途（以官方页面为准） | 需按官方要求申请/引用 |
| COCO 2017 | https://cocodataset.org/ | CC-BY-4.0（图像来源条款见官方） | 检测头初始化使用 |

## 三、论文与文献

本仓库文档引用的论文、会议与期刊成果，其著作权归原作者与出版机构所有；引用仅用于学术说明与技术选型论证，具体文献清单与核验结果见《MEGA-Vision-技术可行性验证报告.md》v3.0 第 3 节与《MEGA-Net-多专家协同自适应图像识别算法设计报告.md》第 5 节。

## 四、免责声明

- 本仓库为**文档与研究报告**，不构成医疗器械或临床诊断产品；所有医学相关内容仅供研究辅助，不作临床诊断依据。
- 文中指标为设计目标/公开文献量级，最终以执行项目时 Gate 实测为准。
- 第三方许可证可能随上游更新而变化，使用前请以官方 LICENSE 原文为准。
