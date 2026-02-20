# T2I 评测基准参考 | Text-to-Image Evaluation Benchmarks

整理自公开论文与评测平台，供评估文生图模型（如 Seedream/即梦）时参考。

---

## 总览

| Benchmark | 能力类型 | 业界认可度 | 数据量级 | 主要指标 |
|-----------|----------|------------|----------|----------|
| PartiPrompt | T2I | Google Research，HF 月下载 1.1k+ | 1,600 条 prompt，12 垂类 + 11 挑战 | 人评（质量、图文相关）+ FID、BLEU/CIDEr/METEOR/SPICE |
| ImageEval | T2I | 智源 + 多校，FlagEval 榜单 | 英 1,624 + 中 339 条，多维度标注 | 人评排序 + 实体/风格/细节维度 |
| DrawBench | T2I | Google Research，HF 月下载 641 | ~200 条，11 类 | 人评（质量、图文相关） |
| GenAI-Bench | T2I | CMU + Meta，CVPR SynData24 | 1,600 条（设计师 prompt） | 人评 + **VQAScore**（图文对齐） |
| Image3 (论文) | T2I | Google DeepMind | GenAI-Bench + DrawBench + DOCCI 等 | Elo、总体偏好、图文对齐、视觉吸引力、细节对齐、数字推理 |
| T2I-CompBench++ | T2I | TPAMI 2025，HF 月下载 121 | 8,000 条组合 prompt | 人评 + FID/IS、BLIP-VQA、UniDet、GPT-4V/ShareGPT4V |
| DALL-EVAL | T2I | UNC 等，2022 | ~7,330 视觉推理 + 145 偏见 | 准确率、计数/空间准确率、性别/肤色分布 |
| HRSBench | T2I | KAUST + AWS AI，2023 | 45,000，50 场景、13 技能 | 检测/对齐/情感/保真/偏差等多指标 |
| BodyMetric | T2I | Amazon，arXiv | BodyRealism ~30k 图、~2k prompt | BodyMetric 人体真实感分数、成对偏好 |
| Human Distortion Benchmark | T2I | 北大 + 腾讯，arXiv | Distortion-5K 4,700 张 | Precision/Recall/F1、IoU、Dice，像素/图像/区域级 |
| RAHF | T2I | Google 等，CVPR 2024 Best Paper | RichHF-18K 18k 图-文对 | 热力图 MSE/NSS/KLD/AUC、评分 PLCC/SRCC、关键词 P/R/F1 |
| Reason-Edit | 指令编辑 | 腾讯 ARC，引用 69+ | 219 对 图像-指令 | Ins-align、PSNR/SSIM/LPIPS、CLIP Score |

---

## 本地可执行单图评估方案（本 skill 脚本）

以下指标可在本地通过 `evaluate_image.py` 一次跑出，并生成统一 HTML 报告：

| 指标 | 说明 | 依赖 |
|------|------|------|
| **CLIPScore** | 图文相似度 0–1 | transformers (CLIP) |
| **VQAScore** | GenAI-Bench 风格，BLIP-2 问答 P(yes) | transformers (BLIP-2)，可选 `--no-vqa` 跳过 |
| **ImageReward** | 人类偏好奖励模型，与人工评判更一致 | `pip install image-reward`，可选 `--no-imagereward` 跳过 |
| **锐度 (Laplacian)** | 无参考清晰度 0–1 | OpenCV |
| **CLIP 美学代理** | 与「高质量/低质量」文本的相似度 0–1 | transformers (CLIP) |

BodyMetric、ViT-HD、RAHF 等需各自专用模型与流程，见下方对应小节。

---

## PartiPrompt

- **能力类型**: T2I  
- **业界认可度**: Google Research 提出，HuggingFace 月下载 1,127+  
- **特点**: 补足 MS-COCO 等不足；1,600 条 prompt，12 垂类 + 11 挑战，难度跨度大。  
- **评测**: 人评（画面质量、图文相关性，每对 5 人）；自动：FID，BLEU、CIDEr、METEOR、SPICE。  
- **数据**: 12 垂类方向 + 11 挑战方向。  
- **资料**: [HF: parti-prompts](https://huggingface.co/datasets/nateraw/parti-prompts) · [arXiv:2206.10789](https://arxiv.org/pdf/2206.10789)

---

## ImageEval

- **能力类型**: T2I  
- **业界认可度**: 智源联合多校，[FlagEval 榜单](https://flageval.baai.ac.cn/#/leaderboard)  
- **特点**: 「能力—任务—指标」三维框架；ImageEval-prompt 细粒度评估实体、风格、细节。  
- **评测**: 每 prompt 生成 8 张，标注者盲评排序并选前三，再标是否表达关键信息。  
- **指标**: 实体（物体、状态、颜色、数量、位置）、风格（绘画、文化）、细节（手部、面部、性别、非逻辑知识）。  
- **数据**: 英 1,624（来自 PartiPrompts）、中 339（自动生成）。  
- **资料**: [FlagEval ImageEval README](https://github.com/flageval-baai/FlagEval/blob/master/imageEval/README.md)

---

## DrawBench

- **能力类型**: T2I  
- **业界认可度**: Google Research，HuggingFace 月下载 641  
- **特点**: 多维语义探测：组合性、基数、空间关系、复杂/生僻词、创造性/超出分布。  
- **评测**: 人评 A/B（8 样本 vs 8 样本），选 Model A / same / Model B；维度：样本保真度、图文对齐。  
- **数据**: 11 类、约 200 条文本提示。  
- **资料**: [arXiv:2205.11487](https://arxiv.org/pdf/2205.11487) · [HF: DrawBench](https://huggingface.co/datasets/shunk031/DrawBench)

---

## GenAI-Bench

- **能力类型**: T2I  
- **业界认可度**: CMU + Meta，CVPR SynData24 Workshop  
- **特点**: 技能体系（基础：对象、属性、场景、关系；高级：计数、比较、区分、否定）；人评 + **VQAScore**（VQA 模型算图文匹配），黑盒可用。  
- **评测**: 3 人打图文相关性；客观 VQAScore（CLIP-FlanT5）。  
- **数据**: 1,600 条，Midjourney 设计师常用 prompt。  
- **资料**: [GenAI-Bench](https://linzhiqiu.github.io/papers/genai_bench/) · [arXiv:2406.13743](https://arxiv.org/pdf/2406.13743) · [HF DataViewer](https://huggingface.co/spaces/BaiqiL/GenAI-Bench-DataViewer)

---

## Image3（论文）

- **能力类型**: T2I  
- **业界认可度**: Google DeepMind，多基准 SOTA  
- **特点**: 用 GenAI-Bench、DrawBench、DOCCI-Test-Pivots、DALL·E 3 Eval、GeckoNum 等；Elo 两两比较；多维度评分。  
- **指标**: 总体偏好、图文对齐、视觉吸引力、细节对齐、数字推理。  
- **数据**: GenAI-Bench 1,600、DrawBench 200、DALL·E 3 Eval 170、DOCCI 1,000、GeckoNum 等。  
- **资料**: [arXiv:2408.07009](https://arxiv.org/pdf/2408.07009)

---

## T2I-CompBench++

- **能力类型**: T2I  
- **业界认可度**: TPAMI 2025，NeurIPS 2023 CompBench 增强版，HF 月下载 121  
- **特点**: 8,000 条组合 prompt，属性绑定、对象关系、生成计数、复杂组合；含违反物理常识的挑战；开源数据与评估代码。  
- **评测**: 人评（3 人，1–5 分，图文对齐）+ 自动。  
- **指标**: FID、IS；BLIP-VQA（属性绑定）、UniDet（空间关系）、3-in-1 综合；GPT-4V/ShareGPT4V 组合性。  
- **数据**: 8,000 条，4 类 8 子类（颜色/形状/纹理绑定，2D/3D 与非空间关系，计数，复杂组合）。  
- **资料**: [GitHub: T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) · [arXiv:2307.06350](https://arxiv.org/pdf/2307.06350)

---

## DALL-EVAL

- **能力类型**: T2I  
- **业界认可度**: UNC 等，2022  
- **特点**: 视觉推理（对象识别、计数、空间关系）+ 社会偏见（性别、肤色）；PaintSkills 诊断集。  
- **指标**: 对象识别/计数/空间关系准确率；性别与肤色分布比例。  
- **数据**: 视觉推理约 7,330 条、约 80 类；偏见约 145 条。  
- **资料**: [arXiv:2202.04053](https://arxiv.org/pdf/2202.04053)

---

## HRSBench

- **能力类型**: T2I  
- **业界认可度**: KAUST + AWS AI，2023，全面 T2I 基准之一  
- **特点**: 13 种技能，准确性/鲁棒性/泛化/公平/偏差；50 场景；人评 + 检测/对齐/情感/保真/偏差等。  
- **评测**: AMT 人评 1–5 分；自动：UniDet（计数、空间、属性）、T2I/TIT/I2I 对齐、AC-T2I、情感分类器、面部属性偏差等。  
- **数据**: 45,000，动作、偏见、公平性、数字、空间、情感、反现实、大小、文字等；每技能 3,000 条，简单/中/难。  
- **资料**: [HRSBench](https://eslambakr.github.io/hrsbench.github.io/) · [arXiv:2304.05390](https://arxiv.org/pdf/2304.05390)

---

## BodyMetric

- **能力类型**: T2I（人体真实感）  
- **业界认可度**: Amazon，arXiv  
- **特点**: 专注人体真实感；多模态（文本、图像、3D）；专家标注 + BodyMetric 自动评分。  
- **指标**: 人体真实感分数 [-1,1]、成对偏好准确率、排序能力。  
- **数据**: BodyRealism ~30k 图（28,917 生成 + 1,705 真实）、~2k prompt；SMPL-X 3D；三档（严重/适度失真、高真实）。  
- **资料**: [arXiv:2412.04086](https://arxiv.org/pdf/2412.04086)

---

## Human Distortion Benchmark

- **能力类型**: T2I（人体失真）  
- **业界认可度**: 北大 + 腾讯，arXiv，GitHub star 14  
- **特点**: 人体失真系统评估（增生、缺失、畸形、融合等）；ViT-HD 检测失真区域；像素/图像/区域级。  
- **指标**: Precision、Recall、F1、IoU、Dice；图像级「是否失真」、区域级 P/R。  
- **数据**: Distortion-5K 4,700 张（4k/300/400 划分），多边形掩码 + 失真类型。  
- **资料**: [arXiv:2503.00811](https://arxiv.org/abs/2503.00811) · [HF: Distorted-5K](https://huggingface.co/datasets/xgklndsgkl/Distorted-5K)

---

## RAHF

- **能力类型**: T2I  
- **业界认可度**: Google Research 等，**CVPR 2024 Best Paper**  
- **特点**: 细粒度多模态反馈（热力图、关键词、多维度评分）；RAHF 预测热力图 + 分数，可解释、可指导 inpainting/数据筛选。  
- **指标**: 热力图 MSE、NSS、KLD、AUC-Judd、SIM、CC；评分 PLCC、SRCC；未匹配关键词 P/R/F1。  
- **数据**: RichHF-18K 18,000 图-文对（16k/1k/1k），来自 Pick-a-Pic；热力图、未对齐关键词、四类评分（合理性、对齐、美学、综合）。  
- **资料**: [arXiv:2312.10240](https://arxiv.org/abs/2312.10240) · [RichHF-18K](https://github.com/google-research/google-research/tree/master/richhf)

---

## Reason-Edit

- **能力类型**: 指令编辑（Instruction-based Editing）  
- **业界认可度**: 腾讯 ARC Lab + Tencent AI Lab，引用 69+  
- **特点**: 专注「复杂理解 + 复杂推理」指令编辑；补 MagicBrush/InstructPix2Pix 在多对象、空间、常识推理上的空白。  
- **指标**: Instruction-Alignment（Ins-align，4 人平均）；背景 PSNR/SSIM/LPIPS；前景 CLIP Score。  
- **数据**: 219 对 图像-指令；复杂理解（位置、颜色、镜中对象、大小等）、复杂推理（常识/世界知识）。  
- **资料**: [arXiv:2312.06739](https://arxiv.org/pdf/2312.06739) · [arXiv:2504.02782](https://arxiv.org/pdf/2504.02782) · [Google Drive](https://drive.google.com/drive/folders/1QGmye23P3vzBBXjVj2BuE7K3n8gaWbyQ)

---

## 使用建议

- **通用图文对齐**: GenAI-Bench + **VQAScore**；可配合 DrawBench、PartiPrompt 人评。  
- **组合与推理**: T2I-CompBench++、HRSBench、DALL-EVAL。  
- **中文 + 细粒度**: ImageEval（中英 prompt + 实体/风格/细节）。  
- **人体**: BodyMetric（真实感）、Human Distortion（失真检测）。  
- **可解释与改进**: RAHF（热力图 + 分数）。  
- **指令编辑**: Reason-Edit。

若要做模型评测或对比，可结合上述基准与指标设计评估方案。
