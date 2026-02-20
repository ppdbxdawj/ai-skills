# Image Evaluation Skill | 图像评估技能

**EN:** Reference for evaluating AI-generated images and T2I models: benchmarks (PartiPrompt, GenAI-Bench, ImageEval, T2I-CompBench++, HRSBench, RAHF, etc.), metrics (VQAScore, FID, human alignment), and dataset info.  
**中文：** AI 生成图像与文生图模型评估参考：主流基准、指标与数据集说明。

[English](#english) | [中文](#中文)

---

<a id="english"></a>

## English

### What it does

- Points to **T2I evaluation benchmarks** (PartiPrompt, GenAI-Bench, ImageEval, DrawBench, T2I-CompBench++, HRSBench, DALL-EVAL, BodyMetric, Human Distortion, RAHF, Reason-Edit).
- Summarizes **metrics** (human eval, FID, VQAScore, CLIPScore, realism, alignment).
- Gives **dataset scale & links**; suggests which benchmark to use for alignment, composition, Chinese, human body, or instruction editing.

### Install

```bash
npx skills add ppdbxdawj/ai-skills@image-evaluation
```

### Get a T2I evaluation report (single image)

From the `image-evaluation` directory, run with your generated image and the prompt you used:

```bash
cd image-evaluation
pip install -r requirements-eval.txt
python evaluate_image.py --image /path/to/your_image.png "Your text-to-image prompt" --output-dir reports
# Optional: add --no-vqa to skip BLIP-2 (saves memory)
```

Report is written to `reports/report.html`.

### Reference

Full tables and per-benchmark details → [reference.md](reference.md)

---

<a id="中文"></a>

## 中文

### 功能

- 提供 **T2I 评测基准** 参考（PartiPrompt、GenAI-Bench、ImageEval、DrawBench、T2I-CompBench++、HRSBench、DALL-EVAL、BodyMetric、Human Distortion、RAHF、Reason-Edit）。
- 汇总 **指标**（人评、FID、VQAScore、CLIPScore、真实感、对齐度等）。
- 说明 **数据集量级与资料链接**，并按需求推荐基准（图文对齐、组合、中文、人体、指令编辑等）。

### 安装

```bash
npx skills add ppdbxdawj/ai-skills@image-evaluation
```

### 得出文生图评估报告（单图）

在 `image-evaluation` 目录下，用你的生成图和生图时用的 prompt 运行：

```bash
cd image-evaluation
pip install -r requirements-eval.txt
python evaluate_image.py --image /path/to/你的图片.png "生图时用的 prompt" --output-dir reports
# 可选：加 --no-vqa 跳过 BLIP-2（省显存/内存）
```

报告会生成到 `reports/report.html`，即文生图评估报告。

### 参考文档

总览表与各基准详情 → [reference.md](reference.md)
