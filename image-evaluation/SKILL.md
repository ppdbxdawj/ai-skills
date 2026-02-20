---
name: image-evaluation
description: |
  Provides reference for evaluating AI-generated images and text-to-image (T2I) models.
  Covers mainstream benchmarks (PartiPrompt, GenAI-Bench, VQAScore, ImageEval, DrawBench,
  T2I-CompBench++, HRSBench, RAHF, BodyMetric, Human Distortion, Reason-Edit, etc.),
  metrics (FID, CLIPScore, VQAScore, human alignment, realism), and dataset scales.
  Use when the user asks about AI image evaluation, T2I benchmarks, image quality metrics,
  图像评估, 文生图评测, benchmark, VQAScore, FID, 图文对齐, or how to evaluate generated images.
license: MIT
metadata:
  author: 葱姜蒜
  version: "1.0.0"
  homepage: https://github.com/ppdbxdawj/ai-skills
  tags:
    - evaluation
    - t2i
    - benchmark
    - image-quality
    - text-image-alignment
---

# Image Evaluation Skill | 图像评估技能

Helps choose and apply **AI image / T2I evaluation** benchmarks and metrics. Full benchmark details (metrics, datasets, links) are in [reference.md](reference.md).

协助选择和应用 **AI 图像 / 文生图评测** 的基准与指标。各基准的指标、数据集与资料链接见 [reference.md](reference.md)。

## When to use | 何时使用

- 需要评估文生图/生成图像质量、图文对齐、人体真实感、组合能力等
- 选 benchmark（如 GenAI-Bench、PartiPrompt、ImageEval、T2I-CompBench++）
- 选指标（VQAScore、FID、CLIPScore、人评、热力图/可解释反馈）
- 查数据集量级与评测方式

## Quick reference | 速查

| 需求 | 推荐 |
|------|------|
| 通用图文对齐 | GenAI-Bench + **VQAScore** |
| 组合/推理 | T2I-CompBench++、HRSBench、DALL-EVAL |
| 中文 + 细粒度 | ImageEval |
| 人体真实感/失真 | BodyMetric、Human Distortion Benchmark |
| 可解释与改进 | RAHF（热力图 + 分数） |
| 指令编辑 | Reason-Edit |

## Single-image evaluation (unified report)

Given one image and one prompt, run **multi-metric evaluation** and get **one HTML report** with:

| 维度 | 指标 | 说明 |
|------|------|------|
| 图文相关性 | CLIPScore | 0–1 |
| 图文相关性 | VQAScore | GenAI-Bench 风格，BLIP-2，可选 `--no-vqa` |
| 图文相关性 | **ImageReward** | 人类偏好奖励模型，与人工评判更一致，可选 `--no-imagereward` |
| 生成质量 | 锐度 Laplacian | 无参考清晰度 0–1 |
| 美学/吸引力 | CLIP 美学代理 | 0–1 |
| 人体真实感/失真/RAHF | — | 需专用模型，见 reference.md |

```bash
cd image-evaluation
python3 -m venv .venv && .venv/bin/pip install -r requirements-eval.txt
.venv/bin/python evaluate_image.py --image /path/to/image.png "生图 prompt" --output-dir reports
# 可选：--no-vqa 跳过 VQAScore；--no-imagereward 跳过 ImageReward
```

报告输出到 `reports/report_YYYY-MM-DD_HH-MM-SS.html`（每次唯一）。布局：顶部粘性摘要 + 左图右表，减少纵向滚动。

## References

- **Full benchmarks & metrics** → [reference.md](reference.md)（总览表 + 各基准说明 + 使用建议）
