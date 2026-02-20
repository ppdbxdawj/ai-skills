#!/usr/bin/env python3
"""
Single-image evaluation: multiple metrics (PartiPrompt-style + GenAI-Bench + quality/aesthetic).
Outputs one unified HTML report.
Usage:
  python evaluate_image.py --image path/to/image.png "生图用的 prompt"
  python evaluate_image.py --image path/to/image.png --prompt "description"
  python evaluate_image.py --image path/to/image.png "description" --output-dir reports --no-vqa
  With --output-dir, each run writes a unique file (e.g. reports/report_2026-02-20_16-30-45.html).
"""

import argparse
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

def load_image_as_base64(path: Path, max_size: int = 800) -> str:
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        ratio = max_size / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_clip_score(image_path: Path, prompt: str, device: str = "cpu") -> float:
    import torch
    from PIL import Image
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    image = Image.open(image_path).convert("RGB")
    # CLIP max 77 tokens; truncate long prompts
    inputs = processor(
        text=[prompt], images=image, return_tensors="pt", padding=True,
        truncation=True, max_length=77
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image.cpu()
    score = torch.sigmoid(logits_per_image).item()
    return round(min(1.0, max(0.0, score)), 4)


def compute_vqa_score(image_path: Path, prompt: str, device: str = "cpu") -> float | None:
    """VQAScore-style: BLIP-2 answers 'Does this image show [prompt]? Yes or no.' -> P(yes)."""
    try:
        import torch
        from PIL import Image
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        model_id = "Salesforce/blip2-opt-2.7b"
        processor = Blip2Processor.from_pretrained(model_id)
        model = Blip2ForConditionalGeneration.from_pretrained(model_id).to(device)

        image = Image.open(image_path).convert("RGB")
        q = f"Does this image show the following? {prompt}. Answer yes or no."
        inputs = processor(images=image, text=q, return_tensors="pt").to(device)

        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits[0, -1, :].cpu().float()
        tokenizer = processor.tokenizer
        yes_ids = tokenizer.encode("yes", add_special_tokens=False)
        no_ids = tokenizer.encode("no", add_special_tokens=False)
        yes_id = yes_ids[0] if yes_ids else tokenizer.unk_token_id
        no_id = no_ids[0] if no_ids else tokenizer.unk_token_id
        if yes_id == tokenizer.unk_token_id or no_id == tokenizer.unk_token_id:
            return None
        log_yes = logits[yes_id].item()
        log_no = logits[no_id].item()
        import math
        exp_yes = math.exp(min(50, log_yes))
        exp_no = math.exp(min(50, log_no))
        score = exp_yes / (exp_yes + exp_no + 1e-8)
        return round(min(1.0, max(0.0, score)), 4)
    except Exception as e:
        print(f"VQAScore skipped: {e}", file=sys.stderr)
        return None


def compute_quality_sharpness(image_path: Path) -> float:
    """No-reference quality proxy: Laplacian sharpness (0–1 normalized)."""
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        lap = cv2.Laplacian(img, cv2.CV_64F)
        var = lap.var()
        # Normalize: typical range ~0–2000+, cap and scale to 0–1
        norm = min(1.0, var / 500.0)
        return round(norm, 4)
    except Exception as e:
        print(f"Quality (sharpness) skipped: {e}", file=sys.stderr)
        return None


def compute_aesthetic_clip(image_path: Path, device: str = "cpu") -> float | None:
    """Aesthetic proxy: CLIP similarity to 'aesthetic professional photo' vs 'low quality'."""
    try:
        import torch
        from PIL import Image
        from transformers import CLIPProcessor, CLIPModel

        model_id = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_id).to(device)
        processor = CLIPProcessor.from_pretrained(model_id)

        image = Image.open(image_path).convert("RGB")
        texts = ["aesthetic, professional photography, high quality, appealing", "low quality, blurry, unappealing"]
        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        logits = out.logits_per_image[0].cpu()
        probs = torch.softmax(logits, dim=0)
        score = probs[0].item()
        return round(min(1.0, max(0.0, score)), 4)
    except Exception as e:
        print(f"Aesthetic (CLIP) skipped: {e}", file=sys.stderr)
        return None


def compute_image_reward(image_path: Path, prompt: str, device: str = "cpu") -> float | None:
    """ImageReward: 人类偏好奖励模型，与 CLIP/美学等相比更贴合人工评判。可 pip install image-reward。"""
    try:
        import ImageReward as RM  # pip install image-reward
        model = RM.load("ImageReward-v1.0")
        # score(prompt, image_path) 或 image_path 列表；单图传路径即可
        score = model.score(prompt, str(image_path))
        if score is None:
            return None
        # 原始分约在 [-2,2] 区间，归一化到 0–1 便于与其它指标一致展示
        import math
        norm = 1.0 / (1.0 + math.exp(-float(score)))
        return round(min(1.0, max(0.0, norm)), 4)
    except Exception as e:
        print(f"ImageReward skipped: {e}", file=sys.stderr)
        return None


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _bar_pct(value: float | None) -> int:
    if value is None:
        return 0
    return min(100, max(0, int(value * 100)))


def _cell_score(value: float | None, bar: bool = True) -> str:
    if value is None:
        return '<span class="muted">—</span>'
    bar_w = _bar_pct(value)
    bar_html = f'<div class="bar-wrap"><div class="bar" style="width: {bar_w}%;"></div></div>' if bar else ""
    return f'<span class="score">{value}</span>{bar_html}'


def build_html_report(
    image_path: Path,
    prompt: str,
    metrics: dict,
    output_path: Path,
    image_base64: str,
) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    prompt_esc = _escape(prompt)

    rows = [
        ("图文相关性", "CLIPScore (0–1)", _cell_score(metrics.get("clip_score"))),
        ("图文相关性", "VQAScore (0–1)", _cell_score(metrics.get("vqa_score"))),
        ("图文相关性", "ImageReward (0–1)", _cell_score(metrics.get("image_reward"))),
        ("生成质量", "锐度 Laplacian (0–1)", _cell_score(metrics.get("quality_sharpness"))),
        ("美学/吸引力", "CLIP 美学代理 (0–1)", _cell_score(metrics.get("aesthetic_clip"))),
        ("人体真实感", "BodyMetric", '<span class="muted">需专用模型</span>'),
        ("人体失真检测", "ViT-HD", '<span class="muted">需专用模型</span>'),
        ("可解释热力图", "RAHF", '<span class="muted">需专用模型</span>'),
    ]
    rows_html = "".join(
        f'<tr><td>{dim}</td><td>{ind}</td><td>{res}</td></tr>'
        for dim, ind, res in rows
    )

    def _sumval(k):
        v = metrics.get(k)
        return str(v) if v is not None else "—"
    summary_items = [
        ("CLIP", _sumval("clip_score")),
        ("VQA", _sumval("vqa_score")),
        ("IR", _sumval("image_reward")),
        ("锐度", _sumval("quality_sharpness")),
        ("美学", _sumval("aesthetic_clip")),
    ]
    summary_html = "".join(
        f'<span class="summary-item"><b>{label}</b> {val}</span>'
        for label, val in summary_items
    )

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
  <title>文生图评估报告 | 多维度统一</title>
  <style>
    :root {
      --bg: #0f0f12; --card: #18181c; --border: #2a2a30; --text: #e4e4e7; --muted: #71717a; --accent: #a78bfa; --green: #34d399;
      --pad: clamp(0.75rem, 3vw, 1.5rem);
      --gap: clamp(0.75rem, 2.5vw, 1.25rem);
      --radius: clamp(6px, 1.5vw, 10px);
      --fs-sm: clamp(0.7rem, 1.5vw, 0.8rem);
      --fs-md: clamp(0.8rem, 1.8vw, 0.9rem);
      --fs-lg: clamp(1rem, 2.5vw, 1.35rem);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html { -webkit-text-size-adjust: 100%; }
    body {
      font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: var(--bg); color: var(--text); min-height: 100vh; line-height: 1.5;
      padding: 0 0 var(--pad); padding-left: env(safe-area-inset-left); padding-right: env(safe-area-inset-right);
    }
    .sticky-summary {
      position: sticky; top: 0; z-index: 10; background: var(--card); border-bottom: 1px solid var(--border);
      padding: 0.5rem var(--pad); display: flex; flex-wrap: wrap; gap: var(--gap); align-items: center;
      font-size: var(--fs-sm); padding-left: max(var(--pad), env(safe-area-inset-left));
    }
    .sticky-summary .summary-item { color: var(--muted); white-space: nowrap; }
    .sticky-summary .summary-item b { color: var(--green); margin-right: 0.2em; }
    .sticky-summary .summary-time { margin-left: auto; color: var(--muted); }
    @media (max-width: 480px) { .sticky-summary .summary-time { margin-left: 0; width: 100%; } }
    .container {
      max-width: min(1100px, 100%); margin: 0 auto; padding: var(--pad);
      padding-left: max(var(--pad), env(safe-area-inset-left)); padding-right: max(var(--pad), env(safe-area-inset-right));
    }
    h1 { font-size: var(--fs-lg); font-weight: 600; margin-bottom: 0.2em; background: linear-gradient(135deg, #fff 0%, var(--accent) 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; word-break: break-word; }
    .sub { color: var(--muted); font-size: var(--fs-sm); margin-bottom: 0.75rem; word-break: break-all; }
    .grid {
      display: grid;
      grid-template-columns: 1fr minmax(min(100%, 280px), 360px);
      gap: var(--gap); align-items: start;
    }
    @media (max-width: 700px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
      padding: var(--pad); min-width: 0;
    }
    .preview { border-radius: var(--radius); overflow: hidden; max-width: 100%; }
    .preview img {
      width: 100%; height: auto; display: block; object-fit: contain; background: #0a0a0c;
      max-height: min(42vh, 420px);
    }
    @media (max-width: 700px) { .preview img { max-height: min(50vh, 360px); } }
    .prompt-wrap { margin-top: var(--gap); }
    .prompt-wrap .label { color: var(--muted); font-size: var(--fs-sm); margin-bottom: 0.3em; }
    .prompt-box {
      color: var(--muted); font-size: var(--fs-sm); padding: 0.5rem 0.6rem;
      background: rgba(255,255,255,0.04); border-radius: 6px;
      max-height: clamp(6em, 18vw, 10em); overflow-y: auto; -webkit-overflow-scrolling: touch;
    }
    .metrics-card { position: sticky; top: calc(2.5rem + env(safe-area-inset-top)); }
    @media (max-width: 700px) { .metrics-card { position: static; } }
    .table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; margin: -0.25rem 0; }
    table { width: 100%; min-width: 240px; border-collapse: collapse; font-size: var(--fs-md); }
    th, td { text-align: left; padding: 0.45em 0.35em 0.45em 0; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { color: var(--muted); font-weight: 500; font-size: var(--fs-sm); }
    .score { font-weight: 600; color: var(--green); }
    .muted { color: var(--muted); }
    .bar-wrap { height: 5px; background: var(--border); border-radius: 3px; overflow: hidden; margin-top: 2px; }
    .bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); border-radius: 3px; }
    .footer { margin-top: var(--gap); color: var(--muted); font-size: var(--fs-sm); word-break: break-all; }
    .note { color: var(--muted); font-size: var(--fs-sm); margin-top: 0.6rem; line-height: 1.4; }
  </style>
</head>
<body>
  <div class="sticky-summary">
    <span style="color: var(--muted); font-weight: 600;">文生图评估</span>
    """ + summary_html + """
    <span class="summary-time muted">""" + ts + """</span>
  </div>
  <div class="container">
    <h1>文生图评估报告</h1>
    <p class="sub">""" + image_path.name + """ · 多维度统一</p>

    <div class="grid">
      <div class="card">
        <div class="preview"><img src="data:image/jpeg;base64,""" + image_base64 + """" alt="Evaluated image" /></div>
        <div class="prompt-wrap">
          <div class="label">生图 Prompt</div>
          <div class="prompt-box">""" + prompt_esc + """</div>
        </div>
      </div>
      <div class="card metrics-card">
        <div class="table-wrap">
          <table>
            <tr><th>维度</th><th>指标</th><th>结果</th></tr>
            """ + rows_html + """
          </table>
        </div>
        <p class="note">本报告为文生图单图评估。CLIPScore / VQAScore / ImageReward / 锐度 / CLIP 美学可本地计算；BodyMetric、ViT-HD、RAHF 见 reference.md。</p>
      </div>
    </div>

    <p class="footer">image-evaluation · """ + image_path.name + """ · """ + ts + """</p>
  </div>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Single-image evaluation, unified HTML report.")
    parser.add_argument("--image", required=True, type=Path, help="Path to image file")
    parser.add_argument("prompt", nargs="?", default=None, help="生图用的 prompt（与 --prompt 二选一）")
    parser.add_argument("--prompt", dest="prompt_opt", default=None, help="生图用的 prompt（与位置参数二选一）")
    parser.add_argument("--output", type=Path, default=Path("report.html"), help="Output HTML path")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for report (each run: report_YYYY-MM-DD_HH-MM-SS.html)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--no-vqa", action="store_true", help="Skip VQAScore (BLIP-2, heavy)")
    parser.add_argument("--no-imagereward", action="store_true", help="Skip ImageReward (pip install image-reward)")
    args = parser.parse_args()
    prompt = args.prompt or args.prompt_opt
    if not prompt:
        parser.error("请提供生图用的 prompt（位置参数或 --prompt）")

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        # 每次生成唯一文件名，避免覆盖
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = args.output_dir / f"report_{ts}.html"
    else:
        out_path = args.output

    metrics = {}

    print("Computing CLIPScore...")
    metrics["clip_score"] = compute_clip_score(args.image, prompt, device=args.device)
    print(f"  CLIPScore: {metrics['clip_score']}")

    if not args.no_vqa:
        print("Computing VQAScore (BLIP-2)...")
        metrics["vqa_score"] = compute_vqa_score(args.image, prompt, device=args.device)
        print(f"  VQAScore: {metrics.get('vqa_score')}")
    else:
        metrics["vqa_score"] = None

    if not args.no_imagereward:
        print("Computing ImageReward...")
        metrics["image_reward"] = compute_image_reward(args.image, prompt, device=args.device)
        print(f"  ImageReward: {metrics.get('image_reward')}")
    else:
        metrics["image_reward"] = None

    print("Computing quality (sharpness)...")
    metrics["quality_sharpness"] = compute_quality_sharpness(args.image)
    print(f"  Quality (sharpness): {metrics.get('quality_sharpness')}")

    print("Computing aesthetic (CLIP)...")
    metrics["aesthetic_clip"] = compute_aesthetic_clip(args.image, device=args.device)
    print(f"  Aesthetic: {metrics.get('aesthetic_clip')}")

    print("Building report...")
    image_b64 = load_image_as_base64(args.image)
    build_html_report(args.image, prompt, metrics, out_path, image_b64)

    result = {"image": str(args.image), "prompt": prompt, "metrics": metrics, "report": str(out_path)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Report saved: {out_path}")


if __name__ == "__main__":
    main()
