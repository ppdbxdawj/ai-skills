#!/usr/bin/env python3
"""
Seedream Image Generation via ModelArk (BytePlus Ark API)

与 SeedreamService (TS) 对齐：默认 BytePlus 端点、seedream-4-0-250828，
从 skills/seedream-image/.env.ark 读取 ARK_API_KEY，生成后下载到本地。

单次运行内请求 Ark 平台不超过 MAX_ARK_REQUESTS 次（默认 20），超过即停止。

Usage:
    python generate_ark.py "一只猫在花园里"
    python generate_ark.py --prompt "竖屏9:16封面" --size 1440x2560
    python generate_ark.py --prompt "组图" --sequential auto --max-images 4
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests not found. Install: pip install requests")
    sys.exit(1)

# 与 TS SeedreamService 一致
API_URL = "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"
DEFAULT_MODEL = "seedream-4-0-250828"
MAX_ARK_REQUESTS = 20  # 单次运行内最多请求 Ark 次数，超过即停止


def _load_env_ark() -> None:
    if os.environ.get("ARK_API_KEY"):
        return
    # 先脚本同目录，再当前工作目录
    for base in (Path(__file__).resolve().parent, Path.cwd()):
        for name in (".env.ark", ".env"):
            env_file = base / name
            if not env_file.is_file():
                continue
            for line in env_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip().strip("'\"").strip()
                    if key and key not in os.environ:
                        os.environ[key] = value
            return


def get_api_key() -> str:
    _load_env_ark()
    key = os.environ.get("ARK_API_KEY", "")
    if not key or key == "在此填入你的方舟APIKey":
        print("Error: ARK_API_KEY is not set.")
        print("Edit skills/seedream-image/.env.ark and set ARK_API_KEY=your_key")
        print("Or: export ARK_API_KEY=your_key")
        print("Get key: https://console.volcengine.com/ark → API Key 管理")
        sys.exit(1)
    return key


def build_request_body(
    prompt: str,
    model: str = DEFAULT_MODEL,
    size: str = "2K",
    sequential: str = "disabled",
    max_images: int = 4,
    response_format: str = "url",
    watermark: bool = False,
    image: str | list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    # 与 TS 一致：enabled → auto
    seq_value = "auto" if sequential == "enabled" else sequential
    body: dict = {
        "model": model,
        "prompt": prompt,
        "sequential_image_generation": seq_value,
        "response_format": response_format,
        "size": size,
        "stream": False,
        "watermark": watermark,
    }
    if image is not None:
        body["image"] = image
    if width is not None:
        body["width"] = width
    if height is not None:
        body["height"] = height
    if seq_value == "auto":
        body["sequential_image_generation_options"] = {"max_images": max_images}
    return body


def generate_one(
    prompt: str,
    model: str = DEFAULT_MODEL,
    size: str = "2K",
    sequential: str = "disabled",
    max_images: int = 4,
    watermark: bool = False,
    response_format: str = "url",
    endpoint: str = API_URL,
) -> dict:
    api_key = get_api_key()
    body = build_request_body(
        prompt=prompt,
        model=model,
        size=size,
        sequential=sequential,
        max_images=max_images,
        response_format=response_format,
        watermark=watermark,
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    print("Generating image (Ark)...")
    print(f"  model: {model}, size: {size}, endpoint: {endpoint[:50]}...")
    try:
        resp = requests.post(endpoint, headers=headers, json=body, timeout=120)
    except Exception as e:
        print(f"Request failed: {e}")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"API error HTTP {resp.status_code}:")
        print(resp.text[:800])
        sys.exit(1)
    return resp.json()


def download_to_file(url: str, filepath: Path, timeout: int = 60) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        filepath.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def save_images_to_dir(data: dict, output_dir: str) -> list[dict]:
    """Save images and return a list of artifacts.

    Each artifact is:
      - {"local": "/path/to/file.png", "url": "https://..."} when a URL was provided
      - {"local": "/path/to/file.png", "url": None} when only b64_json was provided
      - {"local": None, "url": "https://..."} if download failed
    """
    images = data.get("data") or []
    if not images:
        return []
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict] = []
    ts = int(time.time())
    for i, img in enumerate(images):
        url = img.get("url")
        b64 = img.get("b64_json")
        if url:
            path = out / f"seedream_ark_{ts}_{i+1}.png"
            if download_to_file(url, path):
                print(f"  Saved: {path} ({path.stat().st_size / 1024:.0f} KB)")
                print(f"  CDN: {url}")
                artifacts.append({"local": str(path), "url": url})
            else:
                print(f"  CDN: {url}")
                artifacts.append({"local": None, "url": url})
        elif b64:
            path = out / f"seedream_ark_{ts}_{i+1}.png"
            path.write_bytes(base64.b64decode(b64))
            print(f"  Saved: {path} ({path.stat().st_size / 1024:.0f} KB)")
            artifacts.append({"local": str(path), "url": None})
    return artifacts


def main():
    parser = argparse.ArgumentParser(description="Seedream via ModelArk (Ark API), download to local.")
    parser.add_argument("prompt", nargs="?", help="Image generation prompt")
    parser.add_argument("--prompt", dest="prompt_opt", default=None, help="Prompt (alternative)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model (default: {DEFAULT_MODEL})")
    parser.add_argument("--size", default="2K", help="Size: 2K, 4K, or WxH e.g. 1440x2560")
    parser.add_argument("--sequential", default="disabled", choices=["auto", "disabled", "enabled"], help="Group: auto/enabled or disabled")
    parser.add_argument("--max-images", type=int, default=4, help="Max images when sequential=auto")
    parser.add_argument("--output-dir", default="output", help="Directory to save images (default: output)")
    parser.add_argument("--watermark", action="store_true", help="Add watermark")
    parser.add_argument("--endpoint", default=API_URL, help="API URL (default: BytePlus)")
    args = parser.parse_args()

    prompt = args.prompt or args.prompt_opt
    if not prompt:
        parser.print_help()
        print("\nError: prompt required.")
        sys.exit(1)

    # 单次运行内不超过 MAX_ARK_REQUESTS 次请求（本次只请求 1 次）
    request_count = 0
    if request_count >= MAX_ARK_REQUESTS:
        print(f"Stop: Ark request count would exceed {MAX_ARK_REQUESTS}.")
        sys.exit(1)

    result = generate_one(
        prompt=prompt,
        model=args.model,
        size=args.size,
        sequential=args.sequential,
        max_images=args.max_images,
        watermark=args.watermark,
        response_format="url",
        endpoint=args.endpoint,
    )
    request_count += 1

    artifacts = save_images_to_dir(result, args.output_dir)
    if not artifacts:
        print("No images in response:")
        print(json.dumps(result, indent=2, ensure_ascii=False)[:1500])
        sys.exit(1)

    # Print a compact machine-readable summary (one line) for downstream tooling
    try:
        print("\nArtifacts:")
        print(json.dumps(artifacts, ensure_ascii=False))
    except Exception:
        pass

    print(f"\nDone. {len(artifacts)} image(s) in {args.output_dir}/")


if __name__ == "__main__":
    main()
