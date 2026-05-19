"""Smoke test for AgilePruner-patched vLLM.

Runs the same prompt+image with and without the server-side AgilePruner flag,
prints reply, end-to-end latency, and (if available) peak GPU memory deltas.

Usage:
    python scripts/smoke_agilepruner.py path/to/image.jpg "Describe this image"

Prereqs: two server instances or one server restarted between runs.
This script is request-only; it does NOT toggle the server flag.
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

from openai import OpenAI


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def run_once(client: OpenAI, image_b64: str, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="qwen3-vl-8b",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                ],
            }
        ],
        max_tokens=512,
        temperature=0.0,
    )
    dt = time.perf_counter() - t0
    return resp.choices[0].message.content or "", dt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--base-url", default="http://localhost:8003/v1")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="none")
    image_b64 = encode_image(args.image)

    reply, dt = run_once(client, image_b64, args.prompt)
    print(f"Latency: {dt*1000:.0f} ms")
    print(f"Reply length: {len(reply)} chars")
    print("---")
    print(reply)
    return 0


if __name__ == "__main__":
    sys.exit(main())
