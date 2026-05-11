"""Manual smoke test for the Qwen3-VL-8B model.

Run after `vllm serve` is up on the remote host (tunneled to :8003) and
the backend is running on :8000.

Usage:
    cd backend
    python scripts/smoke_qwen3_vl.py path/to/image.jpg "What is in this image?"
"""
from __future__ import annotations

import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.request
from pathlib import Path

BACKEND_URL = "http://localhost:8000/api/chat"
MODEL_ID = "qwen3-vl-8b-vllm"
ERROR_PREFIXES = ("Lỗi kết nối", "Xin lỗi")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"Usage: {argv[0]} <image-path> <question>", file=sys.stderr)
        return 1

    image_path = Path(argv[1])
    question = argv[2]

    if not image_path.exists():
        print(f"Image not found: {image_path}", file=sys.stderr)
        return 1

    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    data = base64.b64encode(image_path.read_bytes()).decode()
    data_url = f"data:{mime};base64,{data}"

    payload = {
        "message": question,
        "history": [],
        "image_urls": [data_url],
        "model_id": MODEL_ID,
    }

    req = urllib.request.Request(
        BACKEND_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    reply = body.get("reply", "")
    print(f"Reply: {reply}")

    if not reply:
        print("FAIL: empty reply", file=sys.stderr)
        return 1
    for prefix in ERROR_PREFIXES:
        if reply.startswith(prefix):
            print(f"FAIL: reply starts with error prefix '{prefix}'", file=sys.stderr)
            return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
