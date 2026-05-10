"""Pure helpers for the image compressor: scanning, hashing, rewriting."""
from __future__ import annotations

import base64
import copy
import hashlib
from typing import Iterator, Optional

import httpx


def has_images(msg: dict) -> bool:
    """True if `msg` carries at least one image_url part with non-empty url."""
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        p.get("type") == "image_url" and p.get("image_url", {}).get("url")
        for p in content
    )


def iter_image_parts(msgs: list[dict]) -> Iterator[tuple[int, int, str]]:
    """Yield (msg_idx, content_idx, url) for every image_url part in `msgs`.
    Messages whose content is a string are skipped silently.
    """
    for i, msg in enumerate(msgs):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for j, part in enumerate(content):
            if part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url:
                    yield i, j, url


def find_latest_image_turn(msgs: list[dict]) -> Optional[int]:
    """Index of the latest USER turn that carries images; None if none."""
    latest: Optional[int] = None
    for i, msg in enumerate(msgs):
        if msg.get("role") == "user" and has_images(msg):
            latest = i
    return latest


def text_of(msg: dict) -> str:
    """Pull the first text part out of a message (string content -> as-is,
    list content -> first {type:'text'} entry, else empty)."""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                return part.get("text") or ""
    return ""


async def hash_image_url(
    url: str,
    fetch_base: str,
    fetch_timeout_s: int = 10,
) -> tuple[str, bytes]:
    """Resolve `url` to raw bytes and SHA-256-hash them.

    Supports `data:`, absolute http(s)://, and relative paths (resolved
    against `fetch_base`, e.g. http://127.0.0.1:8000). Empty / malformed
    inputs raise ValueError; non-2xx http responses raise via `raise_for_status`.
    """
    if url.startswith("data:"):
        if "," not in url:
            raise ValueError("malformed data URL")
        raw = base64.b64decode(url.split(",", 1)[1])
        if not raw:
            raise ValueError("data URL payload is empty")
    elif url.startswith(("http://", "https://")):
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(url)
            r.raise_for_status()
            raw = r.content
    elif url.startswith("/"):
        full = f"{fetch_base.rstrip('/')}{url}"
        async with httpx.AsyncClient(timeout=fetch_timeout_s) as client:
            r = await client.get(full)
            r.raise_for_status()
            raw = r.content
    else:
        raise ValueError(f"Unsupported image URL scheme: {url[:32]!r}")
    return hashlib.sha256(raw).hexdigest(), raw


def rewrite_messages(
    msgs: list[dict],
    keep_idx: Optional[int],
    captions_by_url: dict[str, str],
) -> list[dict]:
    """Deep-copy `msgs`; strip `image_url` parts at every turn except `keep_idx`,
    and append `[Past image #N: <caption>]` text where they were stripped.

    Per-message numbering: each message's stripped images get #1, #2, ...
    starting from 1 again -- so the model can correlate captions to positions
    inside the same turn.
    """
    out = copy.deepcopy(msgs)
    for i, msg in enumerate(out):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if i == keep_idx:
            continue
        new_parts: list[dict] = []
        stripped_captions: list[str] = []
        img_n = 0
        for part in content:
            if part.get("type") == "image_url":
                img_n += 1
                url = part.get("image_url", {}).get("url", "")
                cap = captions_by_url.get(url) or "(no caption)"
                stripped_captions.append(f"[Past image #{img_n}: {cap}]")
            else:
                new_parts.append(part)
        if stripped_captions:
            extra_text = "\n".join(stripped_captions)
            if new_parts and new_parts[-1].get("type") == "text":
                new_parts[-1]["text"] = (
                    new_parts[-1]["text"] + "\n" + extra_text
                )
            else:
                new_parts.append({"type": "text", "text": extra_text})
        msg["content"] = new_parts
    return out
