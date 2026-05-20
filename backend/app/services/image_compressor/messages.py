"""Pure helpers for the image compressor: scanning, hashing, rewriting."""
from __future__ import annotations

import base64
import copy
import hashlib
import re
from typing import Iterator, Optional

import httpx


# Matches the <details>...</details> block emitted by
# ImageCompressorEngine._build_thinking_log. The match is anchored on the
# summary line so we don't strip <details> blocks that came from the LLM's
# own output (which would not start with the 🧠 brain emoji + label).
_THINKING_MD_RE = re.compile(
    r"<details>\s*<summary>🧠 Image compressor reasoning[\s\S]*?</details>\s*",
    flags=re.MULTILINE,
)


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
    latest_image_turn_idx: Optional[int] = None,
) -> list[dict]:
    """Deep-copy `msgs`; strip `image_url` parts at every turn except `keep_idx`,
    and append `[Past image #N: <caption>]` text where they were stripped.

    Per-message numbering: each message's stripped images get #1, #2, ...
    starting from 1 again -- so the model can correlate captions to positions
    inside the same turn.

    `latest_image_turn_idx`: when `keep_idx is None` (router dropped all
    pixels) AND this is set, OLDER image turns silently strip their images
    WITHOUT appending captions. The latest image turn still gets its
    captions appended. This keeps the model focused on the most recent
    image context instead of getting biased by older, longer captions.

    If an older image turn would become empty (had only images, no text)
    after a silent strip, a "[ảnh trước đó]" placeholder text is inserted
    so downstream multimodal handling doesn't choke on empty content.
    """
    out = copy.deepcopy(msgs)
    for i, msg in enumerate(out):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        if i == keep_idx:
            continue

        silent_strip = (
            keep_idx is None
            and latest_image_turn_idx is not None
            and i < latest_image_turn_idx
        )

        new_parts: list[dict] = []
        stripped_captions: list[str] = []
        img_n = 0
        for part in content:
            if part.get("type") == "image_url":
                img_n += 1
                if silent_strip:
                    # Drop the image without leaving a caption trace.
                    continue
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
        elif silent_strip and not new_parts and img_n > 0:
            # Older image-only message would become empty after silent
            # strip. Insert a marker so downstream code doesn't fail on
            # `content: []`.
            new_parts.append({"type": "text", "text": "[ảnh trước đó]"})
        msg["content"] = new_parts
    return out


def strip_thinking_md(text: str) -> str:
    """Remove `<details>...</details>` blocks emitted by the image
    compressor's `_build_thinking_log` from a string.

    These blocks are UI-only metadata streamed to the frontend; they should
    NOT be part of the LLM's context window for past assistant turns since
    they repeat caption text, biasing the model toward older images.

    Targets only the specific summary line ("🧠 Image compressor
    reasoning") so legitimate `<details>` blocks produced by the LLM are
    left untouched. Malformed blocks (no closing tag) are also left
    untouched -- the regex is non-greedy but requires a closing tag.
    """
    if not text:
        return text
    return _THINKING_MD_RE.sub("", text)


def strip_assistant_thinking(msgs: list[dict]) -> list[dict]:
    """Return a copy of `msgs` with every assistant message's string
    content run through `strip_thinking_md`. Non-string assistant content
    (list of parts) and other roles are left untouched.

    Use at the boundary between the image compressor and the final chat
    LLM call so past turns' thinking_md blocks don't reach the model.
    """
    out: list[dict] = []
    for msg in msgs:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                stripped = strip_thinking_md(content)
                if stripped != content:
                    out.append({**msg, "content": stripped})
                    continue
        out.append(msg)
    return out
