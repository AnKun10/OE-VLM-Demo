"""Message builder + image-cap policy for the streaming chat endpoint."""
from __future__ import annotations

import base64
from typing import Any, Iterable

from app.services.files import open_image_bytes

PLACEHOLDER = "[ảnh trong lượt trước đã được lược bỏ do giới hạn 4 ảnh]"


def build_openai_messages(msgs: Iterable) -> list[dict]:
    """Resolve a list of ChatMessageWithAttachments-like objects (with
    `.role`, `.text`, `.attachments[].id`) into OpenAI multimodal content.

    Raises FileNotFoundError if any referenced attachment id is missing.
    """
    out: list[dict] = []
    for m in msgs:
        attachments = list(m.attachments) if m.attachments else []
        if not attachments:
            out.append({"role": m.role, "content": m.text})
            continue
        parts: list[dict[str, Any]] = []
        for att in attachments:
            data = open_image_bytes(att.id)
            if data is None:
                raise FileNotFoundError(f"Attachment {att.id} not found")
            blob, mime = data
            data_uri = f"data:{mime};base64,{base64.b64encode(blob).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
        if m.text:
            parts.append({"type": "text", "text": m.text})
        out.append({"role": m.role, "content": parts})
    return out


def enforce_image_cap(messages: list[dict], max_images: int = 4) -> list[dict]:
    """Defensive backstop: cap the total `image_url` parts at `max_images` by
    replacing the OLDEST images with a dumb placeholder string.

    From Phase 5 onward this is a fallback for when `ImageCompressorEngine`
    crashes or is disabled — the compressor strips history images down to ≤1
    pixel-bearing turn before the request reaches here, so under normal
    operation this function is a no-op. Default `max_images=4` matches vLLM's
    `--limit-mm-per-prompt image=4` hard cap.
    """
    total = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            total += sum(1 for p in c if isinstance(p, dict) and p.get("type") == "image_url")

    if total <= max_images:
        return messages

    to_drop = total - max_images
    out: list[dict] = []
    for m in messages:
        c = m.get("content")
        if not isinstance(c, list) or to_drop == 0:
            out.append(m)
            continue
        had_original_text = any(
            isinstance(p, dict) and p.get("type") == "text" for p in c
        )
        new_parts: list[dict] = []
        for p in c:
            if (
                to_drop > 0
                and isinstance(p, dict)
                and p.get("type") == "image_url"
            ):
                # Replace this oldest image with a text placeholder.
                if new_parts and new_parts[-1].get("type") == "text":
                    # Coalesce with previous text segment.
                    new_parts[-1]["text"] = (
                        new_parts[-1]["text"].rstrip() + " " + PLACEHOLDER
                    ).strip()
                else:
                    new_parts.append({"type": "text", "text": PLACEHOLDER})
                to_drop -= 1
            else:
                new_parts.append(p)

        has_images = any(p.get("type") == "image_url" for p in new_parts)
        if not has_images and not had_original_text:
            # Lone-image message fully replaced: collapse to placeholder string
            # so vLLM doesn't see an empty or text-array-only content.
            joined = " ".join(p["text"] for p in new_parts).strip()
            out.append({"role": m["role"], "content": joined or PLACEHOLDER})
        else:
            out.append({"role": m["role"], "content": new_parts})

    return out
