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
