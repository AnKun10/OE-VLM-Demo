"""Pure message transforms for the Qwen vLLM provider.

All functions in this module are pure: they take OpenAI-format message
lists and return new lists. They do not perform I/O, mutate inputs, or
import the OpenAI SDK.
"""
from __future__ import annotations

import re
from copy import deepcopy

from .config import IMAGE_TOKEN_PATTERNS

_COMPILED_TOKEN_PATTERNS = tuple(re.compile(p) for p in IMAGE_TOKEN_PATTERNS)


def strip_image_tokens(messages: list[dict]) -> list[dict]:
    """Return a copy of messages with Qwen image placeholder tokens
    removed from user text content.
    """
    out = deepcopy(messages)
    for msg in out:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = _strip(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    part["text"] = _strip(part.get("text", ""))
    return out


def _strip(text: str) -> str:
    for pat in _COMPILED_TOKEN_PATTERNS:
        text = pat.sub("", text)
    return text


def inject_pixel_bounds(
    messages: list[dict],
    min_pixels: int,
    max_pixels: int,
) -> list[dict]:
    """Return a copy of messages with min_pixels/max_pixels attached to
    every image_url content part. No-op for text-only messages.

    Walks all roles (not just user) because image parts can in principle
    appear anywhere; in practice only user turns carry them in this
    codebase.
    """
    out = deepcopy(messages)
    for msg in out:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, dict):
                    image_url["min_pixels"] = min_pixels
                    image_url["max_pixels"] = max_pixels
    return out
