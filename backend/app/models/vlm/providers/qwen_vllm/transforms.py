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
