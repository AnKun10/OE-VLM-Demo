"""Type aliases and dataclasses for the image-aware compressor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

# (msg_idx, content_idx, url, sha256_hex, raw_bytes)
Scanned = tuple[int, int, str, str, bytes]


@dataclass(frozen=True)
class StatusEvent:
    message: str
    done: bool = False


@dataclass(frozen=True)
class CompressionResult:
    messages: list[dict]
    thinking_md: str


CompressionEvent = Union[StatusEvent, CompressionResult]
