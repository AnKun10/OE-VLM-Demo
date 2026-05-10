"""ImageCompressorEngine: orchestrates caption + router calls and history
rewriting. Built across Tasks 5-6 of the Phase 5 plan.

Task 5 (this commit): __init__, caption_one, route, ensure_captions.
Task 6 lands `compress()` — the async generator that drives chat_stream.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from app.services.image_compressor.cache import CaptionCache
from app.services.image_compressor.prompts import (
    CAPTION_SYSTEM_PROMPT,
    CAPTION_USER_TEXT,
    ROUTER_SYSTEM_PROMPT,
    ROUTER_USER_TEMPLATE,
)
from app.services.image_compressor.types import Scanned

log = logging.getLogger("image_compressor")


class ImageCompressorEngine:
    def __init__(
        self,
        cache: CaptionCache,
        vlm_manager,  # type: ignore[no-untyped-def]  # circular-import dodge
        *,
        caption_model_id: str,
        router_model_id: str,
        webui_internal_base: str = "http://127.0.0.1:8000",
        caption_max_tokens: int = 80,
        router_max_tokens: int = 60,
        caption_timeout_s: int = 30,
        router_timeout_s: int = 15,
        router_failopen_keep: bool = True,
    ) -> None:
        self.cache = cache
        self.manager = vlm_manager
        self.caption_model_id = caption_model_id
        self.router_model_id = router_model_id
        self.webui_internal_base = webui_internal_base
        self.caption_max_tokens = caption_max_tokens
        self.router_max_tokens = router_max_tokens
        self.caption_timeout_s = caption_timeout_s
        self.router_timeout_s = router_timeout_s
        self.router_failopen_keep = router_failopen_keep

    async def caption_one(self, data_url: str) -> str:
        """Caption a single image. Returns trimmed caption text."""
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": CAPTION_USER_TEXT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ]
        text = await self.manager.generate_raw(
            self.caption_model_id,
            messages=messages,
            max_tokens=self.caption_max_tokens, temperature=0.2,
        )
        return text.strip()

    async def route(
        self, user_text: str, captions: list[str],
    ) -> tuple[bool, str]:
        """Decide whether the LLM needs pixels for this user turn.

        Returns (need_images, reason). On any error (HTTP, JSON parse,
        missing key) returns (router_failopen_keep, "router failure: ...").
        """
        captions_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(captions))
        user_content = ROUTER_USER_TEMPLATE.format(
            captions_block=captions_block, user_text=user_text,
        )
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            raw = await self.manager.generate_raw(
                self.router_model_id, messages,
                max_tokens=self.router_max_tokens, temperature=0.0,
            )
            parsed = json.loads(raw)
            decision = bool(parsed.get("need_images"))
            reason = str(parsed.get("reason", ""))[:200]
            return decision, reason
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            log.warning("router parse failed: %s", e)
            return self.router_failopen_keep, f"router failure: {type(e).__name__}"
        except Exception as e:  # network / runtime
            log.warning("router call failed: %s", e)
            return self.router_failopen_keep, f"router failure: {type(e).__name__}"

    async def ensure_captions(
        self, scanned: list[Scanned], *, user_id: Optional[str] = None,
    ) -> dict[str, str]:
        """Return {url: caption} for every image in `scanned`. Cache hits
        served first; misses captioned in parallel and persisted. A single
        caption-call failure omits that url from the result and does NOT
        bring down the others.
        """
        if not scanned:
            return {}

        hashes = [h for _, _, _, h, _ in scanned]
        hits = await self.cache.get_many(hashes)

        out: dict[str, str] = {}
        misses: list[tuple[str, str, bytes]] = []
        for _, _, url, h, raw in scanned:
            if h in hits:
                out[url] = hits[h]
            else:
                misses.append((url, h, raw))

        if not misses:
            return out

        async def _one(url: str) -> tuple[str, Optional[str]]:
            try:
                cap = await self.caption_one(url)
                return url, cap or None
            except Exception as e:
                log.warning("caption failed for url=%s err=%s", url[:60], e)
                return url, None

        results = await asyncio.gather(*(_one(url) for url, _, _ in misses))

        new_rows: list[tuple[str, str, str, Optional[int], Optional[str]]] = []
        for (url, h, raw), (_, cap) in zip(misses, results):
            if cap:
                out[url] = cap
                new_rows.append((h, cap, self.caption_model_id, len(raw), user_id))

        if new_rows:
            await self.cache.put_many(new_rows)

        return out
