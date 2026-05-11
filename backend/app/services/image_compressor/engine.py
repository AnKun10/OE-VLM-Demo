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
from app.services.image_compressor.types import CompressionResult, Scanned, StatusEvent

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

    def _estimate_image_tokens(self, raw: bytes) -> int:
        """Cheap heuristic: ~800 tokens minimum, +1 token per 800 bytes."""
        return max(800, len(raw) // 800)

    def _build_thinking_log(
        self, *,
        n_images: int, n_misses: int, decision_label: str,
        captions_used: list[tuple[str, str]],
        user_text: Optional[str],
        route_reason: Optional[str],
        tokens_saved: int,
    ) -> str:
        """Build the <details>-wrapped reasoning markdown shown above the
        assistant reply."""
        lines = [
            "<details>",
            f"<summary>🧠 Image compressor reasoning ({n_images} ảnh, "
            f"{n_misses} caption mới, {decision_label})</summary>",
            "",
            "**Step 1 — Image scan**",
            f"- Tổng {n_images} ảnh; cache miss: {n_misses}, "
            f"hit: {n_images - n_misses}",
            "",
        ]
        if captions_used:
            lines.append("**Step 2 — Captions in use**")
            for h_short, cap in captions_used:
                lines.append(f"- `{h_short}` → \"{cap}\"")
            lines.append("")
        if user_text is not None:
            lines.append("**Step 3 — Router**")
            lines.append(f"- User: \"{user_text[:200]}\"")
            lines.append(f"- {decision_label}")
            if route_reason:
                lines.append(f"- Reason: *{route_reason}*")
            lines.append("")
        lines.append("**Step 4 — Rewrite**")
        if tokens_saved > 0:
            lines.append(f"- Token estimate saved: ~{tokens_saved}")
        else:
            lines.append("- Images preserved; no tokens saved")
        lines.append("</details>")
        lines.append("")
        return "\n".join(lines)

    async def compress(self, messages: list[dict]):
        """Async generator. Yields zero or more StatusEvents while working,
        then exactly one terminal CompressionResult. Self-catches every
        internal exception → on failure, yields passthrough(messages, '')."""
        try:
            async for ev in self._compress_impl(messages):
                yield ev
        except Exception as e:
            log.exception("compressor crash; falling through to passthrough: %s", e)
            yield CompressionResult(messages=messages, thinking_md="")

    async def _compress_impl(self, messages: list[dict]):
        from app.services.image_compressor.messages import (
            find_latest_image_turn, has_images, hash_image_url,
            iter_image_parts, rewrite_messages, text_of,
        )

        url_list = list(iter_image_parts(messages))
        if not url_list:
            yield CompressionResult(messages=messages, thinking_md="")
            return

        # Hash each image (fail-soft).
        scanned: list[Scanned] = []
        for msg_idx, c_idx, url in url_list:
            try:
                h, raw = await hash_image_url(url, self.webui_internal_base)
                scanned.append((msg_idx, c_idx, url, h, raw))
            except Exception as e:
                log.warning("hash skipped url=%s err=%s", url[:60], e)

        if not scanned:
            yield CompressionResult(messages=messages, thinking_md="")
            return

        # Decide keep_idx BEFORE captioning so we only caption images we need.
        last_msg = messages[-1] if messages else None
        if last_msg and has_images(last_msg):
            keep_idx: Optional[int] = len(messages) - 1
            decision_label = "kept new upload"
            user_text_for_log: Optional[str] = None
            route_reason: Optional[str] = None
            # Only caption images from non-kept turns.
            scanned_to_caption = [s for s in scanned if s[0] != keep_idx]
        else:
            scanned_to_caption = scanned
            keep_idx = None  # determined after routing below
            decision_label = ""
            user_text_for_log = None
            route_reason = None

        # Cache lookup → count misses for the status banner + thinking log.
        existing = await self.cache.get_many([h for *_, h, _ in scanned_to_caption])
        n_misses = sum(1 for s in scanned_to_caption if s[3] not in existing)
        if n_misses > 0:
            yield StatusEvent(
                message=f"🖼️ Captioning {n_misses} new image(s)..."
            )

        captions_by_url = await self.ensure_captions(scanned_to_caption)

        # For the router path, finish deciding keep_idx now.
        if last_msg and not has_images(last_msg):
            latest_idx = find_latest_image_turn(messages)
            if latest_idx is None:
                # No usable image turn → pure-text history; nothing to compress.
                yield CompressionResult(messages=messages, thinking_md="")
                return
            yield StatusEvent(message="🧭 Routing: do we need pixels?")
            captions_for_router = [
                captions_by_url.get(u, "(no caption)")
                for (mi, _, u, _, _) in scanned if mi == latest_idx
            ]
            user_text_for_log = text_of(last_msg) if last_msg else ""
            decision, route_reason = await self.route(
                user_text_for_log, captions_for_router,
            )
            if decision:
                keep_idx = latest_idx
                decision_label = "🎯 Router: keep images"
            else:
                keep_idx = None
                decision_label = "🎯 Router: drop images"
            yield StatusEvent(message=decision_label)

        # Token-saving estimate.
        if keep_idx is None:
            tokens_saved = sum(
                self._estimate_image_tokens(raw) for *_, raw in scanned
            )
        else:
            tokens_saved = sum(
                self._estimate_image_tokens(raw)
                for (mi, _, _, _, raw) in scanned if mi != keep_idx
            )

        new_messages = rewrite_messages(messages, keep_idx, captions_by_url)

        captions_used = [
            (h[:8], captions_by_url.get(url, "(no caption)"))
            for (_, _, url, h, _) in scanned
        ]
        thinking_md = self._build_thinking_log(
            n_images=len(scanned), n_misses=n_misses,
            decision_label=decision_label, captions_used=captions_used,
            user_text=user_text_for_log, route_reason=route_reason,
            tokens_saved=tokens_saved,
        )

        yield StatusEvent(message="✅ Compressor done", done=True)
        yield CompressionResult(messages=new_messages, thinking_md=thinking_md)
