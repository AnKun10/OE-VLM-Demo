# Phase 4 — Persistence + Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add localStorage persistence for conversations + a handful of UX polish items (image-fallback placeholder, smart auto-scroll, hydrate-time `streaming → stopped` coercion, image-only title fallback, conversation-switch abort decoration). After this phase, a user can reload `/playground` and find their full multi-conversation history intact, including image previews that gracefully degrade if server-side files have been removed.

**Architecture:** Introduce a pure `lib/storage.ts` (versioned read/write) and a `useConversations` hook that wires `useReducer(conversationsReducer)` to storage with a debounced write effect. The reducer gains one new action (`HYDRATE`) which atomically swaps state to a payload while coercing any in-flight `streaming` messages to `stopped` (handles A4.3 reload-during-stream). The hook also performs an initial hydrate via the `useReducer` lazy initializer. `PlaygroundPage` is refactored to consume this hook instead of dispatching against an in-memory reducer. `MessageList` learns smart auto-scroll (track at-bottom position, only autoscroll when at-bottom). `MessageBubble`'s `<img>` tags swap to a `Ảnh đã hết hạn` placeholder on `onError`. Conversation switch and `Stop` consistently dispatch `MARK_STOPPED` on any in-flight assistant placeholder.

**Tech Stack:** TypeScript + React 18, Vitest + jsdom, Playwright (existing). No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-05-09-playground-qwen-vl-parity-design.md` (Phase 4 section starts at line 920; section C.9 details the persistence flow; section C.10 lists polish items).

**Phase 3 plan (sibling):** `docs/superpowers/plans/2026-05-10-playground-qwen-vl-parity-phase-3.md`. Phase 3 added Stop/Regenerate/Edit semantics and the `MARK_STOPPED` action this phase reuses.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| **Task 1 — Storage module** |||
| Create | `frontend/src/playground/lib/storage.ts` | `STORAGE_KEY`, `readState()`, `writeState()`; versioned, defensive |
| Create | `frontend/src/playground/lib/storage.test.ts` | T4.1 round-trip; A4.1 invalid JSON; A4.2 wrong schemaVersion |
| **Task 2 — `HYDRATE` action + coercion** |||
| Modify | `frontend/src/playground/lib/messageReducer.ts` | Add `HYDRATE` action that swaps state and coerces `status: "streaming"` → `"stopped"` |
| Modify | `frontend/src/playground/lib/messageReducer.test.ts` | T4.2 HYDRATE swap; A4.3 streaming coercion; T4.3 RENAME_TITLE isolation; T4.4 DELETE fallback (already partial — extend) |
| **Task 3 — `useConversations` hook** |||
| Create | `frontend/src/playground/hooks/useConversations.ts` | Reducer + lazy hydrate from storage + debounced write effect + quota toast |
| **Task 4 — `PlaygroundPage` integration** |||
| Modify | `frontend/src/pages/PlaygroundPage.tsx` | Replace inline `useReducer` with `useConversations`; ensure first-render still creates default conversation iff storage was empty; Stop on conversation-switch dispatches `MARK_STOPPED` for any streaming placeholder; image-only first-message title fallback uses ISO-ish date |
| **Task 5 — `<img onError>` fallback** |||
| Modify | `frontend/src/playground/components/MessageBubble.tsx` | Add small `AttachmentImg` helper that flips to a `Ảnh đã hết hạn` placeholder on `onError` |
| **Task 6 — Smart auto-scroll** |||
| Modify | `frontend/src/playground/components/MessageList.tsx` | Track at-bottom; only smooth-scroll when user was at bottom |
| **Task 7 — Playwright E2E** |||
| Modify | `frontend/tests/e2e/fixtures/sseFixture.ts` | Add `mockChatStreamShort(page, fullText)` (single-pass response) used by E4.x |
| Create | `frontend/tests/e2e/playground-persistence.spec.ts` | E4.1 reload sees prior messages; E4.2 two convs in sidebar; E4.3 delete removes from sidebar AND localStorage |
| **Task 8 — Manual smoke + final pass** |||
| — | — | Vitest run (all green); Playwright run; manual checklist (focus: A4.5 broken image, A4.7 multi-tab, A4.8 dropdown reflects active conv, A4.10 scroll preservation, A4.13 image-only title) |

---

## Tasks

### Task 1: Storage module + tests

**Files:**
- Create: `frontend/src/playground/lib/storage.ts`
- Create: `frontend/src/playground/lib/storage.test.ts`

- [ ] **Step 1: Write `storage.ts`**

```ts
import type { ConversationsState } from "../types";

const STORAGE_KEY = "playground.conversations.v1";

export function readState(): ConversationsState | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    if (parsed.schemaVersion !== 1) return null;
    if (!parsed.conversations || typeof parsed.conversations !== "object") return null;
    return parsed as ConversationsState;
  } catch (e) {
    console.warn("[storage] parse failed; starting fresh", e);
    return null;
  }
}

export class StorageQuotaError extends Error {
  constructor() { super("Storage quota exceeded"); }
}

export function writeState(state: ConversationsState): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    if (e instanceof DOMException && (e.name === "QuotaExceededError" || e.code === 22)) {
      throw new StorageQuotaError();
    }
    throw e;
  }
}

export const __TEST__ = { STORAGE_KEY };
```

- [ ] **Step 2: Write `storage.test.ts`**

Cover T4.1, A4.1, A4.2 plus a "no localStorage" smoke. Use `beforeEach(() => localStorage.clear())`.

- [ ] **Step 3: Run** `npm --prefix frontend run test:run -- storage` and verify green.

### Task 2: `HYDRATE` action + coercion + reducer tests

**Files:**
- Modify: `frontend/src/playground/lib/messageReducer.ts`
- Modify: `frontend/src/playground/lib/messageReducer.test.ts`

- [ ] **Step 1: Extend `Action` union in `messageReducer.ts`**

Add to the union:
```ts
| { type: "HYDRATE"; state: ConversationsState }
```

- [ ] **Step 2: Add reducer case** at the bottom of the switch (before final brace):

```ts
case "HYDRATE": {
  const incoming = action.state;
  const conversations: Record<string, Conversation> = {};
  for (const [id, conv] of Object.entries(incoming.conversations)) {
    conversations[id] = {
      ...conv,
      messages: conv.messages.map((m) =>
        m.status === "streaming" ? { ...m, status: "stopped" as const } : m,
      ),
    };
  }
  return { ...incoming, conversations };
}
```

- [ ] **Step 3: Append failing tests** to the END of `messageReducer.test.ts`:

```ts
  it("T4.2 — HYDRATE replaces state with payload", () => {
    const initial = withConv(initialState(), "c1", "m");
    const payload: ConversationsState = {
      schemaVersion: 1,
      activeId: "c-other",
      conversations: {
        "c-other": {
          id: "c-other", title: "Old", modelId: "m",
          messages: [], createdAt: 1, updatedAt: 1,
        },
      },
    };
    const out = conversationsReducer(initial, { type: "HYDRATE", state: payload });
    expect(out.activeId).toBe("c-other");
    expect(out.conversations["c-other"]).toBeDefined();
    expect(out.conversations.c1).toBeUndefined();
  });

  it("A4.3 — HYDRATE coerces in-flight streaming → stopped", () => {
    const payload: ConversationsState = {
      schemaVersion: 1,
      activeId: "c1",
      conversations: {
        c1: {
          id: "c1", title: "T", modelId: "m",
          messages: [
            { id: "u", role: "user", text: "hi", status: "done", createdAt: 1 },
            { id: "a", role: "assistant", text: "partial", status: "streaming", createdAt: 2 },
          ],
          createdAt: 1, updatedAt: 2,
        },
      },
    };
    const out = conversationsReducer(initialState(), { type: "HYDRATE", state: payload });
    expect(out.conversations.c1.messages.at(-1)!.status).toBe("stopped");
    expect(out.conversations.c1.messages.at(-1)!.text).toBe("partial");
  });

  it("T4.3 — RENAME_TITLE updates only target conversation", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
    s = conversationsReducer(s, { type: "RENAME_TITLE", conversationId: "c1", title: "NEW" });
    expect(s.conversations.c1.title).toBe("NEW");
    expect(s.conversations.c2.title).toBe("Cuộc hội thoại mới");
  });

  it("T4.4 — DELETE_CONVERSATION fallback: most-recent or null", () => {
    let s = withConv(initialState(), "c1", "m");      // updatedAt 1000
    s = conversationsReducer(s, {
      type: "NEW_CONVERSATION",
      conversationId: "c2", welcomeMessageId: "w2", modelId: "m", now: 5000,
    });
    expect(s.activeId).toBe("c2");
    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c2" });
    expect(s.activeId).toBe("c1");                     // fell back to most-recent remaining
    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c1" });
    expect(s.activeId).toBeNull();
  });
```

Add `import type { ConversationsState } from "../types";` at the top if not present.

- [ ] **Step 4: Run** `npm --prefix frontend run test:run -- messageReducer` and verify green.

### Task 3: `useConversations` hook

**Files:**
- Create: `frontend/src/playground/hooks/useConversations.ts`

- [ ] **Step 1: Write the hook**

```ts
import { useEffect, useReducer, useRef } from "react";
import {
  conversationsReducer,
  initialState,
  type Action,
} from "../lib/messageReducer";
import { readState, writeState, StorageQuotaError } from "../lib/storage";
import type { ConversationsState } from "../types";
import { useToast } from "./useToast";

const WRITE_DEBOUNCE_MS = 250;

function hydratedInit(): ConversationsState {
  const stored = readState();
  if (!stored) return initialState();
  // Reuse the reducer's coercion path so streaming → stopped is applied.
  return conversationsReducer(initialState(), { type: "HYDRATE", state: stored });
}

export function useConversations(): {
  state: ConversationsState;
  dispatch: React.Dispatch<Action>;
  hydrated: boolean;
} {
  const [state, dispatch] = useReducer(conversationsReducer, undefined, hydratedInit);
  const toast = useToast();
  const timerRef = useRef<number | null>(null);
  const quotaWarnedRef = useRef(false);

  useEffect(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      try {
        writeState(state);
      } catch (e) {
        if (e instanceof StorageQuotaError) {
          if (!quotaWarnedRef.current) {
            quotaWarnedRef.current = true;
            toast.push("Bộ nhớ đầy. Hãy xoá cuộc trò chuyện cũ.", "error");
          }
        } else {
          console.error("[useConversations] write failed", e);
        }
      }
    }, WRITE_DEBOUNCE_MS);
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [state, toast]);

  return { state, dispatch, hydrated: true };
}
```

> Note: We don't unit-test this hook directly (would need RTL); A4.4 (quota toast) and A4.14 (1 write per edit) are covered by manual smoke + the underlying reducer being a single dispatch per action.

### Task 4: `PlaygroundPage` integration

**Files:**
- Modify: `frontend/src/pages/PlaygroundPage.tsx`

- [ ] **Step 1:** Replace the inline `useReducer(...)` lazy initializer block (the `const [state, dispatch] = useReducer(...)` near the top of `PlaygroundInner`) with:

```ts
const { state, dispatch } = useConversations();
```

- [ ] **Step 2:** After the destructure, add a one-shot effect that creates the default conversation **only if storage was empty** (i.e., `Object.keys(state.conversations).length === 0`):

```ts
useEffect(() => {
  if (Object.keys(state.conversations).length === 0) {
    dispatch({
      type: "NEW_CONVERSATION",
      conversationId: uid(),
      welcomeMessageId: "w" + uid(),
      modelId: "",
      now: Date.now(),
    });
  }
  // Run once after hydrate; deps intentionally empty.
  // eslint-disable-next-line react-hooks/exhaustive-deps
}, []);
```

Replace the existing access of `activeId!` / `active!` with safe variants — when storage is empty, the first render has `activeId === null`. Use:

```ts
const activeId = state.activeId;
const active = activeId ? state.conversations[activeId] : null;
const messages = active?.messages ?? [];
```

Then short-circuit the JSX render with a "Đang tải…" placeholder when `!active`.

- [ ] **Step 3:** In `handleSend` (and any other place that auto-titles), update the image-only fallback to a date string per A4.13:

```ts
if (messages.filter((m) => m.role === "user").length === 0) {
  const titleSrc = trimmed
    ? trimmed.slice(0, 40) + (trimmed.length > 40 ? "…" : "")
    : `Hội thoại ${new Date().toLocaleString("vi-VN", {
        dateStyle: "short", timeStyle: "short",
      })}`;
  dispatch({
    type: "RENAME_TITLE",
    conversationId: activeId!,
    title: titleSrc,
  });
}
```

- [ ] **Step 4:** In `selectConversation`, dispatch `MARK_STOPPED` for any in-flight assistant placeholder of the *previous* active conversation **before** switching (A4.9):

```ts
function selectConversation(id: string) {
  abort();
  if (active) {
    const streaming = active.messages.find((m) => m.status === "streaming");
    if (streaming) {
      dispatch({
        type: "MARK_STOPPED",
        conversationId: active.id,
        messageId: streaming.id,
      });
    }
  }
  dispatch({ type: "SELECT_CONVERSATION", id });
  setText("");
  setAttachments([]);
  setEditingId(null);
}
```

- [ ] **Step 5:** Import `useConversations` at the top.

- [ ] **Step 6:** Run `npm --prefix frontend run build` to confirm the type-checker is happy.

### Task 5: Image fallback in `MessageBubble`

**Files:**
- Modify: `frontend/src/playground/components/MessageBubble.tsx`

- [ ] **Step 1:** Add a small inline component near the top:

```tsx
function AttachmentImg({ url, alt }: { url: string; alt: string }) {
  const [failed, setFailed] = useState(false);
  if (failed) {
    return (
      <div
        className="w-full h-full flex items-center justify-center text-[11px] text-center px-2"
        style={{ background: "#f3f4f6", color: "#6b7280" }}
      >
        Ảnh đã hết hạn
      </div>
    );
  }
  return (
    <img
      src={url}
      alt={alt}
      className="w-full h-full object-cover"
      onError={() => setFailed(true)}
    />
  );
}
```

(import `useState` from React)

- [ ] **Step 2:** Replace both `<img src={a.url} alt={a.originalName} className="w-full h-full object-cover" />` instances inside `UserBubble` with `<AttachmentImg url={a.url} alt={a.originalName} />`.

### Task 6: Smart auto-scroll in `MessageList`

**Files:**
- Modify: `frontend/src/playground/components/MessageList.tsx`

- [ ] **Step 1:** Replace the `useEffect` that scrolls to bottom with a smart-scroll effect:

```tsx
const containerRef = useRef<HTMLDivElement>(null);
const wasAtBottomRef = useRef(true);

const onScroll = (e: React.UIEvent<HTMLDivElement>) => {
  const el = e.currentTarget;
  wasAtBottomRef.current =
    el.scrollHeight - el.scrollTop - el.clientHeight < 32;
};

useEffect(() => {
  if (wasAtBottomRef.current) {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }
}, [lastMsgKey]);
```

The container needs to be a fixed-height scrollable div for `onScroll` to fire — wrap the existing render in a `<div ref={containerRef} onScroll={onScroll} className="h-full overflow-y-auto">…</div>`. **Caveat:** the parent in `PlaygroundPage` is already an `overflow-y-auto` wrapper. Move the scroll listener up to **that** parent in `PlaygroundPage` instead, OR have `MessageList` accept a `scrollContainerRef` prop. Simpler: keep MessageList unchanged in DOM structure; have it accept the *outer* container ref via prop and attach the listener to it.

To avoid restructuring, the practical move: install the scroll listener inside `MessageList` via `useEffect` against `containerRef.current?.parentElement` (since `PlaygroundPage` wraps `MessageList` in a `div.overflow-y-auto`):

```tsx
useEffect(() => {
  const scrollEl = containerRef.current?.parentElement;
  if (!scrollEl) return;
  const onScroll = () => {
    wasAtBottomRef.current =
      scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight < 32;
  };
  scrollEl.addEventListener("scroll", onScroll);
  return () => scrollEl.removeEventListener("scroll", onScroll);
}, []);
```

And keep the scroll-into-view effect gated by `wasAtBottomRef.current`.

### Task 7: Playwright E2E for persistence

**Files:**
- Modify: `frontend/tests/e2e/fixtures/sseFixture.ts`
- Create: `frontend/tests/e2e/playground-persistence.spec.ts`

- [ ] **Step 1:** Add `mockChatStreamShort(page, fullText)` to the fixture — emits a single `{delta: fullText}` then `{done: true}`. Inspect existing fixture for shape; mirror it.

- [ ] **Step 2:** Write the spec covering:

- **E4.1** — Send 1 user msg, wait for assistant reply, reload, expect both bubbles still visible.
- **E4.2** — Click "Cuộc trò chuyện mới" twice (after sending one msg in each); expect 2 entries in sidebar.
- **E4.3** — Delete one conversation; reload; expect it not in sidebar AND localStorage entry no longer contains its id.

Use `page.evaluate(() => localStorage.getItem("playground.conversations.v1"))` to inspect storage when needed.

- [ ] **Step 3:** Run `npm --prefix frontend run test:e2e -- playground-persistence` and confirm green.

### Task 8: Manual smoke + final pass

- [ ] Run the **full Vitest suite**: `npm --prefix frontend run test:run`.
- [ ] Run the **full Playwright suite**: `npm --prefix frontend run test:e2e`.
- [ ] Run `npm --prefix frontend run build` to confirm TypeScript is clean.
- [ ] **Manual browser smoke** (focus list):
  - A4.5 — manually delete an `images/<id>.png` file on disk → reload page → broken image swaps to "Ảnh đã hết hạn".
  - A4.7 — open `/playground` in two tabs; send in tab A; reload tab B; tab B sees new message.
  - A4.8 — switch between conversations with different `modelId`; dropdown reflects active conversation's model.
  - A4.10 — scroll up mid-stream; tokens still append; view stays put; scroll back down → auto-scroll resumes.
  - A4.13 — send image-only first message; sidebar title falls back to `"Hội thoại …"` with date.

---

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Smart auto-scroll listener attached to parent fails when PlaygroundPage layout changes. | Defensive `if (!scrollEl) return;` guard; verified manually post-Task 4. |
| R2 | `useConversations` hook depends on Toaster being mounted (uses `useToast`). | `PlaygroundPage` already wraps `<PlaygroundInner>` in `<Toaster>`; works as-is. |
| R3 | Debounce delays writes by 250ms; quick reload could lose tail. | Acceptable for demo; could add `pagehide` flush in v2. |
| R4 | localStorage in private/Safari ITP can throw on `setItem`. | `try/catch` + console.error fallback; non-quota errors do not crash app. |
| R5 | Hydration during SSR-style first render. | Vite SPA only, `typeof localStorage !== "undefined"` guard for safety. |
| R6 | `AttachmentImg` re-renders on every parent update lose its `failed` state. | `useState` is per-component-instance; React keys image by `a.id`, so identity stable. |

---

## Acceptance Criteria

A phase is complete when:

1. Vitest tests T4.1–T4.4 + A4.1–A4.3 pass.
2. Playwright E4.1–E4.3 pass.
3. `npm --prefix frontend run build` is green.
4. Manual smoke list above produces no console errors.
5. `simplify` self-review on diff before commit.
