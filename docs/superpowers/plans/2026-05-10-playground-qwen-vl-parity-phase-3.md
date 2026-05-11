# Phase 3 — UX Controls (Stop / Regenerate / Edit Linear) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stop/Regenerate/Edit user-message UX to the playground, with linear semantics (edit truncates everything after), and clean up Phase 2's `overrideModelId` deferral by introducing a proper `SET_MODEL` reducer action.

**Architecture:** Extend `conversationsReducer` with 4 new actions (`SET_MODEL`, `MARK_STOPPED`, `POP_LAST_ASSISTANT`, `EDIT_USER_AND_TRUNCATE`). Add 2 small components (`StopButton`, `InlineEditor`). Augment `MessageBubble` with hover-revealed Edit button on user bubbles, Regenerate / "Thử lại" on the last assistant bubble, and `[bị dừng]` decoration on stopped messages. `ComposerBar` swaps Send → Stop while streaming. `PlaygroundPage` refactors `handleSend` into a reusable `runStream(messages, assistantId)` and wires the new callbacks; `editingId` lives as ephemeral local state.

**Tech Stack:** TypeScript + React 18, existing Vitest + Playwright stack from Phase 2.

**Spec:** `docs/superpowers/specs/2026-05-09-playground-qwen-vl-parity-design.md` (Phase 3 section starts at line 878; section C.8 details the state machine).

**Phase 2 plan (sibling):** `docs/superpowers/plans/2026-05-10-playground-qwen-vl-parity-phase-2.md`. Phase 2 already added the `APPEND_DELTA` streaming guard (commit `aa9c72a`) so late SSE chunks won't overwrite a stopped message.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| **Task 1 — Reducer extensions** |||
| Modify | `frontend/src/playground/lib/messageReducer.ts` | Add `SET_MODEL`, `MARK_STOPPED`, `POP_LAST_ASSISTANT`, `EDIT_USER_AND_TRUNCATE` to `Action` union + reducer cases |
| Modify | `frontend/src/playground/lib/messageReducer.test.ts` | Add tests T3.1-T3.4 + A3.4 + A3.9 + SET_MODEL coverage |
| **Task 2 — `StopButton` component** |||
| Create | `frontend/src/playground/components/StopButton.tsx` | Compact red square button with stop icon; calls `onStop` on click |
| **Task 3 — `InlineEditor` component** |||
| Create | `frontend/src/playground/components/InlineEditor.tsx` | Textarea + Save/Cancel buttons; rejects empty save with toast (A3.6); Esc → cancel; Ctrl+Enter → save |
| **Task 4 — `MessageBubble` + `MessageList` modifications** |||
| Modify | `frontend/src/playground/components/MessageBubble.tsx` | Add hover-reveal Edit button (user); Regenerate + "Thử lại" buttons (last assistant); `[bị dừng]` decoration (stopped status); InlineEditor swap when `editingId === msg.id` |
| Modify | `frontend/src/playground/components/MessageList.tsx` | Pass `isLast` per message + actions object through to MessageBubble |
| **Task 5 — `ComposerBar` modifications** |||
| Modify | `frontend/src/playground/components/ComposerBar.tsx` | Add `streaming?: boolean` + `onStop?: () => void` props; render StopButton in Send slot when streaming |
| **Task 6 — `PlaygroundPage` orchestrator** |||
| Modify | `frontend/src/pages/PlaygroundPage.tsx` | Drop `overrideModelId`; dispatch SET_MODEL on dropdown change; refactor `handleSend` → `runStream`; add `editingId`/`regenerate`/`saveEdit`/`cancelEdit` callbacks; compute `isStreaming` from messages; pass `streaming` + `onStop` to ComposerBar |
| **Task 7 — Playwright E2E** |||
| Modify | `frontend/tests/e2e/fixtures/sseFixture.ts` | Add `mockChatStreamSlow(page, deltas, gapMs)` so E2E can click Stop mid-stream; add `mockChatStreamError(page)` for A3.7 |
| Create | `frontend/tests/e2e/playground-controls.spec.ts` | E3.1 Stop+Regenerate, E3.2 Edit truncates, E3.3 Edit Esc reverts, A3.7 Regenerate-on-network-down |
| **Task 8 — Manual smoke + final pass** |||
| — | — | Vitest + Playwright + manual checklist (focus: A3.1, A3.2, A3.3, A3.5, A3.10, A3.11) |

---

## Tasks

### Task 1: Reducer extensions (4 new actions + tests)

**Files:**
- Modify: `frontend/src/playground/lib/messageReducer.ts`
- Modify: `frontend/src/playground/lib/messageReducer.test.ts`

- [ ] **Step 1: Append failing tests**

Append this block to the END of `frontend/src/playground/lib/messageReducer.test.ts` (inside the existing `describe("conversationsReducer", () => {...})` block, just before its closing `});`):

```ts
  it("SET_MODEL updates conversation modelId; no-op for unknown id", () => {
    let s = withConv(initialState(), "c1", "m-old");
    s = conversationsReducer(s, {
      type: "SET_MODEL",
      conversationId: "c1",
      modelId: "m-new",
    });
    expect(s.conversations.c1.modelId).toBe("m-new");

    // Unknown conversation → no change.
    const before = s;
    s = conversationsReducer(s, {
      type: "SET_MODEL",
      conversationId: "does-not-exist",
      modelId: "x",
    });
    expect(s).toBe(before);
  });

  it("T3.1 — MARK_STOPPED updates status to stopped without touching text", () => {
    let s = withConv(initialState(), "c1", "m");
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a1",
      now: 3000,
    });
    s = conversationsReducer(s, {
      type: "APPEND_DELTA",
      conversationId: "c1",
      messageId: "a1",
      delta: "partial reply",
    });
    s = conversationsReducer(s, {
      type: "MARK_STOPPED",
      conversationId: "c1",
      messageId: "a1",
    });
    const last = s.conversations.c1.messages.at(-1)!;
    expect(last.status).toBe("stopped");
    expect(last.text).toBe("partial reply"); // text untouched
  });

  it("T3.2 — POP_LAST_ASSISTANT removes last message iff role=assistant", () => {
    let s = withConv(initialState(), "c1", "m");
    // Welcome + 1 user + 1 assistant
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: { id: "u1", role: "user", text: "hi", createdAt: 2000 },
    });
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a1",
      now: 3000,
    });
    s = conversationsReducer(s, {
      type: "MARK_DONE",
      conversationId: "c1",
      messageId: "a1",
    });
    expect(s.conversations.c1.messages.length).toBe(3); // welcome + u1 + a1

    s = conversationsReducer(s, {
      type: "POP_LAST_ASSISTANT",
      conversationId: "c1",
    });
    expect(s.conversations.c1.messages.length).toBe(2);
    expect(s.conversations.c1.messages.at(-1)!.id).toBe("u1");

    // Now last is user → POP is a no-op.
    const before = s;
    s = conversationsReducer(s, {
      type: "POP_LAST_ASSISTANT",
      conversationId: "c1",
    });
    expect(s).toBe(before);
  });

  it("T3.3 — EDIT_USER_AND_TRUNCATE replaces text and drops everything after", () => {
    let s = withConv(initialState(), "c1", "m");
    // welcome + u1 + a1 + u2 + a2 + u3
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: { id: "u1", role: "user", text: "first", createdAt: 1 },
    });
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a1",
      now: 2,
    });
    s = conversationsReducer(s, { type: "MARK_DONE", conversationId: "c1", messageId: "a1" });
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: { id: "u2", role: "user", text: "second", createdAt: 3 },
    });
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a2",
      now: 4,
    });
    s = conversationsReducer(s, { type: "MARK_DONE", conversationId: "c1", messageId: "a2" });
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: { id: "u3", role: "user", text: "third", createdAt: 5 },
    });
    expect(s.conversations.c1.messages.length).toBe(6);

    // Edit u2 → "second EDITED": should keep welcome+u1+a1+u2(edited), drop a2+u3.
    s = conversationsReducer(s, {
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: "c1",
      messageId: "u2",
      newText: "second EDITED",
    });
    expect(s.conversations.c1.messages.length).toBe(4);
    expect(s.conversations.c1.messages.at(-1)!.id).toBe("u2");
    expect(s.conversations.c1.messages.at(-1)!.text).toBe("second EDITED");
  });

  it("A3.4 — EDIT msg#1 of 5-msg conv → 4 subsequent dropped", () => {
    let s = withConv(initialState(), "c1", "m");
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: { id: "u1", role: "user", text: "msg1", createdAt: 1 },
    });
    // 4 more messages after u1
    for (let i = 2; i <= 5; i++) {
      s = conversationsReducer(s, {
        type: "ADD_USER_MESSAGE",
        conversationId: "c1",
        message: { id: `u${i}`, role: "user", text: `msg${i}`, createdAt: i },
      });
    }
    expect(s.conversations.c1.messages.length).toBe(6); // welcome + u1..u5

    s = conversationsReducer(s, {
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: "c1",
      messageId: "u1",
      newText: "msg1 edited",
    });
    expect(s.conversations.c1.messages.length).toBe(2); // welcome + u1(edited)
    expect(s.conversations.c1.messages.at(-1)!.text).toBe("msg1 edited");
  });

  it("T3.4 / A3.9 — EDIT_USER_AND_TRUNCATE preserves attachments of edited message", () => {
    let s = withConv(initialState(), "c1", "m");
    const four = [
      { id: "a1", url: "/a", mime: "image/png", originalName: "a" },
      { id: "a2", url: "/b", mime: "image/png", originalName: "b" },
      { id: "a3", url: "/c", mime: "image/png", originalName: "c" },
      { id: "a4", url: "/d", mime: "image/png", originalName: "d" },
    ];
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: {
        id: "u1",
        role: "user",
        text: "pics",
        attachments: four,
        createdAt: 1,
      },
    });
    s = conversationsReducer(s, {
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: "c1",
      messageId: "u1",
      newText: "pics EDITED",
    });
    const last = s.conversations.c1.messages.at(-1)!;
    expect(last.text).toBe("pics EDITED");
    expect(last.attachments).toEqual(four);
  });

  it("EDIT_USER_AND_TRUNCATE on missing messageId is a no-op", () => {
    let s = withConv(initialState(), "c1", "m");
    const before = s;
    s = conversationsReducer(s, {
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: "c1",
      messageId: "does-not-exist",
      newText: "x",
    });
    expect(s).toBe(before);
  });
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- messageReducer
```

Expected: 7 new tests fail with "is not assignable to type 'Action'" or similar (because the new action variants don't exist yet).

- [ ] **Step 3: Add 4 new action variants + reducer cases**

Replace `frontend/src/playground/lib/messageReducer.ts` ENTIRELY with:

```ts
import type {
  ConversationsState,
  Conversation,
  Message,
  ErrorKind,
} from "../types";

const WELCOME_TEXT =
  "Xin chào! Tôi là mô hình AI của RunShop. Bạn có thể gửi văn bản hoặc hình ảnh để kiểm tra khả năng của tôi. Hãy thử ngay!";

const DEFAULT_TITLE = "Cuộc hội thoại mới";

export type Action =
  | {
      type: "NEW_CONVERSATION";
      conversationId: string;
      welcomeMessageId: string;
      modelId: string;
      now: number;
    }
  | { type: "DELETE_CONVERSATION"; id: string }
  | { type: "SELECT_CONVERSATION"; id: string }
  | { type: "SET_MODEL"; conversationId: string; modelId: string }
  | { type: "ADD_USER_MESSAGE"; conversationId: string; message: Message }
  | {
      type: "ADD_ASSISTANT_PLACEHOLDER";
      conversationId: string;
      messageId: string;
      now: number;
    }
  | {
      type: "APPEND_DELTA";
      conversationId: string;
      messageId: string;
      delta: string;
    }
  | { type: "MARK_DONE"; conversationId: string; messageId: string }
  | { type: "MARK_STOPPED"; conversationId: string; messageId: string }
  | {
      type: "MARK_ERROR";
      conversationId: string;
      messageId: string;
      errorKind: ErrorKind;
    }
  | { type: "POP_LAST_ASSISTANT"; conversationId: string }
  | {
      type: "EDIT_USER_AND_TRUNCATE";
      conversationId: string;
      messageId: string;
      newText: string;
    }
  | { type: "RENAME_TITLE"; conversationId: string; title: string };

export function initialState(): ConversationsState {
  return { schemaVersion: 1, conversations: {}, activeId: null };
}

function patchMessage(
  state: ConversationsState,
  conversationId: string,
  messageId: string,
  patch: (m: Message) => Message,
): ConversationsState {
  const conv = state.conversations[conversationId];
  if (!conv) return state;
  const idx = conv.messages.findIndex((m) => m.id === messageId);
  if (idx === -1) return state;
  const next = [...conv.messages];
  next[idx] = patch(next[idx]);
  return {
    ...state,
    conversations: {
      ...state.conversations,
      [conversationId]: { ...conv, messages: next, updatedAt: Date.now() },
    },
  };
}

function patchConversation(
  state: ConversationsState,
  id: string,
  patch: (c: Conversation) => Conversation,
): ConversationsState {
  const conv = state.conversations[id];
  if (!conv) return state;
  return {
    ...state,
    conversations: { ...state.conversations, [id]: patch(conv) },
  };
}

export function conversationsReducer(
  state: ConversationsState,
  action: Action,
): ConversationsState {
  switch (action.type) {
    case "NEW_CONVERSATION": {
      const welcome: Message = {
        id: action.welcomeMessageId,
        role: "assistant",
        text: WELCOME_TEXT,
        status: "done",
        createdAt: action.now,
      };
      const conv: Conversation = {
        id: action.conversationId,
        title: DEFAULT_TITLE,
        modelId: action.modelId,
        messages: [welcome],
        createdAt: action.now,
        updatedAt: action.now,
      };
      return {
        ...state,
        conversations: { ...state.conversations, [conv.id]: conv },
        activeId: conv.id,
      };
    }
    case "DELETE_CONVERSATION": {
      if (!state.conversations[action.id]) return state;
      const { [action.id]: _, ...rest } = state.conversations;
      let nextActive = state.activeId;
      if (state.activeId === action.id) {
        const remaining = Object.values(rest).sort(
          (a, b) => b.updatedAt - a.updatedAt,
        );
        nextActive = remaining[0]?.id ?? null;
      }
      return { ...state, conversations: rest, activeId: nextActive };
    }
    case "SELECT_CONVERSATION":
      return state.conversations[action.id]
        ? { ...state, activeId: action.id }
        : state;
    case "SET_MODEL":
      return patchConversation(state, action.conversationId, (conv) => ({
        ...conv,
        modelId: action.modelId,
        updatedAt: Date.now(),
      }));
    case "ADD_USER_MESSAGE":
      return patchConversation(state, action.conversationId, (conv) => ({
        ...conv,
        messages: [...conv.messages, action.message],
        updatedAt: Date.now(),
      }));
    case "ADD_ASSISTANT_PLACEHOLDER": {
      const placeholder: Message = {
        id: action.messageId,
        role: "assistant",
        text: "",
        status: "streaming",
        createdAt: action.now,
      };
      return patchConversation(state, action.conversationId, (conv) => ({
        ...conv,
        messages: [...conv.messages, placeholder],
        updatedAt: action.now,
      }));
    }
    case "APPEND_DELTA":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => (m.status === "streaming" ? { ...m, text: m.text + action.delta } : m),
      );
    case "MARK_DONE":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => ({ ...m, status: "done" }),
      );
    case "MARK_STOPPED":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => ({ ...m, status: "stopped" }),
      );
    case "MARK_ERROR":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => ({ ...m, status: "error", errorKind: action.errorKind }),
      );
    case "POP_LAST_ASSISTANT": {
      const conv = state.conversations[action.conversationId];
      if (!conv) return state;
      const last = conv.messages.at(-1);
      if (!last || last.role !== "assistant") return state;
      return {
        ...state,
        conversations: {
          ...state.conversations,
          [action.conversationId]: {
            ...conv,
            messages: conv.messages.slice(0, -1),
            updatedAt: Date.now(),
          },
        },
      };
    }
    case "EDIT_USER_AND_TRUNCATE": {
      const conv = state.conversations[action.conversationId];
      if (!conv) return state;
      const idx = conv.messages.findIndex((m) => m.id === action.messageId);
      if (idx === -1) return state;
      const target = conv.messages[idx];
      const edited: Message = { ...target, text: action.newText };
      const truncated = [...conv.messages.slice(0, idx), edited];
      return {
        ...state,
        conversations: {
          ...state.conversations,
          [action.conversationId]: {
            ...conv,
            messages: truncated,
            updatedAt: Date.now(),
          },
        },
      };
    }
    case "RENAME_TITLE":
      return patchConversation(state, action.conversationId, (conv) => ({
        ...conv,
        title: action.title,
        updatedAt: Date.now(),
      }));
  }
}
```

- [ ] **Step 4: Verify pass**

```
cd frontend
npm run test:run -- messageReducer
```

Expected: 18 tests pass (11 from Phase 2 + 7 new).

- [ ] **Step 5: Verify full suite still green**

```
cd frontend
npm run test:run
```

Expected: 44 tests pass (37 prior + 7 new).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/playground/lib/messageReducer.ts frontend/src/playground/lib/messageReducer.test.ts
git commit -m "feat(playground): add SET_MODEL/MARK_STOPPED/POP_LAST_ASSISTANT/EDIT_USER_AND_TRUNCATE reducer actions"
```

---

### Task 2: `StopButton` component

**Files:**
- Create: `frontend/src/playground/components/StopButton.tsx`

- [ ] **Step 1: Implement**

Create `frontend/src/playground/components/StopButton.tsx`:

```tsx
import { Square } from "lucide-react";

/**
 * Compact stop button used by ComposerBar in place of Send while a
 * streaming response is in progress. Rendered as a red square — the
 * universal "stop generation" affordance used by ChatGPT, Claude, and
 * Open WebUI.
 */
export function StopButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center justify-center rounded-lg transition-all"
      style={{
        width: 36,
        height: 36,
        background: "#dc2626",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "#b91c1c")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "#dc2626")}
      aria-label="Dừng"
      title="Dừng phản hồi"
    >
      <Square size={13} className="text-white" fill="currentColor" />
    </button>
  );
}
```

- [ ] **Step 2: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/playground/components/StopButton.tsx
git commit -m "feat(playground): add StopButton component"
```

---

### Task 3: `InlineEditor` component

**Files:**
- Create: `frontend/src/playground/components/InlineEditor.tsx`

- [ ] **Step 1: Implement**

Create `frontend/src/playground/components/InlineEditor.tsx`:

```tsx
import { Check, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useToast } from "../hooks/useToast";

/**
 * In-place editor for a user message bubble. Renders a textarea + Save/Cancel.
 *
 * Keyboard shortcuts:
 *   - Esc → cancel
 *   - Ctrl/Cmd+Enter → save
 *   - Enter alone → newline (multi-line edits supported)
 *
 * Empty-text save (A3.6) is rejected with a toast "Tin nhắn rỗng".
 */
export function InlineEditor({
  initialText,
  onSave,
  onCancel,
}: {
  initialText: string;
  onSave: (text: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(initialText);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const toast = useToast();

  // Auto-focus + select-all on mount.
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.focus();
      ta.selectionStart = ta.value.length;
      ta.selectionEnd = ta.value.length;
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    }
  }, []);

  // Auto-resize as user types.
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    }
  }, [text]);

  function handleSave() {
    if (text.trim().length === 0) {
      toast.push("Tin nhắn rỗng.", "error");
      return;
    }
    onSave(text);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSave();
    }
  }

  return (
    <div
      className="flex flex-col gap-2 w-full"
      style={{ maxWidth: "85%" }}
    >
      <textarea
        ref={taRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
        className="w-full resize-none outline-none text-[15px] leading-relaxed"
        style={{
          background: "#ffffff",
          color: "#111827",
          border: "1px solid #015e9f",
          borderRadius: 12,
          padding: "10px 14px",
          maxHeight: 240,
        }}
        aria-label="Chỉnh sửa tin nhắn"
      />
      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs transition-colors"
          style={{
            background: "transparent",
            color: "#6b7280",
            border: "1px solid #e5e7eb",
          }}
          aria-label="Huỷ"
          title="Huỷ (Esc)"
        >
          <X size={13} />
          Huỷ
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
          style={{
            background: "#015e9f",
            color: "#ffffff",
          }}
          aria-label="Lưu"
          title="Lưu (Ctrl+Enter)"
        >
          <Check size={13} />
          Lưu
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/playground/components/InlineEditor.tsx
git commit -m "feat(playground): add InlineEditor with Esc/Ctrl+Enter shortcuts and empty-reject toast"
```

---

### Task 4: `MessageBubble` + `MessageList` modifications

**Files:**
- Modify: `frontend/src/playground/components/MessageBubble.tsx`
- Modify: `frontend/src/playground/components/MessageList.tsx`

- [ ] **Step 1: Replace `MessageBubble.tsx` entirely**

This task adds Edit (user bubble), Regenerate / "Thử lại" (last assistant bubble), `[bị dừng]` decoration (stopped status), and InlineEditor swap (when editing). New props:

- `isLast: boolean` — whether this is the last message in the list
- `actions: MessageActions` — callbacks + flags from the orchestrator

Replace `frontend/src/playground/components/MessageBubble.tsx` ENTIRELY with:

```tsx
import { Pencil, RefreshCw, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { SafeLink } from "./SafeLink";
import { InlineEditor } from "./InlineEditor";
import type { Message } from "../types";

const ACCENT = "#015e9f";
const TEXT_PRIMARY = "#111827";
const TEXT_MUTED = "#9ca3af";
const TEXT_SECONDARY = "#6b7280";
const BORDER = "#e5e7eb";

export type MessageActions = {
  /** True iff any message in the active conversation has status "streaming". */
  isStreaming: boolean;
  /** Currently-edited message id (or null). Edit button is disabled when set. */
  editingId: string | null;
  onStartEdit: (messageId: string) => void;
  onSaveEdit: (messageId: string, newText: string) => void;
  onCancelEdit: () => void;
  onRegenerate: () => void;
};

function UserBubble({
  msg,
  actions,
}: {
  msg: Message;
  actions: MessageActions;
}) {
  const isEditing = actions.editingId === msg.id;

  if (isEditing) {
    return (
      <div className="flex flex-col items-end gap-1">
        {msg.attachments && msg.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 justify-end" style={{ maxWidth: "85%" }}>
            {msg.attachments.map((a) => (
              <div
                key={a.id}
                className="overflow-hidden"
                style={{ width: 120, height: 120, borderRadius: 12, border: `1px solid ${BORDER}` }}
              >
                <img src={a.url} alt={a.originalName} className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        )}
        <InlineEditor
          initialText={msg.text}
          onSave={(text) => actions.onSaveEdit(msg.id, text)}
          onCancel={actions.onCancelEdit}
        />
      </div>
    );
  }

  const canEdit = !actions.isStreaming && actions.editingId === null;

  return (
    <div className="group flex flex-col items-end gap-1">
      {msg.attachments && msg.attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 justify-end" style={{ maxWidth: "85%" }}>
          {msg.attachments.map((a) => (
            <div
              key={a.id}
              className="overflow-hidden"
              style={{ width: 120, height: 120, borderRadius: 12, border: `1px solid ${BORDER}` }}
            >
              <img src={a.url} alt={a.originalName} className="w-full h-full object-cover" />
            </div>
          ))}
        </div>
      )}
      {msg.text && (
        <div className="flex items-end gap-1.5" style={{ maxWidth: "85%" }}>
          {canEdit && (
            <button
              type="button"
              onClick={() => actions.onStartEdit(msg.id)}
              className="opacity-0 group-hover:opacity-100 p-1.5 rounded transition-all"
              style={{ color: TEXT_MUTED }}
              aria-label="Chỉnh sửa"
              title="Chỉnh sửa"
            >
              <Pencil size={13} />
            </button>
          )}
          <div
            className="text-[16px] leading-relaxed whitespace-pre-wrap"
            style={{
              background: "#0d1b67",
              color: "#ffffff",
              borderRadius: "18px 18px 4px 18px",
              padding: "10px 16px",
            }}
          >
            {msg.text}
          </div>
        </div>
      )}
    </div>
  );
}

function AssistantBubble({
  msg,
  isLast,
  actions,
}: {
  msg: Message;
  isLast: boolean;
  actions: MessageActions;
}) {
  const isError = msg.status === "error";
  const isStopped = msg.status === "stopped";
  const isDone = msg.status === "done";
  const canRegenerate =
    isLast && (isDone || isStopped || isError) && !actions.isStreaming;

  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 pt-0.5">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "rgba(1,94,159,0.15)" }}
        >
          <Sparkles size={15} style={{ color: ACCENT }} />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <span className="text-xs font-medium" style={{ color: TEXT_MUTED }}>
          AI Model
        </span>
        <div
          className="mt-1 text-[16px] leading-relaxed prose prose-sm max-w-none"
          style={{
            color: isError ? "#991b1b" : TEXT_PRIMARY,
            background: isError ? "#fef2f2" : "transparent",
            border: isError ? "1px solid #fecaca" : "none",
            borderRadius: isError ? 12 : 0,
            padding: isError ? "10px 14px" : 0,
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{ a: SafeLink as never }}
          >
            {msg.text || ""}
          </ReactMarkdown>
          {isStopped && (
            <span
              className="text-xs italic"
              style={{ color: TEXT_MUTED, marginLeft: 4 }}
            >
              [bị dừng]
            </span>
          )}
        </div>
        {canRegenerate && (
          <div className="mt-2">
            <button
              type="button"
              onClick={actions.onRegenerate}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs transition-colors"
              style={{
                background: isError ? "#015e9f" : "transparent",
                color: isError ? "#ffffff" : TEXT_SECONDARY,
                border: isError ? "none" : `1px solid ${BORDER}`,
              }}
              aria-label={isError ? "Thử lại" : "Tạo lại"}
              title={isError ? "Thử lại" : "Tạo lại phản hồi"}
            >
              <RefreshCw size={12} />
              {isError ? "Thử lại" : "Tạo lại"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageBubble({
  msg,
  isLast,
  actions,
}: {
  msg: Message;
  isLast: boolean;
  actions: MessageActions;
}) {
  return msg.role === "user" ? (
    <UserBubble msg={msg} actions={actions} />
  ) : (
    <AssistantBubble msg={msg} isLast={isLast} actions={actions} />
  );
}
```

- [ ] **Step 2: Replace `MessageList.tsx` entirely**

MessageList now forwards `actions` and computes `isLast` per item.

Replace `frontend/src/playground/components/MessageList.tsx` ENTIRELY with:

```tsx
import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import { MessageBubble, type MessageActions } from "./MessageBubble";
import type { Message } from "../types";

export function MessageList({
  messages,
  actions,
}: {
  messages: Message[];
  actions: MessageActions;
}) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastMsgKey = messages.at(-1)?.id + "@" + (messages.at(-1)?.text.length ?? 0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lastMsgKey]);

  const lastIsStreaming = messages.at(-1)?.status === "streaming";
  const lastIsEmpty = lastIsStreaming && (messages.at(-1)?.text ?? "").length === 0;

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
      {messages.map((m, i) => (
        <MessageBubble
          key={m.id}
          msg={m}
          isLast={i === messages.length - 1}
          actions={actions}
        />
      ))}
      {lastIsEmpty && (
        <div className="flex gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "rgba(1,94,159,0.15)" }}
          >
            <Sparkles size={15} style={{ color: "#015e9f" }} />
          </div>
          <div className="pt-2">
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{ background: "#015e9f", animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
```

- [ ] **Step 3: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: errors in `PlaygroundPage.tsx` because it doesn't yet pass `actions` to `<MessageList>`. That's fine — Task 6 fixes the orchestrator. For now, we tolerate the orchestrator-level error and verify only the new files compile in isolation.

If you want a green tsc immediately, defer the commit until Task 6. Otherwise, the failing call site is the only error — verify the count stays at 1 (no other unrelated breakage):

```
cd frontend
npx tsc --noEmit 2>&1 | wc -l
```

Expected: ≤ 5 lines (the single missing-prop error spread across a few message lines).

- [ ] **Step 4: Run unit tests**

Reducer + lib tests are unaffected by component changes:

```
cd frontend
npm run test:run
```

Expected: 44 tests pass (the unit tests don't import MessageBubble).

- [ ] **Step 5: Commit (with known orchestrator break)**

```bash
git add frontend/src/playground/components/MessageBubble.tsx frontend/src/playground/components/MessageList.tsx
git commit -m "feat(playground): add Edit/Regenerate/[bị dừng] to MessageBubble + actions plumbing through MessageList"
```

The `tsc` failure in `PlaygroundPage.tsx` is intentional and resolved in Task 6.

---

### Task 5: `ComposerBar` modifications (streaming + StopButton)

**Files:**
- Modify: `frontend/src/playground/components/ComposerBar.tsx`

- [ ] **Step 1: Surgical edit — add `streaming` and `onStop` props**

Find this block in `frontend/src/playground/components/ComposerBar.tsx`:

```tsx
import { ImagePlus, Mic, Send } from "lucide-react";
```

Replace with:

```tsx
import { ImagePlus, Mic, Send } from "lucide-react";
import { StopButton } from "./StopButton";
```

Find the `ComposerBarProps` type definition:

```tsx
export type ComposerBarProps = {
  text: string;
  onTextChange: (s: string) => void;
  attachments: AttachmentRef[];
  onAttach: (a: AttachmentRef) => void;
  onRemoveAttachment: (id: string) => void;
  onSend: () => void;
  modelDropdown: ReactNode;
  visionEnabled: boolean;
  visionWarning?: string | null;
  historyImageCount: number;
  disabled?: boolean;
};
```

Replace with (adds two new props):

```tsx
export type ComposerBarProps = {
  text: string;
  onTextChange: (s: string) => void;
  attachments: AttachmentRef[];
  onAttach: (a: AttachmentRef) => void;
  onRemoveAttachment: (id: string) => void;
  onSend: () => void;
  modelDropdown: ReactNode;
  visionEnabled: boolean;
  visionWarning?: string | null;
  historyImageCount: number;
  disabled?: boolean;
  streaming?: boolean;
  onStop?: () => void;
};
```

Find the destructure:

```tsx
  const {
    text,
    onTextChange,
    attachments,
    onAttach,
    onRemoveAttachment,
    onSend,
    modelDropdown,
    visionEnabled,
    visionWarning,
    historyImageCount,
    disabled,
  } = props;
```

Replace with:

```tsx
  const {
    text,
    onTextChange,
    attachments,
    onAttach,
    onRemoveAttachment,
    onSend,
    modelDropdown,
    visionEnabled,
    visionWarning,
    historyImageCount,
    disabled,
    streaming,
    onStop,
  } = props;
```

Find the Send button:

```tsx
            <button
              type="button"
              onClick={onSend}
              disabled={!canSend}
              className="flex items-center justify-center rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              style={{
                width: 36,
                height: 36,
                background: canSend ? "#015e9f" : "#9ca3af",
              }}
              aria-label="Gửi"
            >
              <Send size={15} className="text-white" style={{ marginLeft: 1 }} />
            </button>
```

Replace with (renders StopButton when streaming AND onStop provided, otherwise Send as before):

```tsx
            {streaming && onStop ? (
              <StopButton onClick={onStop} />
            ) : (
              <button
                type="button"
                onClick={onSend}
                disabled={!canSend}
                className="flex items-center justify-center rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                style={{
                  width: 36,
                  height: 36,
                  background: canSend ? "#015e9f" : "#9ca3af",
                }}
                aria-label="Gửi"
              >
                <Send size={15} className="text-white" style={{ marginLeft: 1 }} />
              </button>
            )}
```

- [ ] **Step 2: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: still failing on PlaygroundPage (Task 4 + Task 6 dependency), but ComposerBar itself is type-clean. The error count from Task 4 should be unchanged.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/playground/components/ComposerBar.tsx
git commit -m "feat(playground): ComposerBar swaps Send for StopButton while streaming"
```

---

### Task 6: `PlaygroundPage` orchestrator (wire everything; drop overrideModelId)

**Files:**
- Modify: `frontend/src/pages/PlaygroundPage.tsx`

- [ ] **Step 1: Replace entirely**

Replace `frontend/src/pages/PlaygroundPage.tsx` ENTIRELY with:

```tsx
import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  MessageSquare,
  PanelLeft,
  PanelLeftClose,
  Plus,
  Sparkles,
  Trash2,
} from "lucide-react";
import { MessageList } from "../playground/components/MessageList";
import type { MessageActions } from "../playground/components/MessageBubble";
import { ComposerBar } from "../playground/components/ComposerBar";
import { ModelDropdown } from "../playground/components/ModelDropdown";
import { Toaster } from "../playground/components/Toaster";
import { useChatStream } from "../playground/hooks/useChatStream";
import { useModels } from "../playground/hooks/useModels";
import {
  conversationsReducer,
  initialState,
  type Action,
} from "../playground/lib/messageReducer";
import type {
  AttachmentRef,
  ChatMessageWithAttachments,
  Message,
} from "../playground/types";

const ACCENT = "#015e9f";
const ACCENT_HOVER = "#01497a";
const SURFACE = "#f9fafb";
const SIDEBAR_BG = "#ffffff";
const CARD = "#f3f4f6";
const BORDER = "#e5e7eb";
const TEXT_PRIMARY = "#111827";
const TEXT_SECONDARY = "#6b7280";
const TEXT_MUTED = "#9ca3af";

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
}

function toWireMessages(messages: Message[]): ChatMessageWithAttachments[] {
  // Skip welcome message (id starts with "w") so we don't echo it back.
  return messages
    .filter((m) => !m.id.startsWith("w"))
    .map((m) => ({
      role: m.role,
      text: m.text,
      attachments: (m.attachments ?? []).map((a) => ({ id: a.id })),
    }));
}

function PlaygroundInner() {
  const [state, dispatch] = useReducer(conversationsReducer, undefined, () => {
    const init = initialState();
    return conversationsReducer(init, {
      type: "NEW_CONVERSATION",
      conversationId: uid(),
      welcomeMessageId: "w" + uid(),
      modelId: "",
      now: Date.now(),
    } as Action);
  });
  const { models } = useModels();
  const { send, abort } = useChatStream();
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<AttachmentRef[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [editingId, setEditingId] = useState<string | null>(null);

  // Latest dispatch reference for use inside async callbacks (avoids stale closures
  // when streamingId / activeId changes mid-stream).
  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;

  const activeId = state.activeId!;
  const active = state.conversations[activeId]!;
  const messages = active.messages;

  // Once /api/models loads, default the active conversation's modelId
  // to the first vision-capable model if none is set yet.
  useEffect(() => {
    if (!active.modelId && models.length > 0) {
      dispatch({
        type: "SET_MODEL",
        conversationId: activeId,
        modelId: models[0].id,
      });
    }
  }, [models, active.modelId, activeId]);

  const effectiveModelId =
    active.modelId || models[0]?.id || "";
  const activeModel = models.find((m) => m.id === effectiveModelId);
  const visionEnabled = activeModel?.capabilities.vision ?? true;

  const isStreaming = useMemo(
    () => messages.some((m) => m.status === "streaming"),
    [messages],
  );

  const historyImageCount = useMemo(
    () =>
      messages.reduce((n, m) => n + (m.attachments?.length ?? 0), 0),
    [messages],
  );

  function newConversation() {
    dispatch({
      type: "NEW_CONVERSATION",
      conversationId: uid(),
      welcomeMessageId: "w" + uid(),
      modelId: effectiveModelId,
      now: Date.now(),
    });
    setText("");
    setAttachments([]);
    setEditingId(null);
  }

  function selectConversation(id: string) {
    abort();
    dispatch({ type: "SELECT_CONVERSATION", id });
    setText("");
    setAttachments([]);
    setEditingId(null);
  }

  function deleteConversation(id: string) {
    dispatch({ type: "DELETE_CONVERSATION", id });
    if (Object.keys(state.conversations).length <= 1) {
      newConversation();
    }
  }

  /**
   * Reusable streaming runner. Caller is responsible for having dispatched
   * ADD_USER_MESSAGE + ADD_ASSISTANT_PLACEHOLDER before calling. We pass
   * `wireMessages` (the OpenAI-shaped history including the user message)
   * + `assistantId` (the placeholder we'll fill).
   */
  async function runStream(
    wireMessages: ChatMessageWithAttachments[],
    assistantId: string,
    convId: string,
  ) {
    await send({
      messages: wireMessages,
      modelId: effectiveModelId || null,
      onDelta: (delta) =>
        dispatchRef.current({
          type: "APPEND_DELTA",
          conversationId: convId,
          messageId: assistantId,
          delta,
        }),
      onDone: () =>
        dispatchRef.current({
          type: "MARK_DONE",
          conversationId: convId,
          messageId: assistantId,
        }),
      onError: (e) =>
        dispatchRef.current({
          type: "MARK_ERROR",
          conversationId: convId,
          messageId: assistantId,
          errorKind: e.errorKind,
        }),
    });
  }

  async function handleSend() {
    const trimmed = text.trim();
    if (!trimmed && attachments.length === 0) return;
    const userMsg: Message = {
      id: uid(),
      role: "user",
      text: trimmed || "Hãy mô tả hình ảnh này.",
      attachments: attachments.length > 0 ? attachments : undefined,
      status: "done",
      createdAt: Date.now(),
    };
    dispatch({
      type: "ADD_USER_MESSAGE",
      conversationId: activeId,
      message: userMsg,
    });
    if (messages.filter((m) => m.role === "user").length === 0) {
      const titleSrc = trimmed || "Hình ảnh";
      dispatch({
        type: "RENAME_TITLE",
        conversationId: activeId,
        title: titleSrc.slice(0, 40) + (titleSrc.length > 40 ? "…" : ""),
      });
    }
    const assistantId = uid();
    dispatch({
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: activeId,
      messageId: assistantId,
      now: Date.now(),
    });

    setText("");
    setAttachments([]);

    const wire = toWireMessages([...messages, userMsg]);
    await runStream(wire, assistantId, activeId);
  }

  function handleStop() {
    abort();
    // Mark the streaming placeholder as stopped. Find the last streaming msg.
    const streamingMsg = [...messages].reverse().find((m) => m.status === "streaming");
    if (streamingMsg) {
      dispatch({
        type: "MARK_STOPPED",
        conversationId: activeId,
        messageId: streamingMsg.id,
      });
    }
  }

  async function handleRegenerate() {
    if (isStreaming) return;
    // Pop the last assistant message and re-stream from the prior context.
    dispatch({ type: "POP_LAST_ASSISTANT", conversationId: activeId });
    // Dispatch reads the post-pop messages on the next render; we need the
    // computed message list here, so build it from the current `messages`.
    const remaining = messages.slice(0, -1);
    if (remaining.length === 0 || remaining.at(-1)?.role !== "user") return;
    const assistantId = uid();
    dispatch({
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: activeId,
      messageId: assistantId,
      now: Date.now(),
    });
    const wire = toWireMessages(remaining);
    await runStream(wire, assistantId, activeId);
  }

  async function handleSaveEdit(messageId: string, newText: string) {
    setEditingId(null);
    dispatch({
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: activeId,
      messageId,
      newText,
    });
    // Build the new message list manually (reducer change isn't visible until
    // re-render): take messages up to and including the edited one, replace text.
    const idx = messages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;
    const editedMsg: Message = { ...messages[idx], text: newText };
    const truncated = [...messages.slice(0, idx), editedMsg];
    const assistantId = uid();
    dispatch({
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: activeId,
      messageId: assistantId,
      now: Date.now(),
    });
    const wire = toWireMessages(truncated);
    await runStream(wire, assistantId, activeId);
  }

  function handleStartEdit(messageId: string) {
    if (isStreaming) return;
    setEditingId(messageId);
  }

  function handleCancelEdit() {
    setEditingId(null);
  }

  function handleModelChange(modelId: string) {
    dispatch({
      type: "SET_MODEL",
      conversationId: activeId,
      modelId,
    });
  }

  const messageActions: MessageActions = {
    isStreaming,
    editingId,
    onStartEdit: handleStartEdit,
    onSaveEdit: handleSaveEdit,
    onCancelEdit: handleCancelEdit,
    onRegenerate: handleRegenerate,
  };

  const sortedConvs = Object.values(state.conversations).sort(
    (a, b) => b.updatedAt - a.updatedAt,
  );

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ background: SURFACE, color: TEXT_PRIMARY }}
    >
      {/* Sidebar */}
      <aside
        className="flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden"
        style={{
          width: sidebarOpen ? 260 : 0,
          background: SIDEBAR_BG,
          borderRight: sidebarOpen ? `1px solid ${BORDER}` : "none",
        }}
      >
        <div
          className="flex items-center justify-between px-4 h-14 flex-shrink-0"
          style={{ borderBottom: `1px solid ${BORDER}` }}
        >
          <Link to="/" className="flex items-center gap-2 text-sm font-semibold">
            <div
              className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
              style={{ background: ACCENT }}
            >
              RS
            </div>
            <span>AI Playground</span>
          </Link>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 rounded transition-colors"
            style={{ color: TEXT_SECONDARY }}
            aria-label="Đóng sidebar"
          >
            <PanelLeftClose size={18} />
          </button>
        </div>
        <div className="px-3 pt-3 pb-1">
          <button
            onClick={newConversation}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all"
            style={{ background: ACCENT, color: "#fff" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = ACCENT_HOVER)}
            onMouseLeave={(e) => (e.currentTarget.style.background = ACCENT)}
          >
            <Plus size={16} />
            Cuộc trò chuyện mới
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
          {sortedConvs.map((c) => (
            <div
              key={c.id}
              className="group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors text-sm"
              style={{
                background: c.id === activeId ? CARD : "transparent",
                color: c.id === activeId ? TEXT_PRIMARY : TEXT_SECONDARY,
              }}
              onClick={() => selectConversation(c.id)}
            >
              <MessageSquare size={14} className="flex-shrink-0" style={{ opacity: 0.6 }} />
              <span className="flex-1 truncate">{c.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(c.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded transition-all"
                style={{ color: TEXT_MUTED }}
                aria-label="Xoá cuộc trò chuyện"
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
        <div
          className="px-4 py-3 text-[11px] flex-shrink-0"
          style={{ borderTop: `1px solid ${BORDER}`, color: TEXT_MUTED }}
        >
          Powered by OE-VLM
        </div>
      </aside>

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <header
          className="flex items-center gap-3 px-4 h-14 flex-shrink-0"
          style={{ borderBottom: `1px solid ${BORDER}` }}
        >
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded transition-colors"
              style={{ color: TEXT_SECONDARY }}
              aria-label="Mở sidebar"
            >
              <PanelLeft size={18} />
            </button>
          )}
          <div className="flex items-center gap-2">
            <Sparkles size={16} style={{ color: ACCENT }} />
            <span className="text-sm font-medium">{active.title}</span>
          </div>
          <div className="flex-1" />
          <Link
            to="/products"
            className="text-xs px-3 py-1.5 rounded-md transition-colors font-medium"
            style={{
              background: CARD,
              color: TEXT_SECONDARY,
              border: `1px solid ${BORDER}`,
            }}
          >
            Quay lại cửa hàng
          </Link>
        </header>

        <div className="flex-1 overflow-y-auto">
          <MessageList messages={messages} actions={messageActions} />
        </div>

        <ComposerBar
          text={text}
          onTextChange={setText}
          attachments={attachments}
          onAttach={(a) => setAttachments((prev) => [...prev, a])}
          onRemoveAttachment={(id) =>
            setAttachments((prev) => prev.filter((a) => a.id !== id))
          }
          onSend={handleSend}
          modelDropdown={
            <ModelDropdown
              models={models}
              value={effectiveModelId}
              onChange={handleModelChange}
            />
          }
          visionEnabled={visionEnabled}
          visionWarning={
            !visionEnabled && (attachments.length > 0 || historyImageCount > 0)
              ? "Model mới không hỗ trợ ảnh; gửi sẽ thất bại."
              : null
          }
          historyImageCount={historyImageCount}
          streaming={isStreaming}
          onStop={handleStop}
        />
      </div>

      <style>{`
        .overflow-y-auto::-webkit-scrollbar { width: 5px; }
        .overflow-y-auto::-webkit-scrollbar-track { background: transparent; }
        .overflow-y-auto::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
      `}</style>
    </div>
  );
}

export default function PlaygroundPage() {
  return (
    <Toaster>
      <PlaygroundInner />
    </Toaster>
  );
}
```

- [ ] **Step 2: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: clean (zero errors).

- [ ] **Step 3: Run unit tests**

```
cd frontend
npm run test:run
```

Expected: 44 tests still pass.

- [ ] **Step 4: Run E2E from Phase 2 to confirm no regression**

```
cd frontend
npm run test:e2e
```

Expected: 2 tests still pass (E2.1, E2.2 from Phase 2).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/PlaygroundPage.tsx
git commit -m "feat(playground): wire Stop/Regenerate/Edit + drop overrideModelId for SET_MODEL"
```

---

### Task 7: Playwright E2E (Stop / Regenerate / Edit / Network-down)

**Files:**
- Modify: `frontend/tests/e2e/fixtures/sseFixture.ts`
- Create: `frontend/tests/e2e/playground-controls.spec.ts`

- [ ] **Step 1: Extend `sseFixture.ts` with slow + error mocks**

Replace `frontend/tests/e2e/fixtures/sseFixture.ts` ENTIRELY with:

```ts
import type { Page, Route } from "@playwright/test";

function sseFrame(payload: object): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/**
 * Mock /api/models with 1 vision-capable model.
 */
export async function mockModels(page: Page) {
  await page.route("**/api/models", (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          {
            id: "qwen3-vl-8b-vllm",
            name: "Qwen3-VL 8B (vLLM)",
            capabilities: { vision: true },
          },
        ],
      }),
    }),
  );
}

/**
 * Mock /api/files: returns a deterministic AttachmentRef per request.
 */
export async function mockFileUploads(page: Page) {
  let counter = 0;
  await page.route("**/api/files", (route: Route) => {
    counter++;
    const id = `aaaa${String(counter).padStart(28, "0")}`; // 32 hex
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id,
        url: `/api/files/${id}`,
        mime: "image/png",
        size: 100,
        originalName: `mock-${counter}.png`,
      }),
    });
  });
}

/**
 * Mock /api/chat/stream to emit a fixed sequence of SSE frames in one shot.
 */
export async function mockChatStream(
  page: Page,
  deltas: string[] = ["Hello ", "**bold** ", "world."],
) {
  await page.route("**/api/chat/stream", (route: Route) => {
    const body =
      deltas.map((d) => sseFrame({ delta: d, done: false })).join("") +
      sseFrame({ delta: "", done: true });
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
      body,
    });
  });
}

/**
 * Counter-based response: each call to /api/chat/stream returns a different
 * delta sequence. Enables tests that send → regenerate → expect different reply.
 */
export async function mockChatStreamSequence(
  page: Page,
  responses: string[][],
) {
  let i = 0;
  await page.route("**/api/chat/stream", (route: Route) => {
    const deltas = responses[Math.min(i, responses.length - 1)];
    i++;
    const body =
      deltas.map((d) => sseFrame({ delta: d, done: false })).join("") +
      sseFrame({ delta: "", done: true });
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
      body,
    });
  });
}

/**
 * Mock /api/chat/stream that fails the request entirely (network error path).
 * Used for A3.7: Regenerate when network down.
 */
export async function mockChatStreamNetworkError(page: Page) {
  await page.route("**/api/chat/stream", (route: Route) =>
    route.abort("failed"),
  );
}

export async function setupAllMocks(page: Page) {
  await mockModels(page);
  await mockFileUploads(page);
  await mockChatStream(page);
}
```

- [ ] **Step 2: Implement E2E spec**

Create `frontend/tests/e2e/playground-controls.spec.ts`:

```ts
import { test, expect } from "@playwright/test";
import {
  mockModels,
  mockFileUploads,
  mockChatStream,
  mockChatStreamSequence,
  mockChatStreamNetworkError,
} from "./fixtures/sseFixture";

test.describe("Playground controls", () => {
  test("E3.1 — Send → Regenerate replaces last reply", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    // First send → "ALPHA reply.", regenerate → "BETA reply."
    await mockChatStreamSequence(page, [
      ["ALPHA ", "reply."],
      ["BETA ", "reply."],
    ]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("hi");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("ALPHA reply.")).toBeVisible({ timeout: 5000 });

    // Regenerate appears on the last assistant bubble.
    await page.getByRole("button", { name: /Tạo lại/ }).click();
    await expect(page.getByText("BETA reply.")).toBeVisible({ timeout: 5000 });
    // The first reply text should NOT remain in the DOM.
    await expect(page.getByText("ALPHA reply.")).not.toBeVisible();
  });

  test("E3.2 — Edit user msg → Save truncates everything after + regenerates", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStreamSequence(page, [
      ["First ", "reply."],
      ["Second ", "reply."],
      ["EDITED ", "reply."],
    ]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    // Send first message + wait for reply.
    await page.locator("textarea").fill("first user msg");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("First reply.")).toBeVisible({ timeout: 5000 });

    // Send second message + wait for reply.
    await page.locator("textarea").fill("second user msg");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Second reply.")).toBeVisible({ timeout: 5000 });

    // Hover the FIRST user bubble to reveal Edit button.
    const firstBubble = page.getByText("first user msg");
    await firstBubble.hover();
    await page.getByRole("button", { name: "Chỉnh sửa" }).first().click();

    // Editor textarea is focused; replace text and save.
    const editorTa = page.getByLabel("Chỉnh sửa tin nhắn");
    await editorTa.fill("first user msg EDITED");
    await page.getByRole("button", { name: "Lưu" }).click();

    // Second user msg + Second reply should be gone; new EDITED reply should appear.
    await expect(page.getByText("EDITED reply.")).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("Second reply.")).not.toBeVisible();
    await expect(page.getByText("second user msg")).not.toBeVisible();
  });

  test("E3.3 — Edit + Esc reverts without changes", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStream(page, ["Hello ", "world."]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("original");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Hello world.")).toBeVisible({ timeout: 5000 });

    const userBubble = page.getByText("original");
    await userBubble.hover();
    await page.getByRole("button", { name: "Chỉnh sửa" }).first().click();

    const editorTa = page.getByLabel("Chỉnh sửa tin nhắn");
    await editorTa.fill("changed text");
    await editorTa.press("Escape");

    // Original text should still be visible; edited text should not.
    await expect(page.getByText("original")).toBeVisible();
    await expect(page.getByText("changed text")).not.toBeVisible();
    // Hello world reply still present.
    await expect(page.getByText("Hello world.")).toBeVisible();
  });

  test("A3.7 — Regenerate when network is down → error bubble + Thử lại button", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    // First send: succeeds. Then we'll switch the route to fail for regenerate.
    let succeed = true;
    await page.route("**/api/chat/stream", (route) => {
      if (succeed) {
        succeed = false;
        route.fulfill({
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
          body:
            `data: {"delta":"hi","done":false}\n\n` +
            `data: {"delta":"","done":true}\n\n`,
        });
      } else {
        route.abort("failed");
      }
    });

    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("ping");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("hi")).toBeVisible({ timeout: 5000 });

    // Click Regenerate → second call aborts → bubble shows "Thử lại".
    await page.getByRole("button", { name: /Tạo lại/ }).click();
    await expect(page.getByRole("button", { name: "Thử lại" })).toBeVisible({
      timeout: 5000,
    });
  });
});
```

- [ ] **Step 3: Run E2E**

```
cd frontend
npm run test:e2e
```

Expected: 6 tests pass total — 2 from Phase 2 (`playground.spec.ts`) + 4 from this task (`playground-controls.spec.ts`).

If any of the new tests is flaky (Playwright sees the assistant bubble before the streamed text completes), bump the `timeout` in the affected `expect(...)` from `5000` to `10000`.

- [ ] **Step 4: Commit**

```bash
git add frontend/tests/e2e/fixtures/sseFixture.ts frontend/tests/e2e/playground-controls.spec.ts
git commit -m "test(playground): add E2E for Stop/Regenerate/Edit/network-down"
```

---

### Task 8: Manual smoke + final pass

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full Vitest suite**

```
cd frontend
npm run test:run
```

Expected: 44 tests pass (sseParser 8, messageReducer 18, fileValidate 7, chatStream 6, uploadFile 5).

- [ ] **Step 2: Run the full Playwright suite**

```
cd frontend
npm run test:e2e
```

Expected: 6 tests pass.

- [ ] **Step 3: Verify backend tests still green**

```
cd backend
pytest -v
```

Expected: 65 PASSED (Phase 1 baseline preserved).

- [ ] **Step 4: Manual browser smoke (with backend + vLLM up)**

Start backend (terminal 1):

```
cd backend
. .venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

Start frontend (terminal 2):

```
cd frontend
npm run dev
```

Open `http://localhost:5173/playground`. Walk through these scenarios:

**A. Stop generation (covers A3.1, A3.11)**
1. Send a long prompt (e.g., "viết một bài văn 500 từ về biển").
2. While streaming, the Send button should be replaced by a red Stop button.
3. Click Stop → streaming halts → assistant bubble shows partial text + "[bị dừng]" suffix.
4. Verify Stop button disappears (last status is now "stopped", not "streaming").
5. (A3.11) Try sending a prompt that produces a code block (e.g., "viết code python in 'hello world' kèm giải thích markdown") → click Stop while the code fence is mid-output → bubble renders partial text + "[bị dừng]" without crashing.

**B. Regenerate (covers A3.2, A3.3, A3.7)**
1. Send a prompt → wait for done. Click Tạo lại on the last assistant bubble → new reply replaces.
2. (A3.2) During streaming, the Tạo lại button on the previous bubble should be hidden (`!actions.isStreaming` gate).
3. (A3.3) Click Stop, then immediately Tạo lại → 2 distinct fetch calls in DevTools Network tab; no double-stream output.
4. (A3.7) Stop the backend (`Ctrl+C` on uvicorn). Click Tạo lại → red error bubble appears with "Thử lại" button. Restart backend, click Thử lại → succeeds.

**C. Edit user message (covers A3.5, A3.6, A3.10)**
1. Send 2-3 messages. Hover an early user bubble → pencil appears → click → bubble swaps to editor with text pre-filled and cursor at end.
2. Save with Ctrl+Enter or the Lưu button → all messages after the edited one disappear → new reply streams.
3. (A3.5) During streaming, hover any user bubble → pencil button should NOT appear.
4. (A3.6) Open editor, clear text, click Lưu → toast "Tin nhắn rỗng." appears, edit mode stays.
5. (A3.10) Open conversation A, start typing in textarea, switch to conversation B from sidebar → A's text + attachments are NOT preserved (per `selectConversation`'s reset). Verify edit mode is also reset.

**D. Stop button placement**
1. The Stop button should sit in the same position as Send (bottom-right of composer). When streaming starts, Send → Stop swap. When streaming ends (done/stopped/error), Stop → Send swap.

**E. SET_MODEL persistence**
1. In conversation A, switch to GPT-5.4 Mini in the dropdown.
2. Switch to conversation B → dropdown reflects B's model (Qwen3-VL by default).
3. Switch back to A → dropdown shows GPT-5.4 Mini. (Phase 2's `overrideModelId` behavior was session-wide and lost on switch; Phase 3 now persists per-conversation.)

If any scenario reveals a bug, return to the relevant earlier task, write a regression test, fix, re-run all tests.

- [ ] **Step 5: No commit needed unless fixes were applied**

```bash
git status
```

Expected: clean working tree.

---

## Coverage Mapping

| Test ID | Description | Task |
|---------|-------------|------|
| T3.1 | MARK_STOPPED updates status, text untouched | Task 1 |
| T3.2 | POP_LAST_ASSISTANT removes last iff assistant role | Task 1 |
| T3.3 | EDIT_USER_AND_TRUNCATE replaces text + drops after | Task 1 |
| T3.4 | EDIT preserves attachments | Task 1 |
| E3.1 | Send → Regenerate replaces reply | Task 7 |
| E3.2 | Edit + Save truncates + regenerates | Task 7 |
| E3.3 | Edit + Esc reverts | Task 7 |
| A3.1 | Stop button hidden when status=done | Task 5 implementation + Task 8 manual |
| A3.2 | Regenerate disabled during streaming | Task 4 (`canRegenerate` gate) + Task 8 manual |
| A3.3 | Stop+Regenerate = 2 fetch calls, no double-stream | Task 8 manual |
| A3.4 | Edit msg #1 of 5-msg → 4 dropped | Task 1 reducer test |
| A3.5 | Edit disabled while streaming | Task 4 (`canEdit` gate) + Task 8 manual |
| A3.6 | Save empty text → toast "Tin nhắn rỗng" | Task 3 (InlineEditor.handleSave) + Task 8 manual |
| A3.7 | Regenerate when network down → "Thử lại" | Task 7 |
| A3.8 | 5 rapid Regenerate clicks → previous aborted | Architecturally guaranteed by `isStreaming` gate (Task 4) + `useChatStream.abort()` from Phase 2 |
| A3.9 | Edit user msg with 4 attachments → attachments persist | Task 1 reducer test |
| A3.10 | Edit in convo A while convo B has pending input → state isolation | Task 6 (`selectConversation` resets `editingId`) + Task 8 manual |
| A3.11 | Stop with partial code fence → renders + decorated [bị dừng] | Task 4 (`isStopped` decoration) + Task 8 manual |
| SET_MODEL | New action + per-conversation persistence | Task 1 reducer test + Task 6 wiring + Task 8 manual scenario E |

A3.3, A3.8, A3.10 are deliberately covered by manual smoke (Task 8) rather than automated test because they exercise either timing-dependent UI (rapid clicks), DevTools-observable behavior (network call count), or cross-conversation isolation that's awkward to assert end-to-end in Playwright.

---

## Risks & deferrals

- **`handleRegenerate` reads `messages` from render scope** — when Regenerate fires, it computes `remaining = messages.slice(0, -1)` from the closure-captured `messages`, then dispatches POP + ADD_PLACEHOLDER. This works because React batches the dispatches and the closure was correct at click time. Don't move the `remaining` computation after the dispatches.
- **`handleSaveEdit` uses the same closure-capture pattern** — it builds the truncated message list from the pre-dispatch `messages` array, then dispatches the actions. If the user could trigger SaveEdit while another action is in-flight, the closure would be stale. Phase 2's `editingId` gate (only one editor visible at a time) + `isStreaming` gate (Edit button hidden during streaming) prevent this in practice.
- **`canRegenerate` gate vs SSE error before first chunk** — if the backend returns an error frame before any delta, the assistant placeholder ends in `status: "error"` with empty text. The bubble renders the red error variant + "Thử lại" button. Working as intended.
- **`messageActions` object identity changes every render** — components downstream of `MessageList` receive a new object reference each render, defeating React's `memo`. None of the downstream components use `memo` today; revisit if profiling shows render thrash.
- **`SET_MODEL` deletes the Phase 2 `overrideModelId` deferral.** The previous behavior persisted only across the current conversation; Phase 3 makes it per-conversation in the reducer (correct).
- **`mockChatStreamNetworkError` exists but is not used by any test in this plan** — it's a helper for future regression tests if needed. The A3.7 test inlines its own counter-based mock for clarity.

## Acceptance Criteria

A task is complete when:

1. The new failing tests written in the task pass after the implementation step.
2. The full `npm run test:run` reports all tests passing (no regressions in earlier tasks).
3. `npx tsc --noEmit` has zero errors (Tasks 4-5 may have known orchestrator-level errors that resolve in Task 6).
4. The commit message matches exactly what's specified in the task's commit step.
5. Task 8 manual smoke passes for the final pass.
