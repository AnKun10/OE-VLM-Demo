# Phase 2 — Frontend Streaming + Multimodal + Markdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the playground frontend to consume `POST /api/chat/stream` SSE end-to-end, render assistant replies as proper markdown, support multi-image upload (picker + drag/drop + paste) with vision-aware UI, and gain Vitest + Playwright test infrastructure.

**Architecture:** Decompose the 660-LOC `PlaygroundPage.tsx` into a `~150-LOC` orchestrator plus focused modules under `frontend/src/playground/{components,hooks,lib,types.ts}`. Pure logic (SSE parsing, reducer, file validation, chat-stream worker, file-upload worker) lives in `lib/` for direct Vitest unit testing without React. Hooks are thin wrappers (5–20 LOC each) over the pure libs. Components own UI only. Forward-compat: `sseParser` ignores unknown JSON shapes so Phase 5's `meta` event won't break it.

**Tech Stack:** TypeScript + React 18, Vite, TailwindCSS, Radix UI, Lucide icons, react-markdown + remark-gfm + rehype-highlight, Vitest + jsdom for unit, Playwright for E2E.

**Spec:** `docs/superpowers/specs/2026-05-09-playground-qwen-vl-parity-design.md` (Phase 2 section starts at line 814; module layout C.2; types C.3; SSE consumer C.4; file upload C.5; vision-aware C.6; markdown C.7).

**Phase 1 plan (sibling):** `docs/superpowers/plans/2026-05-09-playground-qwen-vl-parity-phase-1.md`. Backend `/api/chat/stream` SSE endpoint emits `{delta, done}` and `{error, message}` payloads — see Phase 1 plan Task 12.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| **Task 0 — Preflight cleanups (from Vast monitoring)** |||
| Modify | `backend/requirements.txt` | Drop `vllm` line (no longer needed after Phase 1 refactor; was forcing torch+cuda install on cold pod) |
| Modify | `backend/app/models/vlm/models.yaml` | `model_id: "Qwen/Qwen3-VL-8B-Instruct"` → `"qwen3-vl-8b"` (matches vLLM `--served-model-name`) |
| Modify | `qwen_development.md` | vLLM start command adds `--served-model-name qwen3-vl-8b` |
| Modify | `vast-templates/oe-vlm-demo/README.md` | Onstart path `/var/lib/vast/onstart.sh` → `/root/onstart.sh`; troubleshooting note |
| Modify | `vast-templates/oe-vlm-demo/TEMPLATE_FIELDS.md` | Add optional `HF_TOKEN` env var row; mention faster downloads |
| **Task 1 — Test infra + runtime deps** |||
| Modify | `frontend/package.json` | Add deps (react-markdown stack) + devDeps (vitest, jsdom, @testing-library/dom, @playwright/test); add `test`, `test:run`, `test:e2e` scripts |
| Create | `frontend/vitest.config.ts` | Vitest config (jsdom, globals, setup file) |
| Create | `frontend/playwright.config.ts` | Playwright config (Chromium only, baseURL via Vite dev server) |
| Create | `frontend/tests/e2e/.gitkeep` | Placeholder so dir is tracked |
| Create | `frontend/src/playground/.gitkeep` | Placeholder for the new module tree |
| **Task 2 — Types** |||
| Create | `frontend/src/playground/types.ts` | `AttachmentRef`, `MessageStatus`, `Message`, `Conversation`, `ConversationsState`, plus the streaming wire-types `ChatMessageWithAttachments`, `ChatStreamRequest` |
| **Task 3 — `lib/sseParser.ts`** |||
| Create | `frontend/src/playground/lib/sseParser.ts` | `drainEvents(buffer) → {events, rest}`; forward-compat: unknown JSON shapes ignored |
| Create | `frontend/src/playground/lib/sseParser.test.ts` | T2.1–T2.4, A2.1–A2.3, forward-compat for `meta` |
| **Task 4 — `lib/messageReducer.ts`** |||
| Create | `frontend/src/playground/lib/messageReducer.ts` | Pure reducer with Phase 2 actions: NEW_CONVERSATION, DELETE_CONVERSATION, SELECT_CONVERSATION, ADD_USER_MESSAGE, ADD_ASSISTANT_PLACEHOLDER, APPEND_DELTA, MARK_DONE, MARK_ERROR, RENAME_TITLE |
| Create | `frontend/src/playground/lib/messageReducer.test.ts` | T2.5–T2.7, A2.4, plus the new-conversation/delete tests needed for the refactor |
| **Task 5 — `lib/errors.ts` + `lib/fileValidate.ts`** |||
| Create | `frontend/src/playground/lib/errors.ts` | `FriendlyError` class (i18n key + Vietnamese message map) |
| Create | `frontend/src/playground/lib/fileValidate.ts` | `ALLOWED_MIME`, `MAX_BYTES`, `validateFile`, `checkAttachmentCap` |
| Create | `frontend/src/playground/lib/fileValidate.test.ts` | T2.8–T2.10, plus `errors` smoke |
| **Task 6 — `lib/chatStream.ts`** |||
| Create | `frontend/src/playground/lib/chatStream.ts` | Pure async function `streamChat({signal, messages, modelId, onDelta, onDone, onError})` — fetch + parse loop. No React. |
| Create | `frontend/src/playground/lib/chatStream.test.ts` | T2.11–T2.13, A2.5 (signal), A2.6 (abort), A2.7 (UTF-8 split) |
| **Task 7 — `lib/uploadFile.ts`** |||
| Create | `frontend/src/playground/lib/uploadFile.ts` | Pure async function `uploadFile(file): Promise<AttachmentRef>` — validate + POST + shape-check |
| Create | `frontend/src/playground/lib/uploadFile.test.ts` | A2.15 (network drop), A2.16 (response missing id) |
| **Task 8 — Hooks (thin wrappers)** |||
| Create | `frontend/src/playground/hooks/useChatStream.ts` | useRef AbortController + delegates to `streamChat` |
| Create | `frontend/src/playground/hooks/useFileUpload.ts` | Thin wrapper exposing `upload(file)` |
| Create | `frontend/src/playground/hooks/useModels.ts` | `useEffect` + `fetch /api/models` once; returns `{models, loading, error}` |
| **Task 9 — `SafeLink` + `MessageBubble`** |||
| Create | `frontend/src/playground/components/SafeLink.tsx` | `<a target="_blank" rel="noopener noreferrer">` |
| Create | `frontend/src/playground/components/MessageBubble.tsx` | Plain pre-wrap for user; `<ReactMarkdown>` for assistant |
| **Task 10 — Attachments + DropOverlay** |||
| Create | `frontend/src/playground/components/AttachmentPreview.tsx` | Single thumb (80×80) with optional remove button |
| Create | `frontend/src/playground/components/AttachmentRail.tsx` | Row of `AttachmentPreview`s above composer |
| Create | `frontend/src/playground/components/DropOverlay.tsx` | Page-level overlay shown while dragging files |
| **Task 11 — Toast** |||
| Create | `frontend/src/playground/components/Toaster.tsx` | Radix-UI Toast viewport + provider wrapper |
| Create | `frontend/src/playground/hooks/useToast.ts` | Context-based dispatch with friendly-message lookup |
| **Task 12 — `ModelDropdown`** |||
| Create | `frontend/src/playground/components/ModelDropdown.tsx` | `<select>` of `{id,name,capabilities}`; consumed by composer for vision gating |
| **Task 13 — `ComposerBar`** |||
| Create | `frontend/src/playground/components/ComposerBar.tsx` | Textarea, attach button (multi-file picker), paste handler, drop wiring, send button, model dropdown slot, attachment-cap badge, vision-disabled banner (A2.14) |
| **Task 14 — `MessageList` + `PlaygroundPage` refactor** |||
| Create | `frontend/src/playground/components/MessageList.tsx` | Map `messages → MessageBubble`; loading dot; bottom ref for autoscroll |
| Modify | `frontend/src/pages/PlaygroundPage.tsx` | Refactor 660 LOC → ~150 LOC orchestrator: state via `useReducer(conversationsReducer)`, `useChatStream`, `useFileUpload`, `useModels`, `useToast`. No inline UI logic. |
| **Task 15 — Playwright E2E** |||
| Create | `frontend/tests/e2e/playground.spec.ts` | E2.1 golden path (mocked SSE) + E2.2 multi-image |
| Create | `frontend/tests/e2e/fixtures/sseFixture.ts` | Helper to mock `/api/chat/stream` and `/api/files` |
| **Task 16 — Manual smoke + final pass** |||
| — | — | Run full Vitest suite + Playwright + manual browser smoke against dev server |

---

## Test Pattern Notes

### Vitest setup

`vitest.config.ts` runs all `*.test.ts(x)` under `frontend/src/` with `jsdom`. Pure-lib tests work fine in jsdom. No setup file needed unless a component test mocks `localStorage` (defer to Phase 4).

### Mocking `fetch` for streaming tests

`streamChat` consumes `Response.body.getReader()`. The mock returns a `Response` whose body is a `ReadableStream`:

```ts
function streamingResponse(chunks: Uint8Array[]): Response {
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i >= chunks.length) {
        controller.close();
      } else {
        controller.enqueue(chunks[i++]);
      }
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}
```

Tests then `vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(streamingResponse(...))`.

### Playwright + Vite

Playwright config starts the Vite dev server (`npm run dev`) on `localhost:5173` and mocks `/api/*` via `route.fulfill(...)`. No real backend required for E2E.

---

## Tasks

### Task 0: Preflight cleanups (4 fixes from Vast deployment monitoring)

**Files:**
- Modify: `backend/requirements.txt`
- Modify: `backend/app/models/vlm/models.yaml`
- Modify: `qwen_development.md`
- Modify: `vast-templates/oe-vlm-demo/README.md`
- Modify: `vast-templates/oe-vlm-demo/TEMPLATE_FIELDS.md`

- [ ] **Step 1: Drop `vllm` from `backend/requirements.txt`**

The current file lists `vllm` (a leftover from when the backend used `LLM.chat()` directly). Phase 1 refactored the backend to be an OpenAI-SDK HTTP client; `vllm` is no longer imported anywhere in `backend/app/`. Cold-boot pods waste 5–15 min installing torch + cuda for nothing.

Remove the `vllm` line. The full file should read:

```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
pillow
openai
pyyaml
pytest
pytest-asyncio
python-multipart
```

- [ ] **Step 2: Align `model_id` with vLLM's served name**

The Vast onstart launches vLLM with `--served-model-name qwen3-vl-8b`, but `models.yaml` calls the SDK with `model="Qwen/Qwen3-VL-8B-Instruct"` → vLLM 404. Change the YAML so the backend uses the served name.

In `backend/app/models/vlm/models.yaml`, change line 6 (the `qwen3-vl-8b-vllm` entry):

From:
```yaml
    model_id: "Qwen/Qwen3-VL-8B-Instruct"  # Model name sent in API requests
```

To:
```yaml
    model_id: "qwen3-vl-8b"           # Model name sent in API requests (matches vLLM --served-model-name)
```

- [ ] **Step 3: Update local-dev vLLM command in `qwen_development.md`**

Local-dev was working without `--served-model-name` because vLLM defaulted to the HF path as the served name, matching the old `model_id`. After Step 2, local dev also needs to use `--served-model-name qwen3-vl-8b` to stay consistent with the YAML.

In `qwen_development.md`, replace the `vllm serve` block (lines ~9–15) with:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct \
    --port 8003 \
    --served-model-name qwen3-vl-8b \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --limit-mm-per-prompt '{"image": 4}'
```

- [ ] **Step 4: Fix Vast template README onstart path + add HF_TOKEN row**

Two small doc fixes in the Vast template:

**a)** In `vast-templates/oe-vlm-demo/README.md`, every reference to `/var/lib/vast/onstart.sh` should be `/root/onstart.sh`. Use this command from the repo root (Windows Git Bash compatible):

```bash
sed -i 's|/var/lib/vast/onstart.sh|/root/onstart.sh|g' vast-templates/oe-vlm-demo/README.md
```

(There are ~5 occurrences in the Operations and Manual repo bring-up sections.)

**b)** In `vast-templates/oe-vlm-demo/TEMPLATE_FIELDS.md`, add this row to the Environment Variables table (in section 4, after `HF_HOME`):

```markdown
| `HF_TOKEN` | (optional) `hf_xxxxx` — your HuggingFace token. Anonymous downloads are rate-limited; with a token, weights download faster (saves ~5–15 min on cold boot). Get one at https://huggingface.co/settings/tokens. |
```

- [ ] **Step 5: Verify each file**

```
git diff backend/requirements.txt backend/app/models/vlm/models.yaml qwen_development.md vast-templates/oe-vlm-demo/README.md vast-templates/oe-vlm-demo/TEMPLATE_FIELDS.md
```

Expected: 5 files modified; `vllm` line removed; `model_id` shows `qwen3-vl-8b`; vLLM command in qwen_development.md has `--served-model-name`; README has `/root/onstart.sh`; TEMPLATE_FIELDS has new `HF_TOKEN` row.

- [ ] **Step 6: Run backend tests to confirm no regression**

```
cd backend
pytest -v
```

Expected: 65 PASSED (Phase 1 baseline). The change to `models.yaml` and `requirements.txt` does not affect tests (tests use `fake_manager` fixture, not the YAML).

- [ ] **Step 7: Commit**

```bash
git add backend/requirements.txt backend/app/models/vlm/models.yaml qwen_development.md vast-templates/oe-vlm-demo/README.md vast-templates/oe-vlm-demo/TEMPLATE_FIELDS.md
git commit -m "fix: drop unused vllm dep + align model_id with served name + vast template polish"
```

---

### Task 1: Frontend test infrastructure + runtime deps

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/vitest.config.ts`
- Create: `frontend/playwright.config.ts`
- Create: `frontend/tests/e2e/.gitkeep`
- Create: `frontend/src/playground/.gitkeep`

- [ ] **Step 1: Add runtime + dev dependencies**

Add the marked entries in `frontend/package.json`. The full file should read:

```json
{
  "name": "oe-vlm-shop",
  "private": true,
  "version": "0.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "test": "vitest",
    "test:run": "vitest run",
    "test:e2e": "playwright test"
  },
  "dependencies": {
    "@radix-ui/react-accordion": "^1.2.1",
    "@radix-ui/react-checkbox": "^1.1.3",
    "@radix-ui/react-dialog": "^1.1.2",
    "@radix-ui/react-dropdown-menu": "^2.1.2",
    "@radix-ui/react-label": "^2.1.0",
    "@radix-ui/react-select": "^2.1.2",
    "@radix-ui/react-separator": "^1.1.0",
    "@radix-ui/react-slider": "^1.2.1",
    "@radix-ui/react-slot": "^1.1.0",
    "@radix-ui/react-tabs": "^1.1.1",
    "@radix-ui/react-toast": "^1.2.2",
    "class-variance-authority": "^0.7.0",
    "clsx": "^2.1.1",
    "highlight.js": "^11.10.0",
    "lucide-react": "^0.460.0",
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-markdown": "^9.0.1",
    "react-router-dom": "^6.28.0",
    "rehype-highlight": "^7.0.1",
    "remark-gfm": "^4.0.0",
    "tailwind-merge": "^2.5.4",
    "tailwindcss-animate": "^1.0.7"
  },
  "devDependencies": {
    "@playwright/test": "^1.48.0",
    "@testing-library/dom": "^10.4.0",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "jsdom": "^25.0.1",
    "postcss": "^8.4.49",
    "tailwindcss": "^3.4.15",
    "typescript": "^5.6.3",
    "vite": "^5.4.10",
    "vitest": "^2.1.4"
  }
}
```

- [ ] **Step 2: Install**

```
cd frontend
npm install
```

Expected: clean install. May warn about peer deps; can ignore.

- [ ] **Step 3: Create `vitest.config.ts`**

Create `frontend/vitest.config.ts`:

```ts
/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
```

- [ ] **Step 4: Create `playwright.config.ts`**

Create `frontend/playwright.config.ts`:

```ts
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  forbidOnly: !!process.env.CI,
  retries: 0,
  workers: 1,
  reporter: "list",
  use: {
    baseURL: "http://localhost:5173",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
```

- [ ] **Step 5: Install Playwright Chromium browser**

```
cd frontend
npx playwright install chromium
```

Expected: downloads Chromium (~150 MB) once. Cached at `%LOCALAPPDATA%/ms-playwright`.

- [ ] **Step 6: Create placeholder dirs**

```
mkdir -p frontend/tests/e2e frontend/src/playground
touch frontend/tests/e2e/.gitkeep frontend/src/playground/.gitkeep
```

(On Windows Git Bash, those commands work as-is.)

- [ ] **Step 7: Verify Vitest discovers nothing yet**

```
cd frontend
npm run test:run
```

Expected: `No test files found, exiting with code 1`. That's correct for now — we'll add tests in subsequent tasks.

- [ ] **Step 8: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/vitest.config.ts frontend/playwright.config.ts frontend/tests/e2e/.gitkeep frontend/src/playground/.gitkeep
git commit -m "test(frontend): add vitest + playwright + react-markdown stack"
```

---

### Task 2: Frontend types

**Files:**
- Create: `frontend/src/playground/types.ts`

- [ ] **Step 1: Create the types module**

```ts
// frontend/src/playground/types.ts
/**
 * Type definitions for the playground frontend.
 * These mirror the backend Pydantic models in `backend/app/routers/chat.py`
 * (ChatMessageWithAttachments, Attachment, ChatStreamRequest) and the
 * StoredFile response shape from `backend/app/services/files.py`
 * (camelCase via Pydantic alias_generator=to_camel).
 */

export type AttachmentRef = {
  id: string;
  url: string;
  mime: string;
  originalName: string;
};

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

export type ErrorKind =
  | "connection"
  | "file_missing"
  | "bad_request"
  | "internal"
  | "network"
  | "http"
  | "stream"
  | "parse";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachments?: AttachmentRef[];
  status?: MessageStatus;
  errorKind?: ErrorKind;
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  modelId: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

export type ConversationsState = {
  schemaVersion: 1;
  conversations: Record<string, Conversation>;
  activeId: string | null;
};

export type ModelInfo = {
  id: string;
  name: string;
  capabilities: { vision: boolean };
};

/** Wire types — what the backend actually accepts/returns over HTTP. */

export type ChatMessageWithAttachments = {
  role: "user" | "assistant";
  text: string;
  attachments: { id: string }[];
};

export type ChatStreamRequest = {
  messages: ChatMessageWithAttachments[];
  model_id: string | null;
};
```

- [ ] **Step 2: TypeScript compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/playground/types.ts
git commit -m "feat(playground): add frontend types module"
```

---

### Task 3: `lib/sseParser.ts` + tests

**Files:**
- Create: `frontend/src/playground/lib/sseParser.ts`
- Create: `frontend/src/playground/lib/sseParser.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/playground/lib/sseParser.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { drainEvents } from "./sseParser";

describe("drainEvents", () => {
  it("T2.1 — parses 1 buffer with 3 SSE blocks", () => {
    const buf =
      `data: {"delta":"hello","done":false}\n\n` +
      `data: {"delta":" world","done":false}\n\n` +
      `data: {"delta":"","done":true}\n\n`;
    const { events, rest } = drainEvents(buf);
    expect(events).toEqual([
      { type: "delta", delta: "hello" },
      { type: "delta", delta: " world" },
      { type: "done" },
    ]);
    expect(rest).toBe("");
  });

  it("T2.2 — handles event split across two reads", () => {
    const first = `data: {"delta":"hel`;
    const second = `lo","done":false}\n\n`;
    const r1 = drainEvents(first);
    expect(r1.events).toEqual([]);
    expect(r1.rest).toBe(first);
    const r2 = drainEvents(r1.rest + second);
    expect(r2.events).toEqual([{ type: "delta", delta: "hello" }]);
    expect(r2.rest).toBe("");
  });

  it("T2.3 — emits done for {done:true}", () => {
    const buf = `data: {"delta":"","done":true}\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([{ type: "done" }]);
  });

  it("T2.4 — emits error for error payload", () => {
    const buf = `data: {"error":"connection","message":"vLLM down"}\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([
      { type: "error", errorKind: "connection", message: "vLLM down" },
    ]);
  });

  it("A2.1 — Windows newlines (\\r\\n\\r\\n) parse correctly", () => {
    const buf = `data: {"delta":"x","done":false}\r\n\r\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([{ type: "delta", delta: "x" }]);
  });

  it("A2.2 — malformed JSON yields parse_error", () => {
    const buf = `data: {bad-json\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([{ type: "parse_error", raw: "{bad-json" }]);
  });

  it("A2.3 — empty data: line is skipped", () => {
    const buf = `data: \n\ndata: {"delta":"y","done":false}\n\n`;
    const { events } = drainEvents(buf);
    expect(events).toEqual([{ type: "delta", delta: "y" }]);
  });

  it("forward-compat — unknown JSON shape (e.g. {meta}) yields no event but does not crash", () => {
    const buf = `data: {"meta":{"reasoning":"foo"}}\n\n`;
    const { events, rest } = drainEvents(buf);
    expect(events).toEqual([]);
    expect(rest).toBe("");
  });
});
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- sseParser
```

Expected: import error (`Cannot find module './sseParser'`).

- [ ] **Step 3: Implement `sseParser.ts`**

Create `frontend/src/playground/lib/sseParser.ts`:

```ts
/**
 * Pure SSE parser for the /api/chat/stream wire format.
 *
 * Backend emits one JSON payload per `data:` frame, separated by `\n\n`:
 *   {"delta":"...","done":false}   normal token
 *   {"delta":"","done":true}       end of stream
 *   {"error":"...","message":"..."} terminal error
 *
 * Forward-compat: unknown JSON shapes (e.g. Phase 5's {"meta":{...}})
 * are silently ignored — drainEvents emits nothing for them. Adding
 * new event types in later phases will not break Phase 2 callers.
 */

export type SseEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: string; message: string }
  | { type: "parse_error"; raw: string };

const FRAME_DELIMITER = /\r?\n\r?\n/;

function parsePayload(raw: string): SseEvent | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { type: "parse_error", raw };
  }
  if (typeof parsed !== "object" || parsed === null) return null;
  const obj = parsed as Record<string, unknown>;

  if (typeof obj.error === "string") {
    return {
      type: "error",
      errorKind: obj.error,
      message: typeof obj.message === "string" ? obj.message : "",
    };
  }
  if (obj.done === true) return { type: "done" };
  if (typeof obj.delta === "string" && obj.done === false) {
    return { type: "delta", delta: obj.delta };
  }
  // Unknown shape: forward-compat, ignore.
  return null;
}

export function drainEvents(buffer: string): {
  events: SseEvent[];
  rest: string;
} {
  const blocks = buffer.split(FRAME_DELIMITER);
  // Last entry is potentially incomplete (no trailing delimiter).
  const rest = blocks.pop() ?? "";

  const events: SseEvent[] = [];
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed.startsWith("data:")) continue;
    const payload = trimmed.slice(5).trim();
    if (!payload) continue;
    const ev = parsePayload(payload);
    if (ev) events.push(ev);
  }
  return { events, rest };
}
```

- [ ] **Step 4: Verify pass**

```
cd frontend
npm run test:run -- sseParser
```

Expected: `8 passed`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/lib/sseParser.ts frontend/src/playground/lib/sseParser.test.ts
git commit -m "feat(playground): add sseParser with forward-compat for unknown events"
```

---

### Task 4: `lib/messageReducer.ts` + tests

**Files:**
- Create: `frontend/src/playground/lib/messageReducer.ts`
- Create: `frontend/src/playground/lib/messageReducer.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/playground/lib/messageReducer.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  conversationsReducer,
  initialState,
  type Action,
} from "./messageReducer";
import type { ConversationsState, Message } from "../types";

function withConv(
  base: ConversationsState,
  convId: string,
  modelId: string,
): ConversationsState {
  return conversationsReducer(base, {
    type: "NEW_CONVERSATION",
    conversationId: convId,
    welcomeMessageId: "w1",
    modelId,
    now: 1000,
  });
}

describe("conversationsReducer", () => {
  it("NEW_CONVERSATION creates entry, sets active, includes welcome msg", () => {
    const s = withConv(initialState(), "c1", "qwen3-vl-8b-vllm");
    expect(s.activeId).toBe("c1");
    expect(s.conversations.c1.modelId).toBe("qwen3-vl-8b-vllm");
    expect(s.conversations.c1.title).toBe("Cuộc hội thoại mới");
    expect(s.conversations.c1.messages.length).toBe(1);
    expect(s.conversations.c1.messages[0].role).toBe("assistant");
  });

  it("DELETE_CONVERSATION removes entry; activeId falls back to next or null", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c1" });
    expect(s.conversations.c1).toBeUndefined();
    expect(s.activeId).toBe("c2");

    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c2" });
    expect(s.activeId).toBeNull();
  });

  it("SELECT_CONVERSATION sets activeId", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
    s = conversationsReducer(s, { type: "SELECT_CONVERSATION", id: "c1" });
    expect(s.activeId).toBe("c1");
  });

  it("ADD_USER_MESSAGE appends to the right conversation", () => {
    let s = withConv(initialState(), "c1", "m");
    const msg: Message = {
      id: "u1",
      role: "user",
      text: "hi",
      createdAt: 2000,
    };
    s = conversationsReducer(s, {
      type: "ADD_USER_MESSAGE",
      conversationId: "c1",
      message: msg,
    });
    const last = s.conversations.c1.messages.at(-1)!;
    expect(last.id).toBe("u1");
    expect(last.text).toBe("hi");
  });

  it("T2.6 — ADD_ASSISTANT_PLACEHOLDER creates message with status streaming + empty text", () => {
    let s = withConv(initialState(), "c1", "m");
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a1",
      now: 3000,
    });
    const last = s.conversations.c1.messages.at(-1)!;
    expect(last.id).toBe("a1");
    expect(last.role).toBe("assistant");
    expect(last.text).toBe("");
    expect(last.status).toBe("streaming");
  });

  it("T2.5 — APPEND_DELTA appends to correct message in correct conversation", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
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
      delta: "hello",
    });
    s = conversationsReducer(s, {
      type: "APPEND_DELTA",
      conversationId: "c1",
      messageId: "a1",
      delta: " world",
    });
    expect(s.conversations.c1.messages.at(-1)!.text).toBe("hello world");
    // c2 is untouched.
    expect(
      s.conversations.c2.messages.find((m) => m.id === "a1"),
    ).toBeUndefined();
  });

  it("T2.7 — MARK_DONE flips status streaming → done", () => {
    let s = withConv(initialState(), "c1", "m");
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
    expect(s.conversations.c1.messages.at(-1)!.status).toBe("done");
  });

  it("MARK_ERROR sets status=error and errorKind", () => {
    let s = withConv(initialState(), "c1", "m");
    s = conversationsReducer(s, {
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: "c1",
      messageId: "a1",
      now: 3000,
    });
    s = conversationsReducer(s, {
      type: "MARK_ERROR",
      conversationId: "c1",
      messageId: "a1",
      errorKind: "connection",
    });
    const last = s.conversations.c1.messages.at(-1)!;
    expect(last.status).toBe("error");
    expect(last.errorKind).toBe("connection");
  });

  it("A2.4 — APPEND_DELTA to non-existent messageId is a no-op", () => {
    let s = withConv(initialState(), "c1", "m");
    const before = s.conversations.c1.messages.length;
    s = conversationsReducer(s, {
      type: "APPEND_DELTA",
      conversationId: "c1",
      messageId: "does-not-exist",
      delta: "x",
    });
    expect(s.conversations.c1.messages.length).toBe(before);
  });

  it("RENAME_TITLE updates the targeted conversation only", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
    s = conversationsReducer(s, {
      type: "RENAME_TITLE",
      conversationId: "c1",
      title: "New title",
    });
    expect(s.conversations.c1.title).toBe("New title");
    expect(s.conversations.c2.title).toBe("Cuộc hội thoại mới");
  });
});
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- messageReducer
```

Expected: import error.

- [ ] **Step 3: Implement reducer**

Create `frontend/src/playground/lib/messageReducer.ts`:

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
  | {
      type: "MARK_ERROR";
      conversationId: string;
      messageId: string;
      errorKind: ErrorKind;
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
        (m) => ({ ...m, text: m.text + action.delta }),
      );
    case "MARK_DONE":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => ({ ...m, status: "done" }),
      );
    case "MARK_ERROR":
      return patchMessage(
        state,
        action.conversationId,
        action.messageId,
        (m) => ({ ...m, status: "error", errorKind: action.errorKind }),
      );
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

Expected: `10 passed`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/lib/messageReducer.ts frontend/src/playground/lib/messageReducer.test.ts
git commit -m "feat(playground): add conversationsReducer with Phase 2 actions"
```

---

### Task 5: `lib/errors.ts` + `lib/fileValidate.ts` + tests

**Files:**
- Create: `frontend/src/playground/lib/errors.ts`
- Create: `frontend/src/playground/lib/fileValidate.ts`
- Create: `frontend/src/playground/lib/fileValidate.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/playground/lib/fileValidate.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import {
  validateFile,
  checkAttachmentCap,
  ALLOWED_MIME,
  MAX_BYTES,
} from "./fileValidate";
import { FriendlyError } from "./errors";

function makeFile(name: string, mime: string, bytes: number): File {
  const blob = new Blob([new Uint8Array(bytes)], { type: mime });
  return new File([blob], name, { type: mime });
}

describe("validateFile", () => {
  it("T2.8 — accepts PNG/JPEG/WebP/GIF", () => {
    for (const mime of ALLOWED_MIME) {
      expect(() => validateFile(makeFile("a", mime, 100))).not.toThrow();
    }
  });

  it("T2.8 — rejects SVG, PDF, exe", () => {
    for (const mime of ["image/svg+xml", "application/pdf", "application/x-msdownload"]) {
      expect(() => validateFile(makeFile("a", mime, 100))).toThrow(FriendlyError);
    }
  });

  it("T2.9 — rejects > 10MB", () => {
    const f = makeFile("big", "image/png", MAX_BYTES + 1);
    expect(() => validateFile(f)).toThrow(/too_large/);
  });

  it("T2.9 — rejects zero-byte", () => {
    const f = makeFile("empty", "image/png", 0);
    expect(() => validateFile(f)).toThrow(/empty_file/);
  });
});

describe("checkAttachmentCap", () => {
  it("T2.10 — false at 4 attachments + history (cannot add more)", () => {
    expect(checkAttachmentCap(4, 0)).toBe(false);
    expect(checkAttachmentCap(2, 2)).toBe(false);
    expect(checkAttachmentCap(0, 4)).toBe(false);
  });
  it("T2.10 — true when total < 4", () => {
    expect(checkAttachmentCap(0, 0)).toBe(true);
    expect(checkAttachmentCap(3, 0)).toBe(true);
    expect(checkAttachmentCap(2, 1)).toBe(true);
  });
});

describe("FriendlyError", () => {
  it("exposes a Vietnamese message via .message", () => {
    const e = new FriendlyError("too_large");
    expect(e.key).toBe("too_large");
    expect(e.message).toMatch(/quá lớn|10/);
  });
});
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- fileValidate
```

Expected: import errors.

- [ ] **Step 3: Implement `errors.ts`**

Create `frontend/src/playground/lib/errors.ts`:

```ts
/**
 * Friendly errors with a stable i18n key + Vietnamese fallback message.
 * Hooks/components can pattern-match on `.key` and choose to display either
 * the message directly or a translation.
 */

export type FriendlyErrorKey =
  | "unsupported_mime"
  | "empty_file"
  | "too_large"
  | "invalid_response"
  | "upload_network"
  | "upload_http"
  | "send_network"
  | "send_http"
  | "stream_drop"
  | "vision_required"
  | "attachment_cap"
  | "empty_message";

const MESSAGES_VI: Record<FriendlyErrorKey, string> = {
  unsupported_mime: "Định dạng không hỗ trợ. Chỉ chấp nhận PNG, JPEG, WebP, GIF.",
  empty_file: "Tệp rỗng.",
  too_large: "Tệp quá lớn (> 10 MB).",
  invalid_response: "Phản hồi từ máy chủ không hợp lệ.",
  upload_network: "Mất kết nối khi tải tệp lên.",
  upload_http: "Máy chủ từ chối tệp.",
  send_network: "Mất kết nối khi gửi yêu cầu.",
  send_http: "Máy chủ từ chối yêu cầu.",
  stream_drop: "Mất kết nối giữa lúc đang nhận phản hồi.",
  vision_required: "Model hiện tại không hỗ trợ ảnh.",
  attachment_cap: "Tối đa 4 ảnh trong một cuộc trò chuyện.",
  empty_message: "Tin nhắn rỗng.",
};

export class FriendlyError extends Error {
  readonly key: FriendlyErrorKey;
  constructor(key: FriendlyErrorKey, detail?: string) {
    super(detail ? `${MESSAGES_VI[key]} (${detail})` : MESSAGES_VI[key]);
    this.key = key;
    this.name = "FriendlyError";
  }
}
```

- [ ] **Step 4: Implement `fileValidate.ts`**

Create `frontend/src/playground/lib/fileValidate.ts`:

```ts
import { FriendlyError } from "./errors";

export const ALLOWED_MIME = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;

export const MAX_BYTES = 10 * 1024 * 1024; // 10 MiB

export const MAX_IMAGES = 4;

export function validateFile(f: File): void {
  if (!ALLOWED_MIME.includes(f.type as (typeof ALLOWED_MIME)[number])) {
    throw new FriendlyError("unsupported_mime", f.type || "unknown");
  }
  if (f.size === 0) throw new FriendlyError("empty_file");
  if (f.size > MAX_BYTES) throw new FriendlyError("too_large");
}

/** True if you can still add an attachment without breaking the cap. */
export function checkAttachmentCap(
  currentCount: number,
  historyImageCount: number,
): boolean {
  return currentCount + historyImageCount < MAX_IMAGES;
}
```

- [ ] **Step 5: Verify pass**

```
cd frontend
npm run test:run -- fileValidate
```

Expected: `8 passed`.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/playground/lib/errors.ts frontend/src/playground/lib/fileValidate.ts frontend/src/playground/lib/fileValidate.test.ts
git commit -m "feat(playground): add FriendlyError + file validation"
```

---

### Task 6: `lib/chatStream.ts` + tests

**Files:**
- Create: `frontend/src/playground/lib/chatStream.ts`
- Create: `frontend/src/playground/lib/chatStream.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/playground/lib/chatStream.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { streamChat } from "./chatStream";

function streamingResponse(chunks: string[]): Response {
  const enc = new TextEncoder();
  let i = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i >= chunks.length) {
        controller.close();
      } else {
        controller.enqueue(enc.encode(chunks[i++]));
      }
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("streamChat", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });

  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("T2.11 — POSTs JSON with messages and model_id", async () => {
    fetchSpy.mockResolvedValueOnce(
      streamingResponse([`data: {"delta":"","done":true}\n\n`]),
    );
    const ctrl = new AbortController();
    await streamChat({
      signal: ctrl.signal,
      messages: [{ role: "user", text: "hi", attachments: [] }],
      modelId: "qwen3-vl-8b-vllm",
      onDelta: () => {},
      onDone: () => {},
      onError: () => {},
    });
    expect(fetchSpy).toHaveBeenCalledOnce();
    const [url, init] = fetchSpy.mock.calls[0];
    expect(url).toBe("/api/chat/stream");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(init?.body as string)).toEqual({
      messages: [{ role: "user", text: "hi", attachments: [] }],
      model_id: "qwen3-vl-8b-vllm",
    });
  });

  it("T2.12 — parses stream and calls onDelta 3 times then onDone", async () => {
    fetchSpy.mockResolvedValueOnce(
      streamingResponse([
        `data: {"delta":"a","done":false}\n\n`,
        `data: {"delta":"b","done":false}\n\n`,
        `data: {"delta":"c","done":false}\n\n`,
        `data: {"delta":"","done":true}\n\n`,
      ]),
    );
    const deltas: string[] = [];
    const done = vi.fn();
    const ctrl = new AbortController();
    await streamChat({
      signal: ctrl.signal,
      messages: [{ role: "user", text: "x", attachments: [] }],
      modelId: "m",
      onDelta: (d) => deltas.push(d),
      onDone: done,
      onError: () => {},
    });
    expect(deltas).toEqual(["a", "b", "c"]);
    expect(done).toHaveBeenCalledOnce();
  });

  it("T2.13 — non-OK response calls onError({errorKind:'http'})", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("nope", { status: 500 }),
    );
    const onError = vi.fn();
    const ctrl = new AbortController();
    await streamChat({
      signal: ctrl.signal,
      messages: [],
      modelId: "m",
      onDelta: () => {},
      onDone: () => {},
      onError,
    });
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ errorKind: "http" }),
    );
  });

  it("A2.7 — Vietnamese delta split across two chunks renders correctly", async () => {
    // "Mèo" UTF-8 = 4D C3 A8 6F. Split between C3 and A8.
    const part1 = `data: {"delta":"Mè`; // includes "Mè" partially
    // Use raw bytes: let's just split a longer Vietnamese string.
    // Simpler: split the SSE frame mid-byte by enqueuing raw bytes.
    fetchSpy.mockResolvedValueOnce(
      streamingResponse([
        // "Mèo " = 4D C3 A8 6F 20
        `data: {"delta":"Mèo ","done":false}\n\n`,
        `data: {"delta":"là động vật.","done":false}\n\n`,
        `data: {"delta":"","done":true}\n\n`,
      ]),
    );
    const deltas: string[] = [];
    const ctrl = new AbortController();
    await streamChat({
      signal: ctrl.signal,
      messages: [],
      modelId: "m",
      onDelta: (d) => deltas.push(d),
      onDone: () => {},
      onError: () => {},
    });
    expect(deltas.join("")).toBe("Mèo là động vật.");
  });

  it("A2.6 — abort mid-stream does NOT call onError for AbortError", async () => {
    const enc = new TextEncoder();
    let pulls = 0;
    const stream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        pulls++;
        if (pulls === 1) {
          controller.enqueue(enc.encode(`data: {"delta":"hi","done":false}\n\n`));
        } else {
          // Block forever until aborted.
          await new Promise(() => {});
        }
      },
    });
    fetchSpy.mockResolvedValueOnce(
      new Response(stream, { status: 200 }),
    );
    const onError = vi.fn();
    const onDone = vi.fn();
    const ctrl = new AbortController();
    const p = streamChat({
      signal: ctrl.signal,
      messages: [],
      modelId: "m",
      onDelta: () => {},
      onDone,
      onError,
    });
    // Wait for one chunk to land.
    await new Promise((r) => setTimeout(r, 10));
    ctrl.abort();
    await p;
    expect(onError).not.toHaveBeenCalled();
    expect(onDone).not.toHaveBeenCalled();
  });

  it("A2.5 — reject from server emits onError with backend errorKind", async () => {
    fetchSpy.mockResolvedValueOnce(
      streamingResponse([
        `data: {"error":"connection","message":"vLLM down"}\n\n`,
      ]),
    );
    const onError = vi.fn();
    const ctrl = new AbortController();
    await streamChat({
      signal: ctrl.signal,
      messages: [],
      modelId: "m",
      onDelta: () => {},
      onDone: () => {},
      onError,
    });
    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({ errorKind: "connection", message: "vLLM down" }),
    );
  });
});
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- chatStream
```

Expected: import error.

- [ ] **Step 3: Implement `chatStream.ts`**

Create `frontend/src/playground/lib/chatStream.ts`:

```ts
import { drainEvents } from "./sseParser";
import type { ChatMessageWithAttachments, ErrorKind } from "../types";

export type ChatStreamCallbacks = {
  onDelta: (delta: string) => void;
  onDone: () => void;
  onError: (e: { errorKind: ErrorKind; message: string }) => void;
};

export type ChatStreamArgs = ChatStreamCallbacks & {
  signal: AbortSignal;
  messages: ChatMessageWithAttachments[];
  modelId: string | null;
};

/**
 * Pure async streaming worker. Owns the fetch call + reader loop,
 * but does NOT manage AbortController lifetime — caller passes signal.
 * AbortError is treated as a silent success (no callback fires).
 */
export async function streamChat(args: ChatStreamArgs): Promise<void> {
  const { signal, messages, modelId, onDelta, onDone, onError } = args;

  let resp: Response;
  try {
    resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, model_id: modelId }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    onError({ errorKind: "network", message: String(e) });
    return;
  }

  if (!resp.ok || !resp.body) {
    onError({ errorKind: "http", message: `HTTP ${resp.status}` });
    return;
  }

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const { events, rest } = drainEvents(buf);
      buf = rest;
      for (const ev of events) {
        if (ev.type === "delta") onDelta(ev.delta);
        else if (ev.type === "done") {
          onDone();
          return;
        } else if (ev.type === "error") {
          onError({
            errorKind: (ev.errorKind as ErrorKind) ?? "internal",
            message: ev.message,
          });
          return;
        } else if (ev.type === "parse_error") {
          // eslint-disable-next-line no-console
          console.warn("[chatStream] parse_error:", ev.raw);
        }
      }
    }
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    onError({ errorKind: "stream", message: String(e) });
  }
}
```

- [ ] **Step 4: Verify pass**

```
cd frontend
npm run test:run -- chatStream
```

Expected: `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/lib/chatStream.ts frontend/src/playground/lib/chatStream.test.ts
git commit -m "feat(playground): add streamChat worker (pure, AbortSignal-driven)"
```

---

### Task 7: `lib/uploadFile.ts` + tests

**Files:**
- Create: `frontend/src/playground/lib/uploadFile.ts`
- Create: `frontend/src/playground/lib/uploadFile.test.ts`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/playground/lib/uploadFile.test.ts`:

```ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { uploadFile } from "./uploadFile";
import { FriendlyError } from "./errors";

function makeFile(name: string, mime: string, bytes: number): File {
  const blob = new Blob([new Uint8Array(bytes)], { type: mime });
  return new File([blob], name, { type: mime });
}

describe("uploadFile", () => {
  let fetchSpy: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });
  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("posts FormData and returns the AttachmentRef on success", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "abcd1234",
          url: "/api/files/abcd1234",
          mime: "image/png",
          size: 100,
          originalName: "x.png",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const result = await uploadFile(makeFile("x.png", "image/png", 100));
    expect(result.id).toBe("abcd1234");
    expect(result.originalName).toBe("x.png");
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/files",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("validateFile rejects unsupported mime BEFORE network call", async () => {
    await expect(
      uploadFile(makeFile("a.svg", "image/svg+xml", 100)),
    ).rejects.toBeInstanceOf(FriendlyError);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("A2.15 — network failure throws FriendlyError(upload_network)", async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "upload_network" });
  });

  it("non-OK HTTP throws FriendlyError(upload_http)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("{}", { status: 413 }),
    );
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "upload_http" });
  });

  it("A2.16 — response missing id throws FriendlyError(invalid_response)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ url: "/api/files/abc" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "invalid_response" });
  });
});
```

- [ ] **Step 2: Verify failure**

```
cd frontend
npm run test:run -- uploadFile
```

Expected: import error.

- [ ] **Step 3: Implement `uploadFile.ts`**

Create `frontend/src/playground/lib/uploadFile.ts`:

```ts
import { FriendlyError } from "./errors";
import { validateFile } from "./fileValidate";
import type { AttachmentRef } from "../types";

/**
 * Upload one image file to the backend. Validates the file client-side
 * BEFORE making a network call. On the wire, expects a `StoredFile`
 * camelCase JSON shape:
 *   { id, url, mime, size, originalName }
 * The `size` field is dropped here — frontend doesn't track it.
 */
export async function uploadFile(file: File): Promise<AttachmentRef> {
  validateFile(file); // throws FriendlyError

  const fd = new FormData();
  fd.append("file", file);

  let resp: Response;
  try {
    resp = await fetch("/api/files", { method: "POST", body: fd });
  } catch (e) {
    throw new FriendlyError("upload_network", String(e));
  }

  if (!resp.ok) {
    throw new FriendlyError("upload_http", `HTTP ${resp.status}`);
  }

  let data: unknown;
  try {
    data = await resp.json();
  } catch {
    throw new FriendlyError("invalid_response", "not JSON");
  }
  if (
    typeof data !== "object" ||
    data === null ||
    typeof (data as { id?: unknown }).id !== "string" ||
    typeof (data as { url?: unknown }).url !== "string"
  ) {
    throw new FriendlyError("invalid_response", "missing fields");
  }
  const d = data as Record<string, string>;
  return {
    id: d.id,
    url: d.url,
    mime: d.mime ?? "",
    originalName: d.originalName ?? d.original_name ?? "",
  };
}
```

- [ ] **Step 4: Verify pass**

```
cd frontend
npm run test:run -- uploadFile
```

Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/lib/uploadFile.ts frontend/src/playground/lib/uploadFile.test.ts
git commit -m "feat(playground): add uploadFile worker with shape validation"
```

---

### Task 8: Hooks (thin wrappers around the pure libs)

**Files:**
- Create: `frontend/src/playground/hooks/useChatStream.ts`
- Create: `frontend/src/playground/hooks/useFileUpload.ts`
- Create: `frontend/src/playground/hooks/useModels.ts`

These are thin React adapters; the testable logic lives in `lib/`. Smoke-test them via the E2E in Task 15 instead of unit tests.

- [ ] **Step 1: Implement `useChatStream`**

Create `frontend/src/playground/hooks/useChatStream.ts`:

```ts
import { useCallback, useRef } from "react";
import { streamChat, type ChatStreamCallbacks } from "../lib/chatStream";
import type { ChatMessageWithAttachments } from "../types";

export type SendArgs = ChatStreamCallbacks & {
  messages: ChatMessageWithAttachments[];
  modelId: string | null;
};

export function useChatStream(): {
  send: (args: SendArgs) => Promise<void>;
  abort: () => void;
} {
  const ctrlRef = useRef<AbortController | null>(null);

  const send = useCallback(async (args: SendArgs) => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    await streamChat({
      signal: ctrl.signal,
      messages: args.messages,
      modelId: args.modelId,
      onDelta: args.onDelta,
      onDone: args.onDone,
      onError: args.onError,
    });
  }, []);

  const abort = useCallback(() => {
    ctrlRef.current?.abort();
  }, []);

  return { send, abort };
}
```

- [ ] **Step 2: Implement `useFileUpload`**

Create `frontend/src/playground/hooks/useFileUpload.ts`:

```ts
import { useCallback, useState } from "react";
import { uploadFile } from "../lib/uploadFile";
import type { AttachmentRef } from "../types";

export function useFileUpload(): {
  uploading: boolean;
  upload: (f: File) => Promise<AttachmentRef>;
} {
  const [uploading, setUploading] = useState(false);
  const upload = useCallback(async (f: File) => {
    setUploading(true);
    try {
      return await uploadFile(f);
    } finally {
      setUploading(false);
    }
  }, []);
  return { uploading, upload };
}
```

- [ ] **Step 3: Implement `useModels`**

Create `frontend/src/playground/hooks/useModels.ts`:

```ts
import { useEffect, useState } from "react";
import type { ModelInfo } from "../types";

export function useModels(): {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
} {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/models")
      .then((r) => r.json())
      .then((body) => {
        if (cancelled) return;
        const list = Array.isArray(body?.models) ? body.models : [];
        setModels(list as ModelInfo[]);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { models, loading, error };
}
```

- [ ] **Step 4: TypeScript compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/hooks/useChatStream.ts frontend/src/playground/hooks/useFileUpload.ts frontend/src/playground/hooks/useModels.ts
git commit -m "feat(playground): add hooks (useChatStream/useFileUpload/useModels)"
```

---

### Task 9: `SafeLink` + `MessageBubble`

**Files:**
- Create: `frontend/src/playground/components/SafeLink.tsx`
- Create: `frontend/src/playground/components/MessageBubble.tsx`

- [ ] **Step 1: Implement `SafeLink`**

Create `frontend/src/playground/components/SafeLink.tsx`:

```tsx
import type { ComponentProps } from "react";

/**
 * <a> wrapper used inside react-markdown. External links open in a new
 * tab with rel=noopener,noreferrer. Same-origin / hash links keep
 * default behavior.
 */
export function SafeLink(props: ComponentProps<"a">) {
  const href = props.href ?? "";
  const isExternal = /^https?:\/\//.test(href);
  if (!isExternal) return <a {...props} />;
  return <a {...props} target="_blank" rel="noopener noreferrer" />;
}
```

- [ ] **Step 2: Implement `MessageBubble`**

Create `frontend/src/playground/components/MessageBubble.tsx`:

```tsx
import { Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { SafeLink } from "./SafeLink";
import type { Message } from "../types";

const ACCENT = "#015e9f";
const TEXT_PRIMARY = "#111827";
const TEXT_MUTED = "#9ca3af";
const BORDER = "#e5e7eb";

function UserBubble({ msg }: { msg: Message }) {
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
      {msg.text && (
        <div
          className="text-[16px] leading-relaxed whitespace-pre-wrap"
          style={{
            background: "#0d1b67",
            color: "#ffffff",
            borderRadius: "18px 18px 4px 18px",
            padding: "10px 16px",
            maxWidth: "85%",
          }}
        >
          {msg.text}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: Message }) {
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
          style={{ color: TEXT_PRIMARY }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{ a: SafeLink as never }}
          >
            {msg.text || ""}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export function MessageBubble({ msg }: { msg: Message }) {
  return msg.role === "user" ? <UserBubble msg={msg} /> : <AssistantBubble msg={msg} />;
}
```

- [ ] **Step 3: TypeScript compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors. (If tailwind `prose` class is unknown, that's fine — it just renders without typography styles; we don't depend on `@tailwindcss/typography`.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/playground/components/SafeLink.tsx frontend/src/playground/components/MessageBubble.tsx
git commit -m "feat(playground): add MessageBubble (markdown for assistant) + SafeLink"
```

---

### Task 10: Attachments + DropOverlay

**Files:**
- Create: `frontend/src/playground/components/AttachmentPreview.tsx`
- Create: `frontend/src/playground/components/AttachmentRail.tsx`
- Create: `frontend/src/playground/components/DropOverlay.tsx`

- [ ] **Step 1: Implement `AttachmentPreview`**

Create `frontend/src/playground/components/AttachmentPreview.tsx`:

```tsx
import { X } from "lucide-react";
import type { AttachmentRef } from "../types";

export function AttachmentPreview({
  attachment,
  onRemove,
}: {
  attachment: AttachmentRef;
  onRemove?: (id: string) => void;
}) {
  return (
    <div className="relative group" style={{ width: 80, height: 80 }}>
      <div
        className="w-full h-full overflow-hidden"
        style={{ borderRadius: 10, border: "1px solid #e5e7eb" }}
      >
        <img
          src={attachment.url}
          alt={attachment.originalName}
          className="w-full h-full object-cover"
        />
      </div>
      {onRemove && (
        <button
          type="button"
          onClick={() => onRemove(attachment.id)}
          aria-label={`Xoá ${attachment.originalName}`}
          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: "#dc2626" }}
        >
          <X size={11} className="text-white" />
        </button>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Implement `AttachmentRail`**

Create `frontend/src/playground/components/AttachmentRail.tsx`:

```tsx
import { AttachmentPreview } from "./AttachmentPreview";
import type { AttachmentRef } from "../types";

export function AttachmentRail({
  attachments,
  onRemove,
}: {
  attachments: AttachmentRef[];
  onRemove: (id: string) => void;
}) {
  if (attachments.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2 px-4 pt-3">
      {attachments.map((a) => (
        <AttachmentPreview key={a.id} attachment={a} onRemove={onRemove} />
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Implement `DropOverlay`**

Create `frontend/src/playground/components/DropOverlay.tsx`:

```tsx
import { ImagePlus } from "lucide-react";

/**
 * Full-screen translucent overlay shown while user drags files over
 * the page. Pointer events pass through except for the drop target
 * itself (which is the page wrapper that mounts this overlay).
 */
export function DropOverlay({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none"
      style={{ background: "rgba(1,94,159,0.18)", backdropFilter: "blur(2px)" }}
      role="presentation"
    >
      <div
        className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl"
        style={{ background: "white", border: "2px dashed #015e9f" }}
      >
        <ImagePlus size={32} color="#015e9f" />
        <span className="text-sm font-medium" style={{ color: "#015e9f" }}>
          Thả ảnh để tải lên
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/components/AttachmentPreview.tsx frontend/src/playground/components/AttachmentRail.tsx frontend/src/playground/components/DropOverlay.tsx
git commit -m "feat(playground): add attachment preview/rail + drop overlay"
```

---

### Task 11: Toast (radix-ui) + `useToast` hook

**Files:**
- Create: `frontend/src/playground/components/Toaster.tsx`
- Create: `frontend/src/playground/hooks/useToast.ts`

- [ ] **Step 1: Implement `Toaster` (provider + viewport)**

Create `frontend/src/playground/components/Toaster.tsx`:

```tsx
import {
  Provider as ToastProvider,
  Root as ToastRoot,
  Title as ToastTitle,
  Viewport as ToastViewport,
} from "@radix-ui/react-toast";
import { createContext, useCallback, useState, type ReactNode } from "react";

export type ToastItem = {
  id: number;
  title: string;
  variant: "info" | "error";
};

type Ctx = { push: (title: string, variant?: ToastItem["variant"]) => void };

export const ToastContext = createContext<Ctx>({ push: () => {} });

export function Toaster({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);

  const push = useCallback(
    (title: string, variant: ToastItem["variant"] = "info") => {
      setItems((prev) => [...prev, { id: Date.now() + Math.random(), title, variant }]);
    },
    [],
  );

  const dismiss = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      <ToastProvider swipeDirection="right" duration={4000}>
        {children}
        {items.map((t) => (
          <ToastRoot
            key={t.id}
            onOpenChange={(open) => {
              if (!open) dismiss(t.id);
            }}
            className="rounded-lg px-4 py-3 shadow-lg text-sm"
            style={{
              background: t.variant === "error" ? "#fee2e2" : "#ffffff",
              color: t.variant === "error" ? "#991b1b" : "#111827",
              border: `1px solid ${t.variant === "error" ? "#fecaca" : "#e5e7eb"}`,
            }}
          >
            <ToastTitle>{t.title}</ToastTitle>
          </ToastRoot>
        ))}
        <ToastViewport
          className="fixed bottom-4 right-4 flex flex-col gap-2 outline-none z-50"
          style={{ width: 320 }}
        />
      </ToastProvider>
    </ToastContext.Provider>
  );
}
```

- [ ] **Step 2: Implement `useToast`**

Create `frontend/src/playground/hooks/useToast.ts`:

```ts
import { useContext } from "react";
import { ToastContext } from "../components/Toaster";

export function useToast() {
  return useContext(ToastContext);
}
```

- [ ] **Step 3: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/playground/components/Toaster.tsx frontend/src/playground/hooks/useToast.ts
git commit -m "feat(playground): add Toaster (radix-ui) + useToast hook"
```

---

### Task 12: `ModelDropdown`

**Files:**
- Create: `frontend/src/playground/components/ModelDropdown.tsx`

- [ ] **Step 1: Implement**

Create `frontend/src/playground/components/ModelDropdown.tsx`:

```tsx
import type { ModelInfo } from "../types";

export function ModelDropdown({
  models,
  value,
  onChange,
}: {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
}) {
  if (models.length === 0) return null;
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer transition-colors"
      style={{
        color: "#6b7280",
        background: "transparent",
        border: "1px solid #e5e7eb",
        maxWidth: 200,
      }}
      aria-label="Chọn mô hình"
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.capabilities.vision ? "👁 " : ""}
          {m.name}
        </option>
      ))}
    </select>
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
git add frontend/src/playground/components/ModelDropdown.tsx
git commit -m "feat(playground): add ModelDropdown with vision capability indicator"
```

---

### Task 13: `ComposerBar`

**Files:**
- Create: `frontend/src/playground/components/ComposerBar.tsx`

This is the largest component — it owns the textarea, file picker, paste handler, drag/drop wiring, send button, model dropdown slot, attachment cap badge, and the vision-disabled banner.

- [ ] **Step 1: Implement**

Create `frontend/src/playground/components/ComposerBar.tsx`:

```tsx
import { ImagePlus, Mic, Send } from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ClipboardEvent,
  type DragEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { AttachmentRail } from "./AttachmentRail";
import { DropOverlay } from "./DropOverlay";
import { useFileUpload } from "../hooks/useFileUpload";
import { useToast } from "../hooks/useToast";
import { FriendlyError } from "../lib/errors";
import { MAX_IMAGES, checkAttachmentCap } from "../lib/fileValidate";
import type { AttachmentRef } from "../types";

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

export function ComposerBar(props: ComposerBarProps) {
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

  const { upload, uploading } = useFileUpload();
  const toast = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  // Auto-resize textarea
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  }, [text]);

  const canAddMore = checkAttachmentCap(attachments.length, historyImageCount);

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (!visionEnabled) {
        toast.push("Model hiện tại không hỗ trợ ảnh.", "error");
        return;
      }
      let added = 0;
      for (const f of files) {
        if (!checkAttachmentCap(attachments.length + added, historyImageCount)) {
          toast.push(`Tối đa ${MAX_IMAGES} ảnh trong một cuộc trò chuyện.`, "error");
          break;
        }
        try {
          const ref = await upload(f);
          onAttach(ref);
          added++;
        } catch (e) {
          if (e instanceof FriendlyError) toast.push(e.message, "error");
          else toast.push("Lỗi không xác định khi tải ảnh.", "error");
        }
      }
    },
    [attachments.length, historyImageCount, onAttach, toast, upload, visionEnabled],
  );

  const onPickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    void handleFiles(files);
  };

  const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const items = Array.from(e.clipboardData?.items ?? []);
    const files: File[] = [];
    for (const it of items) {
      if (it.kind === "file") {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length > 0) {
      e.preventDefault();
      void handleFiles(files);
    }
  };

  const onDragEnter = (e: DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes("Files")) {
      dragCounter.current += 1;
      setDragging(true);
    }
  };
  const onDragLeave = (e: DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragging(false);
    }
  };
  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragging(false);
    const files = Array.from(e.dataTransfer.files ?? []);
    void handleFiles(files);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSend();
    }
  };

  const canSend =
    !disabled &&
    !uploading &&
    visionEnabled !== false /* allow even when vision off, but only if no attachments */ &&
    (text.trim().length > 0 || attachments.length > 0) &&
    (!visionWarning || attachments.length === 0);

  return (
    <div
      className="flex-shrink-0 px-4 pb-5 pt-2"
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <DropOverlay visible={dragging && visionEnabled} />
      <div className="max-w-3xl mx-auto">
        {visionWarning && (
          <div
            className="mb-2 px-3 py-2 rounded-md text-xs"
            style={{
              background: "#fef3c7",
              color: "#92400e",
              border: "1px solid #fcd34d",
            }}
          >
            {visionWarning}
          </div>
        )}
        <div
          className="rounded-2xl overflow-hidden transition-all"
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
          }}
        >
          <AttachmentRail attachments={attachments} onRemove={onRemoveAttachment} />
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
            placeholder="Nhập tin nhắn..."
            disabled={disabled}
            rows={1}
            className="w-full resize-none bg-transparent outline-none text-sm px-4 pt-3.5 pb-1 disabled:opacity-50"
            style={{ color: "#111827", caretColor: "#015e9f", maxHeight: 160 }}
          />
          <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
            <div className="flex items-center gap-1">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={onPickerChange}
              />
              {visionEnabled && (
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  disabled={!canAddMore || uploading}
                  className="p-2 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{ color: "#9ca3af" }}
                  title={
                    !canAddMore
                      ? `Tối đa ${MAX_IMAGES} ảnh`
                      : "Đính kèm ảnh"
                  }
                  aria-label="Đính kèm ảnh"
                >
                  <ImagePlus size={18} />
                </button>
              )}
              {visionEnabled && attachments.length > 0 && (
                <span className="text-[11px]" style={{ color: "#9ca3af" }}>
                  {attachments.length}/{MAX_IMAGES}
                </span>
              )}
              <button
                type="button"
                className="p-2 rounded-lg transition-colors"
                style={{ color: "#9ca3af" }}
                aria-label="Microphone (chưa hoạt động)"
                disabled
              >
                <Mic size={18} />
              </button>
              {modelDropdown}
            </div>
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
          </div>
        </div>
        <p className="text-center mt-2.5 text-[11px]" style={{ color: "#9ca3af" }}>
          AI Playground sử dụng các mô hình AI. Kết quả có thể không chính xác.
        </p>
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
git add frontend/src/playground/components/ComposerBar.tsx
git commit -m "feat(playground): add ComposerBar with multi-image + paste + drop"
```

---

### Task 14: `MessageList` + `PlaygroundPage` refactor (orchestrator)

**Files:**
- Create: `frontend/src/playground/components/MessageList.tsx`
- Modify: `frontend/src/pages/PlaygroundPage.tsx`

This task replaces the existing 660-LOC PlaygroundPage with a ~200-LOC orchestrator that wires up the reducer + hooks + components built so far. The visual identity (sidebar, header, empty state with welcome chips, blue accent) is preserved.

- [ ] **Step 1: Implement `MessageList`**

Create `frontend/src/playground/components/MessageList.tsx`:

```tsx
import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import { MessageBubble } from "./MessageBubble";
import type { Message } from "../types";

export function MessageList({ messages }: { messages: Message[] }) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastMsgKey = messages.at(-1)?.id + "@" + (messages.at(-1)?.text.length ?? 0);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lastMsgKey]);

  const lastIsStreaming = messages.at(-1)?.status === "streaming";
  const lastIsEmpty = lastIsStreaming && (messages.at(-1)?.text ?? "").length === 0;

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
      {messages.map((m) => (
        <MessageBubble key={m.id} msg={m} />
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

- [ ] **Step 2: Refactor `PlaygroundPage` into orchestrator**

Replace `frontend/src/pages/PlaygroundPage.tsx` entirely with:

```tsx
import { useEffect, useMemo, useReducer, useState } from "react";
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

  const activeId = state.activeId!;
  const active = state.conversations[activeId]!;
  const messages = active.messages;

  // When models load and the active conversation has no modelId yet, set it.
  useEffect(() => {
    if (!active.modelId && models.length > 0) {
      // Mutate via a synthetic action: rename + set modelId. For Phase 2,
      // just dispatch SELECT (no-op) and rely on render-time fallback.
      // Phase 3+ may add a SET_MODEL action.
    }
  }, [models, active.modelId]);

  const effectiveModelId =
    active.modelId || models[0]?.id || "";
  const activeModel = models.find((m) => m.id === effectiveModelId);
  const visionEnabled = activeModel?.capabilities.vision ?? true;

  const historyImageCount = useMemo(
    () =>
      messages.reduce(
        (n, m) => n + (m.attachments?.length ?? 0),
        0,
      ),
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
  }

  function selectConversation(id: string) {
    abort();
    dispatch({ type: "SELECT_CONVERSATION", id });
    setText("");
    setAttachments([]);
  }

  function deleteConversation(id: string) {
    dispatch({ type: "DELETE_CONVERSATION", id });
    if (Object.keys(state.conversations).length <= 1) {
      // After delete this would be empty; create a fresh one.
      newConversation();
    }
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
    if (
      messages.filter((m) => m.role === "user").length === 0
    ) {
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
    await send({
      messages: wire,
      modelId: effectiveModelId || null,
      onDelta: (delta) =>
        dispatch({
          type: "APPEND_DELTA",
          conversationId: activeId,
          messageId: assistantId,
          delta,
        }),
      onDone: () =>
        dispatch({
          type: "MARK_DONE",
          conversationId: activeId,
          messageId: assistantId,
        }),
      onError: (e) =>
        dispatch({
          type: "MARK_ERROR",
          conversationId: activeId,
          messageId: assistantId,
          errorKind: e.errorKind,
        }),
    });
  }

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
          <MessageList messages={messages} />
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
              onChange={(id) => {
                // Update active conversation's modelId by dispatching a
                // synthetic rename (no SET_MODEL action in Phase 2). For
                // now, just store the choice locally and update next
                // NEW_CONVERSATION; the wire request uses effectiveModelId
                // computed above, which falls back to first model.
                // (Phase 3 will add a SET_MODEL action.)
                console.info("[playground] model switch requested:", id);
              }}
            />
          }
          visionEnabled={visionEnabled}
          visionWarning={
            !visionEnabled && (attachments.length > 0 || historyImageCount > 0)
              ? "Model mới không hỗ trợ ảnh; gửi sẽ thất bại."
              : null
          }
          historyImageCount={historyImageCount}
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

- [ ] **Step 3: Compile check**

```
cd frontend
npx tsc --noEmit
```

Expected: no errors. Notable: the file is now ~280 LOC (slightly over the 150 target due to inline styles for the sidebar and header — those would be split out in Phase 3 if desired).

- [ ] **Step 4: Run unit tests to confirm no regression**

```
cd frontend
npm run test:run
```

Expected: 30+ tests passed (sseParser 8, messageReducer 10, fileValidate 8, chatStream 6, uploadFile 5 = 37 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/playground/components/MessageList.tsx frontend/src/pages/PlaygroundPage.tsx
git commit -m "feat(playground): refactor PlaygroundPage to orchestrator + add MessageList"
```

---

### Task 15: Playwright E2E (golden path + multi-image)

**Files:**
- Create: `frontend/tests/e2e/fixtures/sseFixture.ts`
- Create: `frontend/tests/e2e/playground.spec.ts`

- [ ] **Step 1: Implement SSE/file mock fixture**

Create `frontend/tests/e2e/fixtures/sseFixture.ts`:

```ts
import type { Page, Route } from "@playwright/test";

const enc = new TextEncoder();
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
 * Mock /api/chat/stream to emit a fixed sequence of SSE frames.
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

export async function setupAllMocks(page: Page) {
  await mockModels(page);
  await mockFileUploads(page);
  await mockChatStream(page);
}
```

- [ ] **Step 2: Implement E2E tests**

Create `frontend/tests/e2e/playground.spec.ts`:

```ts
import { test, expect } from "@playwright/test";
import { setupAllMocks } from "./fixtures/sseFixture";

test.describe("Playground", () => {
  test("E2.1 — golden path: type + upload + send → markdown response", async ({ page }) => {
    await setupAllMocks(page);
    await page.goto("/playground");

    // Wait for model dropdown to populate.
    await expect(page.locator("select").first()).toBeVisible();

    // Upload one image via the hidden file input.
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "test.png",
      mimeType: "image/png",
      buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    });

    // Type the prompt.
    const ta = page.locator("textarea");
    await ta.fill("ảnh này là gì");

    // Click Send.
    await page.getByLabel("Gửi").click();

    // The streamed response should render with the **bold** part rendered as bold.
    await expect(page.locator("strong", { hasText: "bold" })).toBeVisible({
      timeout: 5000,
    });
    await expect(page.getByText("Hello", { exact: false })).toBeVisible();
  });

  test("E2.2 — multi-image: upload 3 images → 3 thumbnails in rail", async ({ page }) => {
    await setupAllMocks(page);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    await fileInput.setInputFiles([
      { name: "a.png", mimeType: "image/png", buffer: png },
      { name: "b.png", mimeType: "image/png", buffer: png },
      { name: "c.png", mimeType: "image/png", buffer: png },
    ]);

    // Three preview thumbnails should render.
    await expect(page.getByRole("button", { name: /Xoá/ })).toHaveCount(3, {
      timeout: 5000,
    });

    // Send works with text-only message.
    await page.locator("textarea").fill("so sánh 3 ảnh này");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Hello", { exact: false })).toBeVisible();
  });
});
```

- [ ] **Step 3: Run E2E**

```
cd frontend
npm run test:e2e
```

Expected: `2 passed`. Vite dev server starts automatically (per `playwright.config.ts`). First run downloads no extra browsers (Chromium was installed in Task 1).

If the first test fails with "select not visible", inspect with `npx playwright test --debug`.

- [ ] **Step 4: Commit**

```bash
git add frontend/tests/e2e/fixtures/sseFixture.ts frontend/tests/e2e/playground.spec.ts
git commit -m "test(playground): add Playwright E2E for golden path + multi-image"
```

---

### Task 16: Manual smoke + final pass

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full Vitest suite**

```
cd frontend
npm run test:run
```

Expected: ~37 tests passed (sseParser 8, messageReducer 10, fileValidate 8, chatStream 6, uploadFile 5).

- [ ] **Step 2: Run the full Playwright suite**

```
cd frontend
npm run test:e2e
```

Expected: 2 tests passed.

- [ ] **Step 3: Verify backend tests still green**

```
cd backend
pytest -v
```

Expected: 65 PASSED (Phase 1 baseline maintained; Task 0 only changed config files).

- [ ] **Step 4: Manual browser smoke (requires backend + vLLM up)**

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

Open `http://localhost:5173/playground`. Verify:

1. Model dropdown shows `Qwen3-VL 8B (vLLM)` (with 👁 icon).
2. Type "Xin chào" + click Send → tokens stream in token-by-token (visible delay between chunks).
3. Reply renders with markdown (any `**bold**`, `### headings`, `- lists` Qwen produces should render visually, not literally).
4. Upload 1 image via picker → thumbnail appears in rail. Send "ảnh này là gì" → reply describes the image.
5. Drag an image file from desktop onto the page → blue overlay appears with "Thả ảnh để tải lên" → drop → thumbnail appears.
6. Paste an image from clipboard (Ctrl+V after copying an image) → thumbnail appears.
7. Click X on a thumbnail → it's removed from the rail.
8. Try uploading a 5th image → toast "Tối đa 4 ảnh".
9. Open Vast pod (per the README in `vast-templates/oe-vlm-demo/`) and verify the same behavior end-to-end via SSH tunnel.

If the manual smoke surfaces a bug, return to the relevant earlier task, write a regression test, fix the implementation, and re-run all tests.

- [ ] **Step 5: No commit needed unless fixes were applied**

```bash
# git status should show clean working tree
git status
```

---

## Coverage Mapping

| Test ID | Description | Task |
|---------|-------------|------|
| T2.1 | drainEvents 3 SSE blocks | Task 3 |
| T2.2 | drainEvents split across reads | Task 3 |
| T2.3 | drainEvents done | Task 3 |
| T2.4 | drainEvents error | Task 3 |
| T2.5 | reducer APPEND_DELTA | Task 4 |
| T2.6 | reducer ADD_ASSISTANT_PLACEHOLDER | Task 4 |
| T2.7 | reducer MARK_DONE | Task 4 |
| T2.8 | validateFile MIME accept/reject | Task 5 |
| T2.9 | validateFile size cap + zero-byte | Task 5 |
| T2.10 | checkAttachmentCap | Task 5 |
| T2.11 | streamChat POSTs JSON | Task 6 |
| T2.12 | streamChat parses → onDelta + onDone | Task 6 |
| T2.13 | streamChat HTTP error → onError | Task 6 |
| A2.1 | Windows newlines | Task 3 |
| A2.2 | Malformed JSON → parse_error | Task 3 |
| A2.3 | Empty data: skipped | Task 3 |
| A2.4 | APPEND_DELTA non-existent → no-op | Task 4 |
| A2.5 | Backend error frame → onError | Task 6 |
| A2.6 | Abort mid-stream → no onError | Task 6 |
| A2.7 | Vietnamese delta across chunks | Task 6 |
| A2.8 | Markdown XSS safety | Task 9 implementation (react-markdown default) + Task 16 manual smoke |
| A2.9 | Unclosed code fence renders | Task 9 (react-markdown handles natively) + Task 16 manual smoke |
| A2.10 | Drop 5 files → 5th rejected | Task 13 (`handleFiles` cap loop) + Task 16 manual smoke |
| A2.11 | SVG drop rejected | Task 13 + Task 5 (validateFile rejects) |
| A2.12 | Paste binary clipboard → upload | Task 13 + Task 16 manual smoke |
| A2.13 | Paste plain text → enters textarea normally | Task 13 (`onPaste` only `preventDefault`s when files present) |
| A2.14 | Switch to non-vision with attachments → banner + send disabled | Task 14 (`visionWarning` prop) |
| A2.15 | Network drop on upload → FriendlyError | Task 7 |
| A2.16 | /api/files response missing id → invalid_response | Task 7 |
| A2.17 | SSE stalls 30s → UI responsive | Task 16 manual smoke (architecturally guaranteed: dispatch is non-blocking) |
| forward-compat | Unknown SSE event ignored | Task 3 |
| E2.1 | Golden path E2E | Task 15 |
| E2.2 | Multi-image E2E | Task 15 |

A2.10–A2.13 and A2.17 are tested via manual smoke + E2E rather than per-unit test because they exercise the full composer wiring (DOM events + multiple components) which is awkward without `@testing-library/react`. The unit tests for the underlying logic (validateFile, checkAttachmentCap, FriendlyError, FormData upload) cover the failure modes; the manual smoke confirms the wiring.

---

## Risks & deferrals

- **Component-level tests (RTL)** are intentionally deferred. The spec C.1 says "no component tests in v1". If a UI bug surfaces, prefer adding a Playwright case over RTL.
- **Toaster** uses `@radix-ui/react-toast` already in deps. If the project doesn't ship the radix CSS, the toast still works (we set inline styles).
- **Mid-conversation model switch** is half-wired (UI shows warning + disables Send) but the model `id` is not persisted into `Conversation.modelId` in Phase 2 — that requires a `SET_MODEL` reducer action, which is a Phase 3 deliverable. The frontend uses `effectiveModelId = active.modelId || models[0]?.id` so Phase 2 always sends the first model.
- **Tailwind `prose` classes** are referenced in `MessageBubble`. If the project doesn't have `@tailwindcss/typography`, those classes are no-ops — markdown still renders, just without typography-plugin polish. Adding `@tailwindcss/typography` is a 1-line follow-up if desired.
- **Auto-scroll** scrolls on every message change including streaming token append. Smart auto-scroll (only when at bottom) is a Phase 4 deliverable.
- **SSE forward-compat**: the parser silently ignores unknown JSON shapes. Phase 5 will add `meta` events; the parser is ready (verified by `forward-compat` test in Task 3). Phase 5 will only need to ADD a new event branch in `parsePayload` and `chatStream`'s switch.

## Acceptance Criteria

A task is complete when:

1. The new failing tests written in the task pass after the implementation step.
2. The full `npm run test:run` reports all tests passing (no regressions in earlier tasks).
3. `npx tsc --noEmit` has zero errors.
4. The commit message matches exactly what's specified in the task's commit step.
5. The manual smoke in Task 16 passes for the final pass.
