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
