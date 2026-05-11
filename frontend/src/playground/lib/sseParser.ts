/**
 * Pure SSE parser for the /api/chat/stream wire format.
 *
 * Backend emits one JSON payload per `data:` frame, separated by `\n\n`:
 *   {"delta":"...","done":false}                      normal token
 *   {"delta":"","done":true}                          end of stream
 *   {"error":"...","message":"..."}                   terminal error
 *   {"type":"status","message":"...","done":bool}     ephemeral status (Phase 5)
 *
 * Forward-compat: unknown JSON shapes (e.g. future Phase 6 meta events) are
 * silently ignored — drainEvents emits nothing for them.
 */

export type SseEvent =
  | { type: "delta"; delta: string }
  | { type: "done" }
  | { type: "error"; errorKind: string; message: string }
  | { type: "status"; message: string; statusDone: boolean }
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

  // Status events are checked first — they share the wire with delta/done/error
  // but use a `type` discriminator. Forward-compat: any future `type:` other
  // than "status" is ignored.
  if (obj.type === "status" && typeof obj.message === "string") {
    return {
      type: "status",
      message: obj.message,
      statusDone: obj.done === true,
    };
  }
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
