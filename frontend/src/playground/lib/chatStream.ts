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

  // Abort promise: resolves when the signal fires, letting us break out
  // of reader.read() which does not natively respect AbortSignal.
  const abortPromise = new Promise<never>((_, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
    } else {
      signal.addEventListener(
        "abort",
        () => reject(new DOMException("Aborted", "AbortError")),
        { once: true },
      );
    }
  });

  try {
    while (true) {
      const { done, value } = await Promise.race([reader.read(), abortPromise]);
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
  } finally {
    reader.releaseLock();
  }
}
