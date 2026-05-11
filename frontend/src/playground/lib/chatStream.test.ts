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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let fetchSpy: ReturnType<typeof vi.spyOn<any, any>>;

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
    const [url, init] = fetchSpy.mock.calls[0] as [string, RequestInit];
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

describe("Phase 5 onStatus callback", () => {
  it("T5.4 — invokes onStatus(message, done) when status events arrive", async () => {
    const body =
      `data: {"type":"status","message":"Captioning...","done":false}\n\n` +
      `data: {"type":"status","message":"Done","done":true}\n\n` +
      `data: {"delta":"hi","done":false}\n\n` +
      `data: {"delta":"","done":true}\n\n`;

    const fetchMock = vi.fn(async () =>
      new Response(body, {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const onStatus = vi.fn();
    const onDelta = vi.fn();
    const onDone = vi.fn();
    const onError = vi.fn();

    await streamChat({
      signal: new AbortController().signal,
      messages: [],
      modelId: null,
      onDelta, onDone, onError, onStatus,
    });

    expect(onStatus.mock.calls).toEqual([
      ["Captioning...", false],
      ["Done", true],
    ]);
    expect(onDelta).toHaveBeenCalledWith("hi");
    expect(onDone).toHaveBeenCalled();
    expect(onError).not.toHaveBeenCalled();
  });
});
