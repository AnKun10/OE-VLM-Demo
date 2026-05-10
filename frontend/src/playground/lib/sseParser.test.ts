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
