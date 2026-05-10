import { describe, it, expect, beforeEach } from "vitest";
import { readState, writeState, __TEST__ } from "./storage";
import type { ConversationsState } from "../types";

const STORAGE_KEY = __TEST__.STORAGE_KEY;

function makeState(): ConversationsState {
  return {
    schemaVersion: 1,
    activeId: "c1",
    conversations: {
      c1: {
        id: "c1",
        title: "T",
        modelId: "m",
        messages: [
          { id: "u1", role: "user", text: "hi", status: "done", createdAt: 1 },
        ],
        createdAt: 1,
        updatedAt: 2,
      },
    },
  };
}

describe("storage", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("T4.1 — write + read round-trips state", () => {
    const state = makeState();
    writeState(state);
    const round = readState();
    expect(round).not.toBeNull();
    expect(round!.activeId).toBe("c1");
    expect(round!.conversations.c1.messages.length).toBe(1);
    expect(round!.conversations.c1.messages[0].text).toBe("hi");
  });

  it("returns null when storage is empty", () => {
    expect(readState()).toBeNull();
  });

  it("A4.1 — invalid JSON returns null without throwing", () => {
    localStorage.setItem(STORAGE_KEY, "not-json");
    expect(readState()).toBeNull();
  });

  it("A4.2 — wrong schemaVersion returns null", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ schemaVersion: 99, conversations: {}, activeId: null }),
    );
    expect(readState()).toBeNull();
  });

  it("rejects payload missing conversations field", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ schemaVersion: 1 }));
    expect(readState()).toBeNull();
  });

  it("rejects payload that is not an object (array)", () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([1, 2, 3]));
    expect(readState()).toBeNull();
  });
});
