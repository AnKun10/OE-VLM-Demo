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
