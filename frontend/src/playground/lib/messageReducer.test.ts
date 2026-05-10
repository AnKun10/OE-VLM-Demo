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

  it("APPEND_DELTA is a no-op when target message is not streaming", () => {
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
    // Late delta after MARK_DONE should NOT mutate text.
    s = conversationsReducer(s, {
      type: "APPEND_DELTA",
      conversationId: "c1",
      messageId: "a1",
      delta: "late",
    });
    expect(s.conversations.c1.messages.at(-1)!.text).toBe("");
    expect(s.conversations.c1.messages.at(-1)!.status).toBe("done");
  });

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

  it("T4.2 — HYDRATE replaces state with payload", () => {
    const initial = withConv(initialState(), "c1", "m");
    const payload: ConversationsState = {
      schemaVersion: 1,
      activeId: "c-other",
      conversations: {
        "c-other": {
          id: "c-other",
          title: "Old",
          modelId: "m",
          messages: [],
          createdAt: 1,
          updatedAt: 1,
        },
      },
    };
    const out = conversationsReducer(initial, { type: "HYDRATE", state: payload });
    expect(out.activeId).toBe("c-other");
    expect(out.conversations["c-other"]).toBeDefined();
    expect(out.conversations.c1).toBeUndefined();
  });

  it("A4.3 — HYDRATE coerces in-flight streaming → stopped while preserving text", () => {
    const payload: ConversationsState = {
      schemaVersion: 1,
      activeId: "c1",
      conversations: {
        c1: {
          id: "c1",
          title: "T",
          modelId: "m",
          messages: [
            { id: "u", role: "user", text: "hi", status: "done", createdAt: 1 },
            {
              id: "a",
              role: "assistant",
              text: "partial reply",
              status: "streaming",
              createdAt: 2,
            },
          ],
          createdAt: 1,
          updatedAt: 2,
        },
      },
    };
    const out = conversationsReducer(initialState(), {
      type: "HYDRATE",
      state: payload,
    });
    const last = out.conversations.c1.messages.at(-1)!;
    expect(last.status).toBe("stopped");
    expect(last.text).toBe("partial reply");
  });

  it("T4.3 — RENAME_TITLE updates only the targeted conversation", () => {
    let s = withConv(initialState(), "c1", "m");
    s = withConv(s, "c2", "m");
    s = conversationsReducer(s, {
      type: "RENAME_TITLE",
      conversationId: "c1",
      title: "NEW",
    });
    expect(s.conversations.c1.title).toBe("NEW");
    expect(s.conversations.c2.title).toBe("Cuộc hội thoại mới");
  });

  it("T4.4 — DELETE_CONVERSATION fallback: most-recent remaining or null", () => {
    let s = withConv(initialState(), "c1", "m"); // updatedAt 1000
    s = conversationsReducer(s, {
      type: "NEW_CONVERSATION",
      conversationId: "c2",
      welcomeMessageId: "w2",
      modelId: "m",
      now: 5000,
    });
    expect(s.activeId).toBe("c2");
    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c2" });
    expect(s.activeId).toBe("c1");
    s = conversationsReducer(s, { type: "DELETE_CONVERSATION", id: "c1" });
    expect(s.activeId).toBeNull();
  });
});
