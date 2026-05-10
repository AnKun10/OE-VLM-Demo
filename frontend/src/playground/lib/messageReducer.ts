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
        (m) => (m.status === "streaming" ? { ...m, text: m.text + action.delta } : m),
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
