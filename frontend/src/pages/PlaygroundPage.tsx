import { useEffect, useMemo, useReducer, useRef, useState } from "react";
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
import type { MessageActions } from "../playground/components/MessageBubble";
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
  const [editingId, setEditingId] = useState<string | null>(null);

  // Latest dispatch reference for use inside async callbacks (avoids stale closures
  // when streamingId / activeId changes mid-stream).
  const dispatchRef = useRef(dispatch);
  dispatchRef.current = dispatch;

  const activeId = state.activeId!;
  const active = state.conversations[activeId]!;
  const messages = active.messages;

  // Once /api/models loads, default the active conversation's modelId
  // to the first vision-capable model if none is set yet.
  useEffect(() => {
    if (!active.modelId && models.length > 0) {
      dispatch({
        type: "SET_MODEL",
        conversationId: activeId,
        modelId: models[0].id,
      });
    }
  }, [models, active.modelId, activeId]);

  const effectiveModelId =
    active.modelId || models[0]?.id || "";
  const activeModel = models.find((m) => m.id === effectiveModelId);
  const visionEnabled = activeModel?.capabilities.vision ?? true;

  const isStreaming = useMemo(
    () => messages.some((m) => m.status === "streaming"),
    [messages],
  );

  const historyImageCount = useMemo(
    () =>
      messages.reduce((n, m) => n + (m.attachments?.length ?? 0), 0),
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
    setEditingId(null);
  }

  function selectConversation(id: string) {
    abort();
    dispatch({ type: "SELECT_CONVERSATION", id });
    setText("");
    setAttachments([]);
    setEditingId(null);
  }

  function deleteConversation(id: string) {
    dispatch({ type: "DELETE_CONVERSATION", id });
    if (Object.keys(state.conversations).length <= 1) {
      newConversation();
    }
  }

  /**
   * Reusable streaming runner. Caller is responsible for having dispatched
   * ADD_USER_MESSAGE + ADD_ASSISTANT_PLACEHOLDER before calling. We pass
   * `wireMessages` (the OpenAI-shaped history including the user message)
   * + `assistantId` (the placeholder we'll fill).
   */
  async function runStream(
    wireMessages: ChatMessageWithAttachments[],
    assistantId: string,
    convId: string,
  ) {
    await send({
      messages: wireMessages,
      modelId: effectiveModelId || null,
      onDelta: (delta) =>
        dispatchRef.current({
          type: "APPEND_DELTA",
          conversationId: convId,
          messageId: assistantId,
          delta,
        }),
      onDone: () =>
        dispatchRef.current({
          type: "MARK_DONE",
          conversationId: convId,
          messageId: assistantId,
        }),
      onError: (e) =>
        dispatchRef.current({
          type: "MARK_ERROR",
          conversationId: convId,
          messageId: assistantId,
          errorKind: e.errorKind,
        }),
    });
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
    if (messages.filter((m) => m.role === "user").length === 0) {
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
    await runStream(wire, assistantId, activeId);
  }

  function handleStop() {
    abort();
    // Mark the streaming placeholder as stopped. Find the last streaming msg.
    const streamingMsg = [...messages].reverse().find((m) => m.status === "streaming");
    if (streamingMsg) {
      dispatch({
        type: "MARK_STOPPED",
        conversationId: activeId,
        messageId: streamingMsg.id,
      });
    }
  }

  async function handleRegenerate() {
    if (isStreaming) return;
    // Pop the last assistant message and re-stream from the prior context.
    dispatch({ type: "POP_LAST_ASSISTANT", conversationId: activeId });
    // Dispatch reads the post-pop messages on the next render; we need the
    // computed message list here, so build it from the current `messages`.
    const remaining = messages.slice(0, -1);
    if (remaining.length === 0 || remaining.at(-1)?.role !== "user") return;
    const assistantId = uid();
    dispatch({
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: activeId,
      messageId: assistantId,
      now: Date.now(),
    });
    const wire = toWireMessages(remaining);
    await runStream(wire, assistantId, activeId);
  }

  async function handleSaveEdit(messageId: string, newText: string) {
    setEditingId(null);
    dispatch({
      type: "EDIT_USER_AND_TRUNCATE",
      conversationId: activeId,
      messageId,
      newText,
    });
    // Build the new message list manually (reducer change isn't visible until
    // re-render): take messages up to and including the edited one, replace text.
    const idx = messages.findIndex((m) => m.id === messageId);
    if (idx === -1) return;
    const editedMsg: Message = { ...messages[idx], text: newText };
    const truncated = [...messages.slice(0, idx), editedMsg];
    const assistantId = uid();
    dispatch({
      type: "ADD_ASSISTANT_PLACEHOLDER",
      conversationId: activeId,
      messageId: assistantId,
      now: Date.now(),
    });
    const wire = toWireMessages(truncated);
    await runStream(wire, assistantId, activeId);
  }

  function handleStartEdit(messageId: string) {
    if (isStreaming) return;
    setEditingId(messageId);
  }

  function handleCancelEdit() {
    setEditingId(null);
  }

  function handleModelChange(modelId: string) {
    dispatch({
      type: "SET_MODEL",
      conversationId: activeId,
      modelId,
    });
  }

  const messageActions: MessageActions = {
    isStreaming,
    editingId,
    onStartEdit: handleStartEdit,
    onSaveEdit: handleSaveEdit,
    onCancelEdit: handleCancelEdit,
    onRegenerate: handleRegenerate,
  };

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
          <MessageList messages={messages} actions={messageActions} />
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
              onChange={handleModelChange}
            />
          }
          visionEnabled={visionEnabled}
          visionWarning={
            !visionEnabled && (attachments.length > 0 || historyImageCount > 0)
              ? "Model mới không hỗ trợ ảnh; gửi sẽ thất bại."
              : null
          }
          historyImageCount={historyImageCount}
          streaming={isStreaming}
          onStop={handleStop}
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
