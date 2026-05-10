import { useEffect, useMemo, useReducer, useState } from "react";
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
  const [overrideModelId, setOverrideModelId] = useState<string>("");

  const activeId = state.activeId!;
  const active = state.conversations[activeId]!;
  const messages = active.messages;

  // When models load and the active conversation has no modelId yet, set it.
  useEffect(() => {
    if (!active.modelId && models.length > 0) {
      // Mutate via a synthetic action: rename + set modelId. For Phase 2,
      // just dispatch SELECT (no-op) and rely on render-time fallback.
      // Phase 3+ may add a SET_MODEL action.
    }
  }, [models, active.modelId]);

  const effectiveModelId =
    overrideModelId || active.modelId || models[0]?.id || "";
  const activeModel = models.find((m) => m.id === effectiveModelId);
  const visionEnabled = activeModel?.capabilities.vision ?? true;

  const historyImageCount = useMemo(
    () =>
      messages.reduce(
        (n, m) => n + (m.attachments?.length ?? 0),
        0,
      ),
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
  }

  function selectConversation(id: string) {
    abort();
    dispatch({ type: "SELECT_CONVERSATION", id });
    setText("");
    setAttachments([]);
  }

  function deleteConversation(id: string) {
    dispatch({ type: "DELETE_CONVERSATION", id });
    if (Object.keys(state.conversations).length <= 1) {
      // After delete this would be empty; create a fresh one.
      newConversation();
    }
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
    if (
      messages.filter((m) => m.role === "user").length === 0
    ) {
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
    await send({
      messages: wire,
      modelId: effectiveModelId || null,
      onDelta: (delta) =>
        dispatch({
          type: "APPEND_DELTA",
          conversationId: activeId,
          messageId: assistantId,
          delta,
        }),
      onDone: () =>
        dispatch({
          type: "MARK_DONE",
          conversationId: activeId,
          messageId: assistantId,
        }),
      onError: (e) =>
        dispatch({
          type: "MARK_ERROR",
          conversationId: activeId,
          messageId: assistantId,
          errorKind: e.errorKind,
        }),
    });
  }

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
          <MessageList messages={messages} />
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
              onChange={(id) => {
                // Phase 2: store override in local state so the wire request
                // honours the user's choice within the session.
                // Phase 3 will add a SET_MODEL reducer action that persists
                // the choice into Conversation.modelId.
                setOverrideModelId(id);
              }}
            />
          }
          visionEnabled={visionEnabled}
          visionWarning={
            !visionEnabled && (attachments.length > 0 || historyImageCount > 0)
              ? "Model mới không hỗ trợ ảnh; gửi sẽ thất bại."
              : null
          }
          historyImageCount={historyImageCount}
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
