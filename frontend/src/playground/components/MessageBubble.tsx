import { Pencil, RefreshCw, Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { SafeLink } from "./SafeLink";
import { InlineEditor } from "./InlineEditor";
import type { Message } from "../types";

const ACCENT = "#015e9f";
const TEXT_PRIMARY = "#111827";
const TEXT_MUTED = "#9ca3af";
const TEXT_SECONDARY = "#6b7280";
const BORDER = "#e5e7eb";

export type MessageActions = {
  /** True iff any message in the active conversation has status "streaming". */
  isStreaming: boolean;
  /** Currently-edited message id (or null). Edit button is disabled when set. */
  editingId: string | null;
  onStartEdit: (messageId: string) => void;
  onSaveEdit: (messageId: string, newText: string) => void;
  onCancelEdit: () => void;
  onRegenerate: () => void;
};

function UserBubble({
  msg,
  actions,
}: {
  msg: Message;
  actions: MessageActions;
}) {
  const isEditing = actions.editingId === msg.id;

  if (isEditing) {
    return (
      <div className="flex flex-col items-end gap-1">
        {msg.attachments && msg.attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 justify-end" style={{ maxWidth: "85%" }}>
            {msg.attachments.map((a) => (
              <div
                key={a.id}
                className="overflow-hidden"
                style={{ width: 120, height: 120, borderRadius: 12, border: `1px solid ${BORDER}` }}
              >
                <img src={a.url} alt={a.originalName} className="w-full h-full object-cover" />
              </div>
            ))}
          </div>
        )}
        <InlineEditor
          initialText={msg.text}
          onSave={(text) => actions.onSaveEdit(msg.id, text)}
          onCancel={actions.onCancelEdit}
        />
      </div>
    );
  }

  const canEdit = !actions.isStreaming && actions.editingId === null;

  return (
    <div className="group flex flex-col items-end gap-1">
      {msg.attachments && msg.attachments.length > 0 && (
        <div className="flex flex-wrap gap-2 justify-end" style={{ maxWidth: "85%" }}>
          {msg.attachments.map((a) => (
            <div
              key={a.id}
              className="overflow-hidden"
              style={{ width: 120, height: 120, borderRadius: 12, border: `1px solid ${BORDER}` }}
            >
              <img src={a.url} alt={a.originalName} className="w-full h-full object-cover" />
            </div>
          ))}
        </div>
      )}
      {msg.text && (
        <div className="flex items-end gap-1.5" style={{ maxWidth: "85%" }}>
          {canEdit && (
            <button
              type="button"
              onClick={() => actions.onStartEdit(msg.id)}
              className="opacity-0 group-hover:opacity-100 p-1.5 rounded transition-all"
              style={{ color: TEXT_MUTED }}
              aria-label="Chỉnh sửa"
              title="Chỉnh sửa"
            >
              <Pencil size={13} />
            </button>
          )}
          <div
            className="text-[16px] leading-relaxed whitespace-pre-wrap"
            style={{
              background: "#0d1b67",
              color: "#ffffff",
              borderRadius: "18px 18px 4px 18px",
              padding: "10px 16px",
            }}
          >
            {msg.text}
          </div>
        </div>
      )}
    </div>
  );
}

function AssistantBubble({
  msg,
  isLast,
  actions,
}: {
  msg: Message;
  isLast: boolean;
  actions: MessageActions;
}) {
  const isError = msg.status === "error";
  const isStopped = msg.status === "stopped";
  const isDone = msg.status === "done";
  const canRegenerate =
    isLast && (isDone || isStopped || isError) && !actions.isStreaming;

  return (
    <div className="flex gap-3">
      <div className="flex-shrink-0 pt-0.5">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: "rgba(1,94,159,0.15)" }}
        >
          <Sparkles size={15} style={{ color: ACCENT }} />
        </div>
      </div>
      <div className="flex-1 min-w-0">
        <span className="text-xs font-medium" style={{ color: TEXT_MUTED }}>
          AI Model
        </span>
        <div
          className="mt-1 text-[16px] leading-relaxed prose prose-sm max-w-none"
          style={{
            color: isError ? "#991b1b" : TEXT_PRIMARY,
            background: isError ? "#fef2f2" : "transparent",
            border: isError ? "1px solid #fecaca" : "none",
            borderRadius: isError ? 12 : 0,
            padding: isError ? "10px 14px" : 0,
          }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{ a: SafeLink as never }}
          >
            {msg.text || ""}
          </ReactMarkdown>
          {isStopped && (
            <span
              className="text-xs italic"
              style={{ color: TEXT_MUTED, marginLeft: 4 }}
            >
              [bị dừng]
            </span>
          )}
        </div>
        {canRegenerate && (
          <div className="mt-2">
            <button
              type="button"
              onClick={actions.onRegenerate}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs transition-colors"
              style={{
                background: isError ? "#015e9f" : "transparent",
                color: isError ? "#ffffff" : TEXT_SECONDARY,
                border: isError ? "none" : `1px solid ${BORDER}`,
              }}
              aria-label={isError ? "Thử lại" : "Tạo lại"}
              title={isError ? "Thử lại" : "Tạo lại phản hồi"}
            >
              <RefreshCw size={12} />
              {isError ? "Thử lại" : "Tạo lại"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export function MessageBubble({
  msg,
  isLast,
  actions,
}: {
  msg: Message;
  isLast: boolean;
  actions: MessageActions;
}) {
  return msg.role === "user" ? (
    <UserBubble msg={msg} actions={actions} />
  ) : (
    <AssistantBubble msg={msg} isLast={isLast} actions={actions} />
  );
}
