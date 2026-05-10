import { Sparkles } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { SafeLink } from "./SafeLink";
import type { Message } from "../types";

const ACCENT = "#015e9f";
const TEXT_PRIMARY = "#111827";
const TEXT_MUTED = "#9ca3af";
const BORDER = "#e5e7eb";

function UserBubble({ msg }: { msg: Message }) {
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
      {msg.text && (
        <div
          className="text-[16px] leading-relaxed whitespace-pre-wrap"
          style={{
            background: "#0d1b67",
            color: "#ffffff",
            borderRadius: "18px 18px 4px 18px",
            padding: "10px 16px",
            maxWidth: "85%",
          }}
        >
          {msg.text}
        </div>
      )}
    </div>
  );
}

function AssistantBubble({ msg }: { msg: Message }) {
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
          style={{ color: TEXT_PRIMARY }}
        >
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{ a: SafeLink as never }}
          >
            {msg.text || ""}
          </ReactMarkdown>
        </div>
      </div>
    </div>
  );
}

export function MessageBubble({ msg }: { msg: Message }) {
  return msg.role === "user" ? <UserBubble msg={msg} /> : <AssistantBubble msg={msg} />;
}
