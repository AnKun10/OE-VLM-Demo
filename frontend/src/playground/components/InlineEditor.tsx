import { Check, X } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { useToast } from "../hooks/useToast";

/**
 * In-place editor for a user message bubble. Renders a textarea + Save/Cancel.
 *
 * Keyboard shortcuts:
 *   - Esc → cancel
 *   - Ctrl/Cmd+Enter → save
 *   - Enter alone → newline (multi-line edits supported)
 *
 * Empty-text save (A3.6) is rejected with a toast "Tin nhắn rỗng".
 */
export function InlineEditor({
  initialText,
  onSave,
  onCancel,
}: {
  initialText: string;
  onSave: (text: string) => void;
  onCancel: () => void;
}) {
  const [text, setText] = useState(initialText);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const toast = useToast();

  // Auto-focus + select-all on mount.
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.focus();
      ta.selectionStart = ta.value.length;
      ta.selectionEnd = ta.value.length;
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    }
  }, []);

  // Auto-resize as user types.
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
    }
  }, [text]);

  function handleSave() {
    if (text.trim().length === 0) {
      toast.push("Tin nhắn rỗng.", "error");
      return;
    }
    onSave(text);
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      onCancel();
    } else if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSave();
    }
  }

  return (
    <div
      className="flex flex-col gap-2 w-full"
      style={{ maxWidth: "85%" }}
    >
      <textarea
        ref={taRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        rows={1}
        className="w-full resize-none outline-none text-[15px] leading-relaxed"
        style={{
          background: "#ffffff",
          color: "#111827",
          border: "1px solid #015e9f",
          borderRadius: 12,
          padding: "10px 14px",
          maxHeight: 240,
        }}
        aria-label="Chỉnh sửa tin nhắn"
      />
      <div className="flex items-center justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs transition-colors"
          style={{
            background: "transparent",
            color: "#6b7280",
            border: "1px solid #e5e7eb",
          }}
          aria-label="Huỷ"
          title="Huỷ (Esc)"
        >
          <X size={13} />
          Huỷ
        </button>
        <button
          type="button"
          onClick={handleSave}
          className="flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium transition-colors"
          style={{
            background: "#015e9f",
            color: "#ffffff",
          }}
          aria-label="Lưu"
          title="Lưu (Ctrl+Enter)"
        >
          <Check size={13} />
          Lưu
        </button>
      </div>
    </div>
  );
}
