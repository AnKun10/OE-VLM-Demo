import { ImagePlus, Mic, Send } from "lucide-react";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ClipboardEvent,
  type DragEvent,
  type KeyboardEvent,
  type ReactNode,
} from "react";
import { AttachmentRail } from "./AttachmentRail";
import { DropOverlay } from "./DropOverlay";
import { useFileUpload } from "../hooks/useFileUpload";
import { useToast } from "../hooks/useToast";
import { FriendlyError } from "../lib/errors";
import { MAX_IMAGES, checkAttachmentCap } from "../lib/fileValidate";
import type { AttachmentRef } from "../types";

export type ComposerBarProps = {
  text: string;
  onTextChange: (s: string) => void;
  attachments: AttachmentRef[];
  onAttach: (a: AttachmentRef) => void;
  onRemoveAttachment: (id: string) => void;
  onSend: () => void;
  modelDropdown: ReactNode;
  visionEnabled: boolean;
  visionWarning?: string | null;
  historyImageCount: number;
  disabled?: boolean;
};

export function ComposerBar(props: ComposerBarProps) {
  const {
    text,
    onTextChange,
    attachments,
    onAttach,
    onRemoveAttachment,
    onSend,
    modelDropdown,
    visionEnabled,
    visionWarning,
    historyImageCount,
    disabled,
  } = props;

  const { upload, uploading } = useFileUpload();
  const toast = useToast();
  const fileRef = useRef<HTMLInputElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  // Auto-resize textarea
  useEffect(() => {
    const ta = taRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  }, [text]);

  const canAddMore = checkAttachmentCap(attachments.length, historyImageCount);

  const handleFiles = useCallback(
    async (files: File[]) => {
      if (!visionEnabled) {
        toast.push("Model hiện tại không hỗ trợ ảnh.", "error");
        return;
      }
      let added = 0;
      for (const f of files) {
        if (!checkAttachmentCap(attachments.length + added, historyImageCount)) {
          toast.push(`Tối đa ${MAX_IMAGES} ảnh trong một cuộc trò chuyện.`, "error");
          break;
        }
        try {
          const ref = await upload(f);
          onAttach(ref);
          added++;
        } catch (e) {
          if (e instanceof FriendlyError) toast.push(e.message, "error");
          else toast.push("Lỗi không xác định khi tải ảnh.", "error");
        }
      }
    },
    [attachments.length, historyImageCount, onAttach, toast, upload, visionEnabled],
  );

  const onPickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    void handleFiles(files);
  };

  const onPaste = (e: ClipboardEvent<HTMLTextAreaElement>) => {
    const items = Array.from(e.clipboardData?.items ?? []);
    const files: File[] = [];
    for (const it of items) {
      if (it.kind === "file") {
        const f = it.getAsFile();
        if (f) files.push(f);
      }
    }
    if (files.length > 0) {
      e.preventDefault();
      void handleFiles(files);
    }
  };

  const onDragEnter = (e: DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.types.includes("Files")) {
      dragCounter.current += 1;
      setDragging(true);
    }
  };
  const onDragLeave = (e: DragEvent) => {
    e.preventDefault();
    dragCounter.current -= 1;
    if (dragCounter.current <= 0) {
      dragCounter.current = 0;
      setDragging(false);
    }
  };
  const onDragOver = (e: DragEvent) => {
    e.preventDefault();
  };
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    dragCounter.current = 0;
    setDragging(false);
    const files = Array.from(e.dataTransfer.files ?? []);
    void handleFiles(files);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSend();
    }
  };

  const canSend =
    !disabled &&
    !uploading &&
    (visionEnabled || attachments.length === 0) &&
    (text.trim().length > 0 || attachments.length > 0) &&
    (!visionWarning || attachments.length === 0);

  return (
    <div
      className="flex-shrink-0 px-4 pb-5 pt-2"
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <DropOverlay visible={dragging && visionEnabled} />
      <div className="max-w-3xl mx-auto">
        {visionWarning && (
          <div
            className="mb-2 px-3 py-2 rounded-md text-xs"
            style={{
              background: "#fef3c7",
              color: "#92400e",
              border: "1px solid #fcd34d",
            }}
          >
            {visionWarning}
          </div>
        )}
        <div
          className="rounded-2xl overflow-hidden transition-all"
          style={{
            background: "#ffffff",
            border: "1px solid #e5e7eb",
            boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
          }}
        >
          <AttachmentRail attachments={attachments} onRemove={onRemoveAttachment} />
          <textarea
            ref={taRef}
            value={text}
            onChange={(e) => onTextChange(e.target.value)}
            onKeyDown={onKeyDown}
            onPaste={onPaste}
            placeholder="Nhập tin nhắn..."
            disabled={disabled}
            rows={1}
            className="w-full resize-none bg-transparent outline-none text-sm px-4 pt-3.5 pb-1 disabled:opacity-50"
            style={{ color: "#111827", caretColor: "#015e9f", maxHeight: 160 }}
          />
          <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
            <div className="flex items-center gap-1">
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                multiple
                className="hidden"
                onChange={onPickerChange}
              />
              {visionEnabled && (
                <button
                  type="button"
                  onClick={() => fileRef.current?.click()}
                  disabled={!canAddMore || uploading}
                  className="p-2 rounded-lg transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{ color: "#9ca3af" }}
                  title={
                    !canAddMore
                      ? `Tối đa ${MAX_IMAGES} ảnh`
                      : "Đính kèm ảnh"
                  }
                  aria-label="Đính kèm ảnh"
                >
                  <ImagePlus size={18} />
                </button>
              )}
              {visionEnabled && attachments.length > 0 && (
                <span className="text-[11px]" style={{ color: "#9ca3af" }}>
                  {attachments.length}/{MAX_IMAGES}
                </span>
              )}
              <button
                type="button"
                className="p-2 rounded-lg transition-colors"
                style={{ color: "#9ca3af" }}
                aria-label="Microphone (chưa hoạt động)"
                disabled
              >
                <Mic size={18} />
              </button>
              {modelDropdown}
            </div>
            <button
              type="button"
              onClick={onSend}
              disabled={!canSend}
              className="flex items-center justify-center rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
              style={{
                width: 36,
                height: 36,
                background: canSend ? "#015e9f" : "#9ca3af",
              }}
              aria-label="Gửi"
            >
              <Send size={15} className="text-white" style={{ marginLeft: 1 }} />
            </button>
          </div>
        </div>
        <p className="text-center mt-2.5 text-[11px]" style={{ color: "#9ca3af" }}>
          AI Playground sử dụng các mô hình AI. Kết quả có thể không chính xác.
        </p>
      </div>
    </div>
  );
}
