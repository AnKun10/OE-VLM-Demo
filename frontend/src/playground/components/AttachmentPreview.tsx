import { X } from "lucide-react";
import type { AttachmentRef } from "../types";

export function AttachmentPreview({
  attachment,
  onRemove,
}: {
  attachment: AttachmentRef;
  onRemove?: (id: string) => void;
}) {
  return (
    <div className="relative group" style={{ width: 80, height: 80 }}>
      <div
        className="w-full h-full overflow-hidden"
        style={{ borderRadius: 10, border: "1px solid #e5e7eb" }}
      >
        <img
          src={attachment.url}
          alt={attachment.originalName}
          className="w-full h-full object-cover"
        />
      </div>
      {onRemove && (
        <button
          type="button"
          onClick={() => onRemove(attachment.id)}
          aria-label={`Xoá ${attachment.originalName}`}
          className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          style={{ background: "#dc2626" }}
        >
          <X size={11} className="text-white" />
        </button>
      )}
    </div>
  );
}
