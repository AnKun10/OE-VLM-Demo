import { AttachmentPreview } from "./AttachmentPreview";
import type { AttachmentRef } from "../types";

export function AttachmentRail({
  attachments,
  onRemove,
}: {
  attachments: AttachmentRef[];
  onRemove: (id: string) => void;
}) {
  if (attachments.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-2 px-4 pt-3">
      {attachments.map((a) => (
        <AttachmentPreview key={a.id} attachment={a} onRemove={onRemove} />
      ))}
    </div>
  );
}
