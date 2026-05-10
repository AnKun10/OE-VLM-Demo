import { FriendlyError } from "./errors";

export const ALLOWED_MIME = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;

export const MAX_BYTES = 10 * 1024 * 1024; // 10 MiB

export const MAX_IMAGES = 4;

export function validateFile(f: File): void {
  if (!ALLOWED_MIME.includes(f.type as (typeof ALLOWED_MIME)[number])) {
    throw new FriendlyError("unsupported_mime", f.type || "unknown");
  }
  if (f.size === 0) throw new FriendlyError("empty_file");
  if (f.size > MAX_BYTES) throw new FriendlyError("too_large");
}

/** True if you can still add an attachment without breaking the cap. */
export function checkAttachmentCap(
  currentCount: number,
  historyImageCount: number,
): boolean {
  return currentCount + historyImageCount < MAX_IMAGES;
}
