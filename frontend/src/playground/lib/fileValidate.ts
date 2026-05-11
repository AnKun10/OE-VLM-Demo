import { FriendlyError } from "./errors";

export const ALLOWED_MIME = [
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
] as const;

export const MAX_BYTES = 10 * 1024 * 1024; // 10 MiB

/**
 * Per-turn image cap. The user can attach at most MAX_IMAGES images to the
 * tin nhắn currently being composed. History images are NOT counted —
 * Phase 5's image compressor strips them down to ≤1 pixel-bearing turn at
 * the backend, so a 50-message conversation can still accept a fresh
 * 4-image upload on the next turn.
 */
export const MAX_IMAGES = 4;

export function validateFile(f: File): void {
  if (!ALLOWED_MIME.includes(f.type as (typeof ALLOWED_MIME)[number])) {
    throw new FriendlyError("unsupported_mime", f.type || "unknown");
  }
  if (f.size === 0) throw new FriendlyError("empty_file");
  if (f.size > MAX_BYTES) throw new FriendlyError("too_large");
}

/** True if you can still add an attachment without breaking the per-turn cap. */
export function checkAttachmentCap(currentCount: number): boolean {
  return currentCount < MAX_IMAGES;
}
