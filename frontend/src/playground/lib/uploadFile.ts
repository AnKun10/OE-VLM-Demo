import { FriendlyError } from "./errors";
import { validateFile } from "./fileValidate";
import type { AttachmentRef } from "../types";

/**
 * Upload one image file to the backend. Validates the file client-side
 * BEFORE making a network call. On the wire, expects a `StoredFile`
 * camelCase JSON shape:
 *   { id, url, mime, size, originalName }
 * The `size` field is dropped here — frontend doesn't track it.
 */
export async function uploadFile(file: File): Promise<AttachmentRef> {
  validateFile(file); // throws FriendlyError

  const fd = new FormData();
  fd.append("file", file);

  let resp: Response;
  try {
    resp = await fetch("/api/files", { method: "POST", body: fd });
  } catch (e) {
    throw new FriendlyError("upload_network", String(e));
  }

  if (!resp.ok) {
    throw new FriendlyError("upload_http", `HTTP ${resp.status}`);
  }

  let data: unknown;
  try {
    data = await resp.json();
  } catch {
    throw new FriendlyError("invalid_response", "not JSON");
  }
  if (
    typeof data !== "object" ||
    data === null ||
    typeof (data as { id?: unknown }).id !== "string" ||
    typeof (data as { url?: unknown }).url !== "string"
  ) {
    throw new FriendlyError("invalid_response", "missing fields");
  }
  const d = data as Record<string, string>;
  return {
    id: d.id,
    url: d.url,
    mime: d.mime ?? "",
    originalName: d.originalName ?? d.original_name ?? "",
  };
}
