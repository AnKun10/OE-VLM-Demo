/**
 * Friendly errors with a stable i18n key + Vietnamese fallback message.
 * Hooks/components can pattern-match on `.key` and choose to display either
 * the message directly or a translation.
 */

export type FriendlyErrorKey =
  | "unsupported_mime"
  | "empty_file"
  | "too_large"
  | "invalid_response"
  | "upload_network"
  | "upload_http"
  | "send_network"
  | "send_http"
  | "stream_drop"
  | "vision_required"
  | "attachment_cap"
  | "empty_message";

const MESSAGES_VI: Record<FriendlyErrorKey, string> = {
  unsupported_mime: "Định dạng không hỗ trợ. Chỉ chấp nhận PNG, JPEG, WebP, GIF.",
  empty_file: "Tệp rỗng.",
  too_large: "Tệp quá lớn (> 10 MB).",
  invalid_response: "Phản hồi từ máy chủ không hợp lệ.",
  upload_network: "Mất kết nối khi tải tệp lên.",
  upload_http: "Máy chủ từ chối tệp.",
  send_network: "Mất kết nối khi gửi yêu cầu.",
  send_http: "Máy chủ từ chối yêu cầu.",
  stream_drop: "Mất kết nối giữa lúc đang nhận phản hồi.",
  vision_required: "Model hiện tại không hỗ trợ ảnh.",
  attachment_cap: "Tối đa 4 ảnh trong một cuộc trò chuyện.",
  empty_message: "Tin nhắn rỗng.",
};

export class FriendlyError extends Error {
  readonly key: FriendlyErrorKey;
  constructor(key: FriendlyErrorKey, detail?: string) {
    const base = detail ? `${MESSAGES_VI[key]} (${detail})` : MESSAGES_VI[key];
    super(`[${key}] ${base}`);
    this.key = key;
    this.name = "FriendlyError";
  }
}
