// frontend/src/playground/types.ts
/**
 * Type definitions for the playground frontend.
 * These mirror the backend Pydantic models in `backend/app/routers/chat.py`
 * (ChatMessageWithAttachments, Attachment, ChatStreamRequest) and the
 * StoredFile response shape from `backend/app/services/files.py`
 * (camelCase via Pydantic alias_generator=to_camel).
 */

export type AttachmentRef = {
  id: string;
  url: string;
  mime: string;
  originalName: string;
};

export type MessageStatus = "streaming" | "done" | "stopped" | "error";

export type ErrorKind =
  | "connection"
  | "file_missing"
  | "bad_request"
  | "internal"
  | "network"
  | "http"
  | "stream"
  | "parse";

export type Message = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachments?: AttachmentRef[];
  status?: MessageStatus;
  errorKind?: ErrorKind;
  createdAt: number;
};

export type Conversation = {
  id: string;
  title: string;
  modelId: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
};

export type ConversationsState = {
  schemaVersion: 1;
  conversations: Record<string, Conversation>;
  activeId: string | null;
};

export type ModelInfo = {
  id: string;
  name: string;
  capabilities: { vision: boolean };
};

/** Wire types — what the backend actually accepts/returns over HTTP. */

export type ChatMessageWithAttachments = {
  role: "user" | "assistant";
  text: string;
  attachments: { id: string }[];
};

export type ChatStreamRequest = {
  messages: ChatMessageWithAttachments[];
  model_id: string | null;
};
