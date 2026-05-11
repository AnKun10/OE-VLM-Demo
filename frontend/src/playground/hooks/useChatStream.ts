import { useCallback, useRef } from "react";
import { streamChat, type ChatStreamCallbacks } from "../lib/chatStream";
import type { ChatMessageWithAttachments } from "../types";

export type SendArgs = ChatStreamCallbacks & {
  messages: ChatMessageWithAttachments[];
  modelId: string | null;
};

export function useChatStream(): {
  send: (args: SendArgs) => Promise<void>;
  abort: () => void;
} {
  const ctrlRef = useRef<AbortController | null>(null);

  const send = useCallback(async (args: SendArgs) => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    await streamChat({
      signal: ctrl.signal,
      messages: args.messages,
      modelId: args.modelId,
      onDelta: args.onDelta,
      onDone: args.onDone,
      onError: args.onError,
      onStatus: args.onStatus,
    });
  }, []);

  const abort = useCallback(() => {
    ctrlRef.current?.abort();
  }, []);

  return { send, abort };
}
