import { useEffect, useRef } from "react";
import { Sparkles } from "lucide-react";
import { MessageBubble, type MessageActions } from "./MessageBubble";
import type { Message } from "../types";

const AT_BOTTOM_TOLERANCE = 32;

export function MessageList({
  messages,
  actions,
}: {
  messages: Message[];
  actions: MessageActions;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const wasAtBottomRef = useRef(true);
  const lastMsgKey =
    messages.at(-1)?.id + "@" + (messages.at(-1)?.text.length ?? 0);

  // Attach scroll listener to the parent overflow-y-auto container in
  // PlaygroundPage so we can track whether the user is at the bottom.
  useEffect(() => {
    const scrollEl = containerRef.current?.parentElement;
    if (!scrollEl) return;
    const onScroll = () => {
      wasAtBottomRef.current =
        scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight <
        AT_BOTTOM_TOLERANCE;
    };
    scrollEl.addEventListener("scroll", onScroll, { passive: true });
    // Initial seed.
    onScroll();
    return () => scrollEl.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    if (wasAtBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [lastMsgKey]);

  const lastIsStreaming = messages.at(-1)?.status === "streaming";
  const lastIsEmpty = lastIsStreaming && (messages.at(-1)?.text ?? "").length === 0;

  return (
    <div ref={containerRef} className="max-w-3xl mx-auto px-4 py-8 space-y-6">
      {messages.map((m, i) => (
        <MessageBubble
          key={m.id}
          msg={m}
          isLast={i === messages.length - 1}
          actions={actions}
        />
      ))}
      {lastIsEmpty && (
        <div className="flex gap-3">
          <div
            className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
            style={{ background: "rgba(1,94,159,0.15)" }}
          >
            <Sparkles size={15} style={{ color: "#015e9f" }} />
          </div>
          <div className="pt-2">
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-2 h-2 rounded-full animate-bounce"
                  style={{ background: "#015e9f", animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
