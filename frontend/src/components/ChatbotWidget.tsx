import { useState, useRef, useEffect } from "react";
import { Send, X, Mic, Plus } from "lucide-react";

interface Attachment {
  imageUrl: string;
  productName: string;
}

interface MessageImage {
  url: string;
  name: string;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: MessageImage[];
  timestamp: Date;
}

interface ChatResponse {
  reply: string;
  products?: { id: string; name: string; image_url?: string | null }[];
}

async function sendChatMessage(
  message: string,
  history: { role: string; content: string }[]
): Promise<ChatResponse> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok) throw new Error("Chat request failed");
  return res.json();
}

function BotIcon({ size = 24, className = "" }: { size?: number; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" className={className}>
      <path d="M12 2a2 2 0 0 1 2 2v1h2a3 3 0 0 1 3 3v8a3 3 0 0 1-3 3H8a3 3 0 0 1-3-3V8a3 3 0 0 1 3-3h2V4a2 2 0 0 1 2-2z" />
      <circle cx="9" cy="11" r="1" fill="currentColor" stroke="none" />
      <circle cx="15" cy="11" r="1" fill="currentColor" stroke="none" />
      <path d="M9 15s1 1.5 3 1.5 3-1.5 3-1.5" />
      <line x1="12" y1="2" x2="12" y2="4" />
    </svg>
  );
}

export default function ChatbotWidget() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Xin chào! Tôi là trợ lý mua sắm của RunShop. Tôi có thể giúp bạn tìm giày chạy bộ phù hợp. Bạn cần tìm loại giày nào?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
        inputRef.current?.focus();
      }, 50);
    }
  }, [isOpen, messages]);

  // Listen for product image add-to-chat events — only stage as attachment, don't send
  useEffect(() => {
    const handler = (e: CustomEvent<{ imageUrl: string; productName: string }>) => {
      const { imageUrl, productName } = e.detail;
      setIsOpen(true);
      setAttachments((prev) => {
        if (prev.some((a) => a.imageUrl === imageUrl)) return prev;
        return [...prev, { imageUrl, productName }];
      });
    };
    window.addEventListener("add-to-chat", handler as EventListener);
    return () => window.removeEventListener("add-to-chat", handler as EventListener);
  }, []);

  async function handleSend() {
    const text = input.trim();
    if ((!text && attachments.length === 0) || loading) return;

    const content = text || "Cho tôi biết thêm về sản phẩm này.";
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      images: attachments.length > 0
        ? attachments.map((a) => ({ url: a.imageUrl, name: a.productName }))
        : undefined,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setAttachments([]);
    setLoading(true);

    try {
      const history = messages.map((m) => ({ role: m.role, content: m.content }));
      const data = await sendChatMessage(content, history);
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: data.reply, timestamp: new Date() },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { id: (Date.now() + 1).toString(), role: "assistant", content: "Xin lỗi, đã có lỗi xảy ra. Vui lòng thử lại sau.", timestamp: new Date() },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function removeAttachment(imageUrl: string) {
    setAttachments((prev) => prev.filter((a) => a.imageUrl !== imageUrl));
  }

  const canSend = (input.trim().length > 0 || attachments.length > 0) && !loading;

  return (
    <>
      {/* Chat panel */}
      {isOpen && (
        <div
          className="fixed bottom-20 right-4 z-50 w-[360px] sm:w-[400px] flex flex-col rounded-[28px] overflow-hidden"
          style={{
            background: "#f5f5f5",
            border: "1px solid #eeeeee",
            boxShadow: "0 24px 60px rgba(0,0,0,0.14), 0 4px 16px rgba(0,0,0,0.06)",
            maxHeight: "calc(100vh - 6rem)",
          }}
        >
          {/* Header */}
          <div className="relative flex items-center gap-3 px-5 py-4 flex-shrink-0" style={{ background: "#015e9f" }}>
            <div
              className="flex items-center justify-center rounded-full flex-shrink-0"
              style={{ width: 48, height: 48, background: "rgba(255,255,255,0.18)" }}
            >
              <BotIcon size={26} className="text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="font-semibold text-white text-[15px] leading-tight">Trợ lý RunShop</p>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-2 h-2 rounded-full bg-green-400 inline-block" />
                <span className="text-[12px]" style={{ color: "rgba(255,255,255,0.75)" }}>Luôn sẵn sàng hỗ trợ</span>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="flex items-center justify-center rounded-full transition-colors flex-shrink-0"
              style={{ width: 32, height: 32, background: "rgba(255,255,255,0.15)" }}
              onMouseEnter={e => (e.currentTarget.style.background = "rgba(255,255,255,0.25)")}
              onMouseLeave={e => (e.currentTarget.style.background = "rgba(255,255,255,0.15)")}
            >
              <X size={16} className="text-white" />
            </button>
          </div>

          {/* Messages */}
          <div
            className="flex-1 overflow-y-auto px-4 py-4 space-y-3"
            style={{ minHeight: 390, maxHeight: 570, background: "#f5f5f5" }}
          >
            {messages.map((msg) => (
              <div key={msg.id} className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
                {msg.role === "assistant" && (
                  <div
                    className="flex-shrink-0 flex items-center justify-center rounded-full self-end"
                    style={{ width: 32, height: 32, background: "#015e9f" }}
                  >
                    <BotIcon size={16} className="text-white" />
                  </div>
                )}
                <div
                  className="max-w-[78%] flex flex-col gap-1.5"
                  style={{ alignItems: msg.role === "user" ? "flex-end" : "flex-start" }}
                >
                  {/* Attached product images row */}
                  {msg.images && msg.images.length > 0 && (
                    <div className="flex flex-wrap gap-1.5" style={{ justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
                      {msg.images.map((img, i) => (
                        <div
                          key={i}
                          className="overflow-hidden flex-shrink-0"
                          title={img.name}
                          style={{
                            width: 80,
                            height: 80,
                            borderRadius: 14,
                            border: "1px solid rgba(255,255,255,0.3)",
                            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
                          }}
                        >
                          <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                        </div>
                      ))}
                    </div>
                  )}
                  {/* Text bubble */}
                  <div
                    className="px-3.5 py-2.5 text-[13.5px] leading-relaxed whitespace-pre-wrap"
                    style={
                      msg.role === "user"
                        ? { background: "#0d1b67", color: "white", borderRadius: "18px 18px 4px 18px" }
                        : { background: "white", color: "#2f2f2f", borderRadius: "18px 18px 18px 4px", border: "1px solid #ececec", boxShadow: "0 2px 8px rgba(0,0,0,0.05)" }
                    }
                  >
                    {msg.content}
                  </div>
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex gap-2.5 flex-row">
                <div
                  className="flex-shrink-0 flex items-center justify-center rounded-full"
                  style={{ width: 32, height: 32, background: "#015e9f" }}
                >
                  <BotIcon size={16} className="text-white" />
                </div>
                <div
                  className="px-4 py-3"
                  style={{ background: "white", borderRadius: "18px 18px 18px 4px", border: "1px solid #ececec", boxShadow: "0 2px 8px rgba(0,0,0,0.05)" }}
                >
                  <span className="flex gap-1 items-center">
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#015e9f", animationDelay: "0ms" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#015e9f", animationDelay: "150ms" }} />
                    <span className="w-2 h-2 rounded-full animate-bounce" style={{ background: "#015e9f", animationDelay: "300ms" }} />
                  </span>
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          {/* Input bubble */}
          <div className="px-4 pb-4 pt-2 flex-shrink-0" style={{ background: "#f5f5f5" }}>
            <div
              style={{
                background: "#f8f8f8",
                border: "1px solid #ececec",
                borderRadius: 24,
                boxShadow: "0 12px 30px rgba(0,0,0,0.04)",
                padding: "10px 14px 10px 14px",
              }}
            >
              {/* Staged attachment thumbnails */}
              {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2.5">
                  {attachments.map((att) => (
                    <div
                      key={att.imageUrl}
                      className="relative group flex-shrink-0"
                      style={{ width: 72, height: 72 }}
                    >
                      {/* Tooltip */}
                      <div
                        className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 px-2 py-1 rounded-lg text-[11px] font-medium whitespace-nowrap pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-10"
                        style={{
                          background: "rgba(13,27,103,0.92)",
                          color: "white",
                          maxWidth: 180,
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {att.productName}
                      </div>

                      {/* Thumbnail */}
                      <div
                        className="w-full h-full overflow-hidden cursor-pointer"
                        style={{ borderRadius: 14, border: "1.5px solid #d8d8d8" }}
                        onClick={() => removeAttachment(att.imageUrl)}
                      >
                        <img src={att.imageUrl} alt={att.productName} className="w-full h-full object-cover" />
                      </div>

                      {/* Remove X overlay on hover */}
                      <div
                        className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-150 cursor-pointer pointer-events-none group-hover:pointer-events-auto"
                        style={{ borderRadius: 14, background: "rgba(0,0,0,0.45)" }}
                        onClick={() => removeAttachment(att.imageUrl)}
                      >
                        <X size={18} className="text-white" strokeWidth={2.5} />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Text input */}
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={attachments.length > 0 ? "Thêm tin nhắn (tuỳ chọn)..." : "Nhập tin nhắn..."}
                disabled={loading}
                className="w-full text-[14px] bg-transparent outline-none disabled:opacity-50 px-2"
                style={{ color: "#2f2f2f", caretColor: "#015e9f" }}
              />

              {/* Action row */}
              <div className="flex items-center justify-between mt-2 px-1">
                <div className="flex items-center gap-3">
                  <button className="flex items-center justify-center transition-opacity hover:opacity-70" style={{ color: "#434343" }} tabIndex={-1} type="button">
                    <Plus size={22} strokeWidth={1.8} />
                  </button>
                  <button className="flex items-center gap-1.5 transition-opacity hover:opacity-70" style={{ color: "#434343" }} tabIndex={-1} type="button">
                    <Mic size={18} strokeWidth={1.9} />
                  </button>
                </div>
                <button
                  onClick={handleSend}
                  disabled={!canSend}
                  className="flex items-center justify-center rounded-full transition-all disabled:opacity-35 disabled:cursor-not-allowed hover:brightness-110 active:scale-95"
                  style={{ width: 34, height: 34, background: "#015e9f" }}
                >
                  <Send size={14} className="text-white" style={{ marginLeft: 1 }} />
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Toggle button */}
      <button
        onClick={() => setIsOpen((v) => !v)}
        aria-label="Mở/đóng chat hỗ trợ"
        className="fixed bottom-4 right-4 z-50 flex items-center justify-center rounded-full transition-all duration-200 hover:scale-105 active:scale-95"
        style={{
          width: 56,
          height: 56,
          background: isOpen ? "#444956" : "#015e9f",
          boxShadow: "0 6px 20px rgba(1,94,159,0.35)",
        }}
      >
        {isOpen
          ? <X size={22} className="text-white" />
          : (
            <div className="relative">
              <BotIcon size={26} className="text-white" />
              {/* Badge when there are pending attachments and chat is closed */}
              {attachments.length > 0 && (
                <span
                  className="absolute -top-1.5 -right-1.5 flex items-center justify-center rounded-full text-[10px] font-bold text-white"
                  style={{ width: 16, height: 16, background: "#ff4a3a" }}
                >
                  {attachments.length}
                </span>
              )}
            </div>
          )
        }
      </button>
    </>
  );
}
