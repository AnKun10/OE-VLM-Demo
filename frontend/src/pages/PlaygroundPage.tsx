import { useState, useRef, useEffect } from "react";
import { Link } from "react-router-dom";
import {
  Send,
  Plus,
  Mic,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  PanelLeft,
  Sparkles,
  ImagePlus,
  X,
  ChevronDown,
} from "lucide-react";

/* ───────────────────── Types ───────────────────── */

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  images?: { url: string; name: string }[];
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
}

/* ───────────────────── API ───────────────────── */

async function chatAPI(
  message: string,
  history: { role: string; content: string }[],
  imageUrls: string[] = [],
  modelId: string = "",
): Promise<{ reply: string }> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      history,
      image_urls: imageUrls,
      model_id: modelId || undefined,
    }),
  });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

/* ───────────────────── Constants ───────────────────── */

const SURFACE = "#f9fafb";       // gray-50 – matches bg-gray-50 on other pages
const SIDEBAR_BG = "#ffffff";
const CARD = "#f3f4f6";          // gray-100
const BORDER = "#e5e7eb";        // gray-200
const ACCENT = "#015e9f";        // RunShop blue (from chatbot header)
const ACCENT_HOVER = "#01497a";
const TEXT_PRIMARY = "#111827";   // gray-900
const TEXT_SECONDARY = "#6b7280"; // gray-500
const TEXT_MUTED = "#9ca3af";     // gray-400
const INPUT_BG = "#ffffff";

const WELCOME_MSG: Message = {
  id: "welcome",
  role: "assistant",
  content:
    "Xin chào! Tôi là mô hình AI của RunShop. Bạn có thể gửi văn bản hoặc hình ảnh để kiểm tra khả năng của tôi. Hãy thử ngay!",
  timestamp: new Date(),
};

/* ───────────────────── Component ───────────────────── */

export default function PlaygroundPage() {
  /* State */
  const [conversations, setConversations] = useState<Conversation[]>(() => [
    { id: "1", title: "Cuộc hội thoại mới", messages: [WELCOME_MSG], createdAt: new Date() },
  ]);
  const [activeId, setActiveId] = useState("1");
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [attachments, setAttachments] = useState<{ url: string; name: string }[]>([]);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const [models, setModels] = useState<{ id: string; name: string }[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");

  useEffect(() => {
    fetch("/api/models")
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models);
        if (data.models.length > 0) setSelectedModel(data.models[0].id);
      })
      .catch(() => {});
  }, []);

  const active = conversations.find((c) => c.id === activeId)!;
  const messages = active.messages;

  /* Auto‑scroll */
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  /* Auto‑resize textarea */
  useEffect(() => {
    const ta = inputRef.current;
    if (ta) {
      ta.style.height = "auto";
      ta.style.height = Math.min(ta.scrollHeight, 160) + "px";
    }
  }, [input]);

  /* Helpers */
  function uid() {
    return Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  }

  function updateConvo(id: string, patch: Partial<Conversation>) {
    setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, ...patch } : c)));
  }

  function newConversation() {
    const c: Conversation = {
      id: uid(),
      title: "Cuộc hội thoại mới",
      messages: [{ ...WELCOME_MSG, id: uid() }],
      createdAt: new Date(),
    };
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
    setInput("");
    setAttachments([]);
  }

  function deleteConversation(id: string) {
    setConversations((prev) => {
      const next = prev.filter((c) => c.id !== id);
      if (next.length === 0) {
        const fresh: Conversation = {
          id: uid(),
          title: "Cuộc hội thoại mới",
          messages: [{ ...WELCOME_MSG, id: uid() }],
          createdAt: new Date(),
        };
        setActiveId(fresh.id);
        return [fresh];
      }
      if (id === activeId) setActiveId(next[0].id);
      return next;
    });
  }

  /* Send */
  async function handleSend() {
    const text = input.trim();
    if ((!text && attachments.length === 0) || loading) return;

    const apiContent = text || "Hãy mô tả hình ảnh này.";
    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: text,
      images: attachments.length > 0 ? attachments : undefined,
      timestamp: new Date(),
    };

    const newMessages = [...messages, userMsg];
    updateConvo(activeId, { messages: newMessages });

    // Auto‑title from first real user message
    if (messages.filter((m) => m.role === "user").length === 0) {
      const titleText = text || "Hình ảnh";
      updateConvo(activeId, { title: titleText.slice(0, 40) + (titleText.length > 40 ? "…" : "") });
    }

    setInput("");
    setAttachments([]);
    setLoading(true);

    try {
      const history = newMessages.map((m) => ({ role: m.role, content: m.content || "Hãy mô tả hình ảnh này." }));
      const imageUrls = userMsg.images?.map((img) => img.url) ?? [];
      const data = await chatAPI(apiContent, history, imageUrls, selectedModel);
      const assistantMsg: Message = {
        id: uid(),
        role: "assistant",
        content: data.reply,
        timestamp: new Date(),
      };
      updateConvo(activeId, { messages: [...newMessages, assistantMsg] });
    } catch {
      updateConvo(activeId, {
        messages: [
          ...newMessages,
          { id: uid(), role: "assistant", content: "Chatbot is currently unavailable!", timestamp: new Date() },
        ],
      });
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

  function handleFileSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setAttachments((prev) => [...prev, { url: reader.result as string, name: file.name }]);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  }

  const canSend = (input.trim().length > 0 || attachments.length > 0) && !loading;

  /* ───────────────────── Render ───────────────────── */
  return (
    <div className="flex h-screen overflow-hidden" style={{ background: SURFACE, color: TEXT_PRIMARY }}>
      {/* ── Sidebar ── */}
      <aside
        className="flex flex-col flex-shrink-0 transition-all duration-300 overflow-hidden"
        style={{
          width: sidebarOpen ? 260 : 0,
          background: SIDEBAR_BG,
          borderRight: sidebarOpen ? `1px solid ${BORDER}` : "none",
        }}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between px-4 h-14 flex-shrink-0" style={{ borderBottom: `1px solid ${BORDER}` }}>
          <Link to="/" className="flex items-center gap-2 text-sm font-semibold" style={{ color: TEXT_PRIMARY }}>
            <div
              className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold text-white"
              style={{ background: ACCENT }}
            >
              RS
            </div>
            <span style={{ fontFamily: "'DM Sans', sans-serif", letterSpacing: "-0.02em" }}>AI Playground</span>
          </Link>
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 rounded transition-colors"
            style={{ color: TEXT_SECONDARY }}
            onMouseEnter={(e) => (e.currentTarget.style.background = CARD)}
            onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
          >
            <PanelLeftClose size={18} />
          </button>
        </div>

        {/* New chat button */}
        <div className="px-3 pt-3 pb-1">
          <button
            onClick={newConversation}
            className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all"
            style={{ background: ACCENT, color: "#fff" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = ACCENT_HOVER)}
            onMouseLeave={(e) => (e.currentTarget.style.background = ACCENT)}
          >
            <Plus size={16} />
            Cuộc trò chuyện mới
          </button>
        </div>

        {/* Conversation list */}
        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-0.5">
          {conversations.map((c) => (
            <div
              key={c.id}
              className="group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors text-sm"
              style={{
                background: c.id === activeId ? CARD : "transparent",
                color: c.id === activeId ? TEXT_PRIMARY : TEXT_SECONDARY,
              }}
              onClick={() => {
                setActiveId(c.id);
                setAttachments([]);
              }}
              onMouseEnter={(e) => {
                if (c.id !== activeId) e.currentTarget.style.background = "#f9fafb";
              }}
              onMouseLeave={(e) => {
                if (c.id !== activeId) e.currentTarget.style.background = "transparent";
              }}
            >
              <MessageSquare size={14} className="flex-shrink-0" style={{ opacity: 0.6 }} />
              <span className="flex-1 truncate">{c.title}</span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  deleteConversation(c.id);
                }}
                className="opacity-0 group-hover:opacity-100 p-1 rounded transition-all"
                style={{ color: TEXT_MUTED }}
                onMouseEnter={(e) => (e.currentTarget.style.color = "#dc2626")}
                onMouseLeave={(e) => (e.currentTarget.style.color = TEXT_MUTED)}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>

        {/* Sidebar footer */}
        <div className="px-4 py-3 text-[11px] flex-shrink-0" style={{ borderTop: `1px solid ${BORDER}`, color: TEXT_MUTED }}>
          Powered by OE-VLM
        </div>
      </aside>

      {/* ── Main area ── */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <header
          className="flex items-center gap-3 px-4 h-14 flex-shrink-0"
          style={{ borderBottom: `1px solid ${BORDER}` }}
        >
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1.5 rounded transition-colors"
              style={{ color: TEXT_SECONDARY }}
              onMouseEnter={(e) => (e.currentTarget.style.background = CARD)}
              onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
            >
              <PanelLeft size={18} />
            </button>
          )}
          <div className="flex items-center gap-2">
            <Sparkles size={16} style={{ color: ACCENT }} />
            <span className="text-sm font-medium" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              {active.title}
            </span>
            <ChevronDown size={14} style={{ color: TEXT_MUTED }} />
          </div>
          <div className="flex-1" />
          <Link
            to="/products"
            className="text-xs px-3 py-1.5 rounded-md transition-colors font-medium"
            style={{ background: CARD, color: TEXT_SECONDARY, border: `1px solid ${BORDER}` }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = BORDER;
              e.currentTarget.style.color = TEXT_PRIMARY;
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = CARD;
              e.currentTarget.style.color = TEXT_SECONDARY;
            }}
          >
            Quay lại cửa hàng
          </Link>
        </header>

        {/* Messages area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto px-4 py-8 space-y-6">
            {/* Empty state */}
            {messages.length <= 1 && messages[0]?.id === "welcome" && (
              <div className="flex flex-col items-center justify-center pt-16 pb-8">
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center mb-6"
                  style={{ background: "rgba(1,94,159,0.12)", border: "1px solid rgba(1,94,159,0.18)" }}
                >
                  <Sparkles size={28} style={{ color: ACCENT }} />
                </div>
                <h2
                  className="text-2xl font-semibold mb-2"
                  style={{ fontFamily: "'DM Sans', sans-serif", letterSpacing: "-0.03em" }}
                >
                  Tôi có thể giúp gì cho bạn?
                </h2>
                <p className="text-sm text-center max-w-md" style={{ color: TEXT_SECONDARY, lineHeight: 1.6 }}>
                  Gửi tin nhắn hoặc hình ảnh để bắt đầu trò chuyện với mô hình AI.
                  Bạn có thể kiểm tra khả năng phân tích hình ảnh, trả lời câu hỏi, và nhiều hơn nữa.
                </p>

                {/* Suggestion chips */}
                <div className="flex flex-wrap gap-2 mt-8 justify-center max-w-lg">
                  {[
                    "Giới thiệu về bạn",
                    "Phân tích hình ảnh sản phẩm",
                    "Tư vấn giày chạy bộ",
                    "So sánh hai đôi giày",
                  ].map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => {
                        setInput(suggestion);
                        inputRef.current?.focus();
                      }}
                      className="px-4 py-2 rounded-full text-sm transition-all"
                      style={{
                        background: CARD,
                        border: `1px solid ${BORDER}`,
                        color: TEXT_SECONDARY,
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.borderColor = ACCENT;
                        e.currentTarget.style.color = TEXT_PRIMARY;
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.borderColor = BORDER;
                        e.currentTarget.style.color = TEXT_SECONDARY;
                      }}
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            {messages
              .filter((m) => m.id !== "welcome" || messages.length > 1)
              .map((msg) =>
                msg.role === "user" ? (
                  /* ── User message: right-aligned bubble ── */
                  <div key={msg.id} className="flex flex-col items-end gap-1">
                    {/* Attached images */}
                    {msg.images && msg.images.length > 0 && (
                      <div className="flex flex-wrap gap-2 justify-end" style={{ maxWidth: "85%" }}>
                        {msg.images.map((img, i) => (
                          <div
                            key={i}
                            className="overflow-hidden"
                            style={{
                              width: 120,
                              height: 120,
                              borderRadius: 12,
                              border: `1px solid ${BORDER}`,
                            }}
                          >
                            <img src={img.url} alt={img.name} className="w-full h-full object-cover" />
                          </div>
                        ))}
                      </div>
                    )}
                    {msg.content && (
                      <div
                        className="text-[16px] leading-relaxed whitespace-pre-wrap"
                        style={{
                          background: "#0d1b67",
                          color: "#ffffff",
                          borderRadius: "18px 18px 4px 18px",
                          padding: "10px 16px",
                          maxWidth: "85%",
                        }}
                      >
                        {msg.content}
                      </div>
                    )}
                  </div>
                ) : (
                  /* ── Assistant message: left-aligned with avatar ── */
                  <div key={msg.id} className="flex gap-3">
                    <div className="flex-shrink-0 pt-0.5">
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center"
                        style={{ background: "rgba(1,94,159,0.15)" }}
                      >
                        <Sparkles size={15} style={{ color: ACCENT }} />
                      </div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <span className="text-xs font-medium" style={{ color: TEXT_MUTED }}>
                        AI Model
                      </span>
                      <div
                        className="mt-1 text-[16px] leading-relaxed whitespace-pre-wrap"
                        style={{ color: TEXT_PRIMARY }}
                      >
                        {msg.content}
                      </div>
                    </div>
                  </div>
                ),
              )}

            {/* Loading indicator */}
            {loading && (
              <div className="flex gap-3">
                <div
                  className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: "rgba(1,94,159,0.15)" }}
                >
                  <Sparkles size={15} style={{ color: ACCENT }} />
                </div>
                <div className="pt-2">
                  <div className="flex gap-1.5">
                    {[0, 1, 2].map((i) => (
                      <span
                        key={i}
                        className="w-2 h-2 rounded-full animate-bounce"
                        style={{ background: ACCENT, animationDelay: `${i * 150}ms` }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </div>

        {/* ── Input area ── */}
        <div className="flex-shrink-0 px-4 pb-5 pt-2">
          <div className="max-w-3xl mx-auto">
            <div
              className="rounded-2xl overflow-hidden transition-all"
              style={{
                background: INPUT_BG,
                border: `1px solid ${BORDER}`,
                boxShadow: "0 4px 24px rgba(0,0,0,0.06)",
              }}
            >
              {/* Staged attachments */}
              {attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 px-4 pt-3">
                  {attachments.map((att) => (
                    <div key={att.url} className="relative group" style={{ width: 80, height: 80 }}>
                      <div
                        className="w-full h-full overflow-hidden"
                        style={{ borderRadius: 10, border: `1px solid ${BORDER}` }}
                      >
                        <img src={att.url} alt={att.name} className="w-full h-full object-cover" />
                      </div>
                      <button
                        onClick={() => setAttachments((p) => p.filter((a) => a.url !== att.url))}
                        className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                        style={{ background: "#dc2626" }}
                      >
                        <X size={11} className="text-white" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Textarea */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Nhập tin nhắn..."
                disabled={loading}
                rows={1}
                className="w-full resize-none bg-transparent outline-none text-sm px-4 pt-3.5 pb-1 disabled:opacity-50"
                style={{ color: TEXT_PRIMARY, caretColor: ACCENT, maxHeight: 160 }}
              />

              {/* Action row */}
              <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
                <div className="flex items-center gap-1">
                  <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFileSelect} />
                  <button
                    onClick={() => fileRef.current?.click()}
                    className="p-2 rounded-lg transition-colors"
                    style={{ color: TEXT_MUTED }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = `${BORDER}`;
                      e.currentTarget.style.color = TEXT_SECONDARY;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                      e.currentTarget.style.color = TEXT_MUTED;
                    }}
                  >
                    <ImagePlus size={18} />
                  </button>
                  <button
                    className="p-2 rounded-lg transition-colors"
                    style={{ color: TEXT_MUTED }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = `${BORDER}`;
                      e.currentTarget.style.color = TEXT_SECONDARY;
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = "transparent";
                      e.currentTarget.style.color = TEXT_MUTED;
                    }}
                  >
                    <Mic size={18} />
                  </button>
                  {models.length > 0 && (
                    <select
                      value={selectedModel}
                      onChange={(e) => setSelectedModel(e.target.value)}
                      className="text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer transition-colors"
                      style={{
                        color: TEXT_SECONDARY,
                        background: "transparent",
                        border: `1px solid ${BORDER}`,
                        maxWidth: 160,
                      }}
                    >
                      {models.map((m) => (
                        <option key={m.id} value={m.id}>
                          {m.name}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <button
                  onClick={handleSend}
                  disabled={!canSend}
                  className="flex items-center justify-center rounded-lg transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                  style={{
                    width: 36,
                    height: 36,
                    background: canSend ? ACCENT : TEXT_MUTED,
                  }}
                  onMouseEnter={(e) => {
                    if (canSend) e.currentTarget.style.background = ACCENT_HOVER;
                  }}
                  onMouseLeave={(e) => {
                    if (canSend) e.currentTarget.style.background = ACCENT;
                  }}
                >
                  <Send size={15} className="text-white" style={{ marginLeft: 1 }} />
                </button>
              </div>
            </div>

            <p className="text-center mt-2.5 text-[11px]" style={{ color: TEXT_MUTED }}>
              AI Playground sử dụng các mô hình AI. Kết quả có thể không chính xác.
            </p>
          </div>
        </div>
      </div>

      {/* ── Scrollbar override ── */}
      <style>{`
        .overflow-y-auto::-webkit-scrollbar { width: 5px; }
        .overflow-y-auto::-webkit-scrollbar-track { background: transparent; }
        .overflow-y-auto::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 4px; }
        .overflow-y-auto::-webkit-scrollbar-thumb:hover { background: #9ca3af; }
      `}</style>
    </div>
  );
}
