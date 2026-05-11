import { Square } from "lucide-react";

/**
 * Compact stop button used by ComposerBar in place of Send while a
 * streaming response is in progress. Rendered as a red square — the
 * universal "stop generation" affordance used by ChatGPT, Claude, and
 * Open WebUI.
 */
export function StopButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center justify-center rounded-lg transition-all"
      style={{
        width: 36,
        height: 36,
        background: "#dc2626",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "#b91c1c")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "#dc2626")}
      aria-label="Dừng"
      title="Dừng phản hồi"
    >
      <Square size={13} className="text-white" fill="currentColor" />
    </button>
  );
}
