import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";

export type StatusBannerState = {
  message: string;
  done: boolean;
};

type Props = {
  status: StatusBannerState | null;
  /** Fires 1500ms after a `done:true` status arrives. The parent uses this
   * to dismiss the banner. */
  onClear?: () => void;
};

const AUTO_CLEAR_MS = 1500;

export function StatusBanner({ status, onClear }: Props) {
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    // On every status change: cancel any pending timer.
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (status?.done && onClear) {
      timerRef.current = window.setTimeout(() => {
        timerRef.current = null;
        onClear();
      }, AUTO_CLEAR_MS);
    }
    return () => {
      if (timerRef.current !== null) {
        window.clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [status, onClear]);

  if (!status) return null;

  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 mx-auto mt-3 px-3 py-1.5 rounded-full text-xs"
      style={{
        background: "#eff6ff",
        color: "#1e40af",
        border: "1px solid #bfdbfe",
        width: "fit-content",
        maxWidth: "90%",
      }}
    >
      {!status.done && (
        <Loader2 size={12} className="animate-spin" aria-hidden="true" />
      )}
      <span>{status.message}</span>
    </div>
  );
}
