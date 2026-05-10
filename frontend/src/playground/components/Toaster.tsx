import {
  Provider as ToastProvider,
  Root as ToastRoot,
  Title as ToastTitle,
  Viewport as ToastViewport,
} from "@radix-ui/react-toast";
import { createContext, useCallback, useState, type ReactNode } from "react";

export type ToastItem = {
  id: number;
  title: string;
  variant: "info" | "error";
};

type Ctx = { push: (title: string, variant?: ToastItem["variant"]) => void };

export const ToastContext = createContext<Ctx>({ push: () => {} });

export function Toaster({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<ToastItem[]>([]);

  const push = useCallback(
    (title: string, variant: ToastItem["variant"] = "info") => {
      setItems((prev) => [...prev, { id: Date.now() + Math.random(), title, variant }]);
    },
    [],
  );

  const dismiss = useCallback((id: number) => {
    setItems((prev) => prev.filter((t) => t.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      <ToastProvider swipeDirection="right" duration={4000}>
        {children}
        {items.map((t) => (
          <ToastRoot
            key={t.id}
            onOpenChange={(open) => {
              if (!open) dismiss(t.id);
            }}
            className="rounded-lg px-4 py-3 shadow-lg text-sm"
            style={{
              background: t.variant === "error" ? "#fee2e2" : "#ffffff",
              color: t.variant === "error" ? "#991b1b" : "#111827",
              border: `1px solid ${t.variant === "error" ? "#fecaca" : "#e5e7eb"}`,
            }}
          >
            <ToastTitle>{t.title}</ToastTitle>
          </ToastRoot>
        ))}
        <ToastViewport
          className="fixed bottom-4 right-4 flex flex-col gap-2 outline-none z-50"
          style={{ width: 320 }}
        />
      </ToastProvider>
    </ToastContext.Provider>
  );
}
