import { useEffect, useState } from "react";
import type { ModelInfo } from "../types";

export function useModels(): {
  models: ModelInfo[];
  loading: boolean;
  error: string | null;
} {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/models")
      .then((r) => r.json())
      .then((body) => {
        if (cancelled) return;
        const list = Array.isArray(body?.models) ? body.models : [];
        setModels(list as ModelInfo[]);
        setLoading(false);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e));
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { models, loading, error };
}
