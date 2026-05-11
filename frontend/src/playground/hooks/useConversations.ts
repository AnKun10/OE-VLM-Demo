import { useEffect, useReducer, useRef, type Dispatch } from "react";
import {
  conversationsReducer,
  initialState,
  type Action,
} from "../lib/messageReducer";
import { readState, writeState, StorageQuotaError } from "../lib/storage";
import type { ConversationsState } from "../types";
import { useToast } from "./useToast";

const WRITE_DEBOUNCE_MS = 250;

function hydratedInit(): ConversationsState {
  const stored = readState();
  if (!stored) return initialState();
  // Reuse the reducer's HYDRATE path so streaming → stopped coercion is applied.
  return conversationsReducer(initialState(), { type: "HYDRATE", state: stored });
}

export function useConversations(): {
  state: ConversationsState;
  dispatch: Dispatch<Action>;
} {
  const [state, dispatch] = useReducer(
    conversationsReducer,
    undefined,
    hydratedInit,
  );
  const toast = useToast();
  const timerRef = useRef<number | null>(null);
  const quotaWarnedRef = useRef(false);

  useEffect(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
    }
    timerRef.current = window.setTimeout(() => {
      try {
        writeState(state);
      } catch (e) {
        if (e instanceof StorageQuotaError) {
          if (!quotaWarnedRef.current) {
            quotaWarnedRef.current = true;
            toast.push("Bộ nhớ đầy. Hãy xoá cuộc trò chuyện cũ.", "error");
          }
        } else {
          console.error("[useConversations] write failed", e);
        }
      }
    }, WRITE_DEBOUNCE_MS);
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, [state, toast]);

  return { state, dispatch };
}
