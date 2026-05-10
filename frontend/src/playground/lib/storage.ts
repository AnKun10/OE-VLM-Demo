import type { ConversationsState } from "../types";

const STORAGE_KEY = "playground.conversations.v1";

export function readState(): ConversationsState | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    if (parsed.schemaVersion !== 1) return null;
    if (!parsed.conversations || typeof parsed.conversations !== "object") return null;
    return parsed as ConversationsState;
  } catch (e) {
    console.warn("[storage] parse failed; starting fresh", e);
    return null;
  }
}

export class StorageQuotaError extends Error {
  constructor() {
    super("Storage quota exceeded");
    this.name = "StorageQuotaError";
  }
}

export function writeState(state: ConversationsState): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    if (
      e instanceof DOMException &&
      (e.name === "QuotaExceededError" || e.code === 22)
    ) {
      throw new StorageQuotaError();
    }
    throw e;
  }
}

export const __TEST__ = { STORAGE_KEY };
