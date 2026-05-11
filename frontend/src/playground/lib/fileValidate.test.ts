import { describe, it, expect } from "vitest";
import {
  validateFile,
  checkAttachmentCap,
  ALLOWED_MIME,
  MAX_BYTES,
} from "./fileValidate";
import { FriendlyError } from "./errors";

function makeFile(name: string, mime: string, bytes: number): File {
  const blob = new Blob([new Uint8Array(bytes)], { type: mime });
  return new File([blob], name, { type: mime });
}

describe("validateFile", () => {
  it("T2.8 — accepts PNG/JPEG/WebP/GIF", () => {
    for (const mime of ALLOWED_MIME) {
      expect(() => validateFile(makeFile("a", mime, 100))).not.toThrow();
    }
  });

  it("T2.8 — rejects SVG, PDF, exe", () => {
    for (const mime of ["image/svg+xml", "application/pdf", "application/x-msdownload"]) {
      expect(() => validateFile(makeFile("a", mime, 100))).toThrow(FriendlyError);
    }
  });

  it("T2.9 — rejects > 10MB", () => {
    const f = makeFile("big", "image/png", MAX_BYTES + 1);
    expect(() => validateFile(f)).toThrow(/too_large/);
  });

  it("T2.9 — rejects zero-byte", () => {
    const f = makeFile("empty", "image/png", 0);
    expect(() => validateFile(f)).toThrow(/empty_file/);
  });
});

describe("checkAttachmentCap", () => {
  it("T2.10 (Phase 5) — rejects when current >= MAX_IMAGES (per-turn only)", () => {
    expect(checkAttachmentCap(0)).toBe(true);
    expect(checkAttachmentCap(3)).toBe(true);
    expect(checkAttachmentCap(4)).toBe(false);
    expect(checkAttachmentCap(5)).toBe(false);
  });
});

describe("FriendlyError", () => {
  it("exposes a Vietnamese message via .message", () => {
    const e = new FriendlyError("too_large");
    expect(e.key).toBe("too_large");
    expect(e.message).toMatch(/quá lớn|10/);
  });
});
