import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { uploadFile } from "./uploadFile";
import { FriendlyError } from "./errors";

function makeFile(name: string, mime: string, bytes: number): File {
  const blob = new Blob([new Uint8Array(bytes)], { type: mime });
  return new File([blob], name, { type: mime });
}

describe("uploadFile", () => {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let fetchSpy: ReturnType<typeof vi.spyOn<any, any>>;
  beforeEach(() => {
    fetchSpy = vi.spyOn(globalThis, "fetch");
  });
  afterEach(() => {
    fetchSpy.mockRestore();
  });

  it("posts FormData and returns the AttachmentRef on success", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(
        JSON.stringify({
          id: "abcd1234",
          url: "/api/files/abcd1234",
          mime: "image/png",
          size: 100,
          originalName: "x.png",
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ),
    );
    const result = await uploadFile(makeFile("x.png", "image/png", 100));
    expect(result.id).toBe("abcd1234");
    expect(result.originalName).toBe("x.png");
    expect(fetchSpy).toHaveBeenCalledWith(
      "/api/files",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("validateFile rejects unsupported mime BEFORE network call", async () => {
    await expect(
      uploadFile(makeFile("a.svg", "image/svg+xml", 100)),
    ).rejects.toBeInstanceOf(FriendlyError);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("A2.15 — network failure throws FriendlyError(upload_network)", async () => {
    fetchSpy.mockRejectedValueOnce(new TypeError("Failed to fetch"));
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "upload_network" });
  });

  it("non-OK HTTP throws FriendlyError(upload_http)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response("{}", { status: 413 }),
    );
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "upload_http" });
  });

  it("A2.16 — response missing id throws FriendlyError(invalid_response)", async () => {
    fetchSpy.mockResolvedValueOnce(
      new Response(JSON.stringify({ url: "/api/files/abc" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    await expect(
      uploadFile(makeFile("x.png", "image/png", 100)),
    ).rejects.toMatchObject({ key: "invalid_response" });
  });
});
