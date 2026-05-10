import type { Page, Route } from "@playwright/test";

const enc = new TextEncoder();
function sseFrame(payload: object): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/**
 * Mock /api/models with 1 vision-capable model.
 */
export async function mockModels(page: Page) {
  await page.route("**/api/models", (route: Route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        models: [
          {
            id: "qwen3-vl-8b-vllm",
            name: "Qwen3-VL 8B (vLLM)",
            capabilities: { vision: true },
          },
        ],
      }),
    }),
  );
}

/**
 * Mock /api/files: returns a deterministic AttachmentRef per request.
 */
export async function mockFileUploads(page: Page) {
  let counter = 0;
  await page.route("**/api/files", (route: Route) => {
    counter++;
    const id = `aaaa${String(counter).padStart(28, "0")}`; // 32 hex
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        id,
        url: `/api/files/${id}`,
        mime: "image/png",
        size: 100,
        originalName: `mock-${counter}.png`,
      }),
    });
  });
}

/**
 * Mock /api/chat/stream to emit a fixed sequence of SSE frames.
 */
export async function mockChatStream(
  page: Page,
  deltas: string[] = ["Hello ", "**bold** ", "world."],
) {
  await page.route("**/api/chat/stream", (route: Route) => {
    const body =
      deltas.map((d) => sseFrame({ delta: d, done: false })).join("") +
      sseFrame({ delta: "", done: true });
    route.fulfill({
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
      body,
    });
  });
}

export async function setupAllMocks(page: Page) {
  await mockModels(page);
  await mockFileUploads(page);
  await mockChatStream(page);
}
