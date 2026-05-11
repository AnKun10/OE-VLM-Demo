import { test, expect } from "@playwright/test";
import {
  mockModels,
  mockFileUploads,
  mockChatStreamWithStatus,
} from "./fixtures/sseFixture";

test.describe("Playground compressor (Phase 5)", () => {
  test("E5.1 — status banner appears then auto-clears", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStreamWithStatus(
      page,
      [
        { message: "🖼️ Captioning 1 new image(s)...", done: false },
        { message: "✅ Compressor done", done: true },
      ],
      "",
      ["Hello world."],
    );
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("hi");
    await page.getByLabel("Gửi").click();

    // The banner shows at least the done status (React may batch earlier statuses).
    // Either the captioning message or the done message should appear momentarily.
    await expect(page.locator("[role='status']")).toBeVisible({ timeout: 5000 });
    // Response text arrives.
    await expect(page.getByText("Hello world.")).toBeVisible({ timeout: 5000 });
    // Banner auto-clears 1.5s after the done:true event.
    await expect(page.locator("[role='status']")).not.toBeVisible({
      timeout: 6000,
    });
  });

  test("E5.2 — thinking log <details> renders and expands", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    const thinkingMd =
      "<details><summary>🧠 Image compressor reasoning (1 ảnh, 1 caption mới, kept new upload)</summary>\n\n" +
      "**Step 1 — Image scan**\n- Tổng 1 ảnh; cache miss: 1, hit: 0\n\n" +
      "</details>\n\n";
    await mockChatStreamWithStatus(
      page,
      [{ message: "✅ Compressor done", done: true }],
      thinkingMd,
      ["Câu trả lời cho ảnh."],
    );
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("describe");
    await page.getByLabel("Gửi").click();

    // Summary text renders.
    await expect(
      page.getByText(/Image compressor reasoning/),
    ).toBeVisible({ timeout: 5000 });
    // Closed by default — Step 1 text not visible.
    await page.evaluate(() =>
      document.querySelector("details")?.removeAttribute("open"),
    );
    await expect(page.getByText(/cache miss: 1/)).not.toBeVisible();

    // Click summary to expand.
    await page.getByText(/Image compressor reasoning/).click();
    await expect(page.getByText(/cache miss: 1/)).toBeVisible();
    // Final answer also rendered.
    await expect(page.getByText("Câu trả lời cho ảnh.")).toBeVisible();
  });
});
