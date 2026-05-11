import { test, expect } from "@playwright/test";
import { setupAllMocks } from "./fixtures/sseFixture";

test.describe("Playground", () => {
  test("E2.1 — golden path: type + upload + send → markdown response", async ({ page }) => {
    await setupAllMocks(page);
    await page.goto("/playground");

    // Wait for model dropdown to populate.
    await expect(page.locator("select").first()).toBeVisible();

    // Upload one image via the hidden file input.
    const fileInput = page.locator('input[type="file"]');
    await fileInput.setInputFiles({
      name: "test.png",
      mimeType: "image/png",
      buffer: Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    });

    // Type the prompt.
    const ta = page.locator("textarea");
    await ta.fill("ảnh này là gì");

    // Click Send.
    await page.getByLabel("Gửi").click();

    // The streamed response should render with the **bold** part rendered as bold.
    await expect(page.locator("strong", { hasText: "bold" })).toBeVisible({
      timeout: 5000,
    });
    await expect(page.getByText("Hello", { exact: false })).toBeVisible();
  });

  test("E2.2 — multi-image: upload 3 images → 3 thumbnails in rail", async ({ page }) => {
    await setupAllMocks(page);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    const fileInput = page.locator('input[type="file"]');
    const png = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    await fileInput.setInputFiles([
      { name: "a.png", mimeType: "image/png", buffer: png },
      { name: "b.png", mimeType: "image/png", buffer: png },
      { name: "c.png", mimeType: "image/png", buffer: png },
    ]);

    // Three preview thumbnails should render.
    await expect(page.getByRole("button", { name: /Xoá/ })).toHaveCount(3, {
      timeout: 5000,
    });

    // Send works with text-only message.
    await page.locator("textarea").fill("so sánh 3 ảnh này");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Hello", { exact: false })).toBeVisible();
  });
});
