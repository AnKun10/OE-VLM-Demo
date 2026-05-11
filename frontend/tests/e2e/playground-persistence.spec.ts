import { test, expect } from "@playwright/test";
import {
  mockModels,
  mockFileUploads,
  mockChatStream,
} from "./fixtures/sseFixture";

const STORAGE_KEY = "playground.conversations.v1";

test.describe("Playground persistence", () => {
  test("E4.1 — Send 2 messages → reload → both bubbles still visible", async ({
    page,
  }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStream(page, ["Hello ", "world."]);

    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("đây là tin nhắn đầu tiên");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Hello world.")).toBeVisible({ timeout: 5000 });

    // Wait for the debounced storage write to flush.
    await page.waitForFunction(
      (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return false;
        try {
          const parsed = JSON.parse(raw);
          const conv = Object.values(parsed.conversations)[0] as {
            messages: { text: string }[];
          };
          return conv?.messages.some((m) => m.text.includes("Hello world."));
        } catch {
          return false;
        }
      },
      STORAGE_KEY,
      { timeout: 5000 },
    );

    await page.reload();
    await expect(page.locator("select").first()).toBeVisible();

    // Both the user prompt and the assistant reply should re-render.
    // (Multiple matches expected: sidebar title, header, message bubble.)
    await expect(
      page.getByText("đây là tin nhắn đầu tiên").first(),
    ).toBeVisible();
    await expect(page.getByText("Hello world.")).toBeVisible();
  });

  test("E4.2 — Two conversations show in sidebar", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStream(page, ["reply."]);

    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    // Send in default conversation.
    await page.locator("textarea").fill("first conv title");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("reply.").first()).toBeVisible({
      timeout: 5000,
    });

    // Click "Cuộc trò chuyện mới".
    await page.getByRole("button", { name: /Cuộc trò chuyện mới/ }).click();

    // Send in the new conversation.
    await page.locator("textarea").fill("second conv title");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("reply.").first()).toBeVisible({
      timeout: 5000,
    });

    // Sidebar should now have at least 2 entries (titles match the first
    // user message).
    await expect(
      page.locator("aside").getByText("first conv title"),
    ).toBeVisible();
    await expect(
      page.locator("aside").getByText("second conv title"),
    ).toBeVisible();
  });

  test("E4.3 — Delete conversation removes from sidebar and localStorage", async ({
    page,
  }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStream(page, ["reply."]);

    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    // Send in default conv.
    await page.locator("textarea").fill("convA");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("reply.").first()).toBeVisible({
      timeout: 5000,
    });

    // New conv + send.
    await page.getByRole("button", { name: /Cuộc trò chuyện mới/ }).click();
    await page.locator("textarea").fill("convB");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("reply.").first()).toBeVisible({
      timeout: 5000,
    });

    // Delete convA from the sidebar (the row is a `.group` div containing
    // the title text + the delete button).
    const convARow = page
      .locator("aside .group")
      .filter({ hasText: "convA" });
    await convARow.hover();
    await convARow
      .getByRole("button", { name: "Xoá cuộc trò chuyện" })
      .click();

    // Wait for the debounced write to settle, then verify localStorage no
    // longer contains "convA" as a title.
    await page.waitForFunction(
      (key) => {
        const raw = localStorage.getItem(key);
        if (!raw) return false;
        try {
          const parsed = JSON.parse(raw);
          const titles = Object.values(parsed.conversations).map(
            (c) => (c as { title: string }).title,
          );
          return !titles.includes("convA") && titles.includes("convB");
        } catch {
          return false;
        }
      },
      STORAGE_KEY,
      { timeout: 5000 },
    );

    await expect(page.locator("aside").getByText("convA")).toHaveCount(0);
    await expect(page.locator("aside").getByText("convB")).toBeVisible();
  });
});
