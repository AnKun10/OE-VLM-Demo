import { test, expect } from "@playwright/test";
import {
  mockModels,
  mockFileUploads,
  mockChatStream,
  mockChatStreamSequence,
} from "./fixtures/sseFixture";

test.describe("Playground controls", () => {
  test("E3.1 — Send → Regenerate replaces last reply", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    // First send → "ALPHA reply.", regenerate → "BETA reply."
    await mockChatStreamSequence(page, [
      ["ALPHA ", "reply."],
      ["BETA ", "reply."],
    ]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("hi");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("ALPHA reply.")).toBeVisible({ timeout: 5000 });

    // Regenerate appears on the last assistant bubble.
    await page.getByRole("button", { name: /Tạo lại/ }).click();
    await expect(page.getByText("BETA reply.")).toBeVisible({ timeout: 5000 });
    // The first reply text should NOT remain in the DOM.
    await expect(page.getByText("ALPHA reply.")).not.toBeVisible();
  });

  test("E3.2 — Edit user msg → Save truncates everything after + regenerates", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStreamSequence(page, [
      ["First ", "reply."],
      ["Second ", "reply."],
      ["EDITED ", "reply."],
    ]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    // Send first message + wait for reply.
    await page.locator("textarea").fill("first user msg");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("First reply.")).toBeVisible({ timeout: 5000 });

    // Send second message + wait for reply.
    await page.locator("textarea").fill("second user msg");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Second reply.")).toBeVisible({ timeout: 5000 });

    // Hover the FIRST user bubble to reveal Edit button.
    // Scope to the actual bubble div (whitespace-pre-wrap) to avoid strict-mode
    // violations from sidebar/header elements that repeat the text.
    const firstBubble = page
      .locator("div.whitespace-pre-wrap", { hasText: "first user msg" })
      .first();
    await firstBubble.hover();
    await page.getByRole("button", { name: "Chỉnh sửa" }).first().click();

    // Editor textarea is focused; replace text and save.
    const editorTa = page.getByLabel("Chỉnh sửa tin nhắn");
    await editorTa.fill("first user msg EDITED");
    await page.getByRole("button", { name: "Lưu" }).click();

    // Second user msg + Second reply should be gone; new EDITED reply should appear.
    await expect(page.getByText("EDITED reply.")).toBeVisible({ timeout: 5000 });
    await expect(page.getByText("Second reply.")).not.toBeVisible();
    await expect(page.getByText("second user msg")).not.toBeVisible();
  });

  test("E3.3 — Edit + Esc reverts without changes", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    await mockChatStream(page, ["Hello ", "world."]);
    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("original");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("Hello world.")).toBeVisible({ timeout: 5000 });

    // Scope to the actual bubble div to avoid strict-mode violations from
    // sidebar/header elements that repeat the text.
    const userBubble = page
      .locator("div.whitespace-pre-wrap", { hasText: "original" })
      .first();
    await userBubble.hover();
    await page.getByRole("button", { name: "Chỉnh sửa" }).first().click();

    const editorTa = page.getByLabel("Chỉnh sửa tin nhắn");
    await editorTa.fill("changed text");
    await editorTa.press("Escape");

    // Original text should still be visible; edited text should not.
    await expect(
      page.locator("div.whitespace-pre-wrap", { hasText: "original" }).first(),
    ).toBeVisible();
    await expect(page.getByText("changed text")).not.toBeVisible();
    // Hello world reply still present.
    await expect(page.getByText("Hello world.")).toBeVisible();
  });

  test("A3.7 — Regenerate when network is down → error bubble + Thử lại button", async ({ page }) => {
    await mockModels(page);
    await mockFileUploads(page);
    // First send: succeeds. Then we'll switch the route to fail for regenerate.
    let succeed = true;
    await page.route("**/api/chat/stream", (route) => {
      if (succeed) {
        succeed = false;
        route.fulfill({
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
          body:
            `data: {"delta":"hi","done":false}\n\n` +
            `data: {"delta":"","done":true}\n\n`,
        });
      } else {
        route.abort("failed");
      }
    });

    await page.goto("/playground");
    await expect(page.locator("select").first()).toBeVisible();

    await page.locator("textarea").fill("ping");
    await page.getByLabel("Gửi").click();
    await expect(page.getByText("hi")).toBeVisible({ timeout: 5000 });

    // Click Regenerate → second call aborts → bubble shows "Thử lại".
    await page.getByRole("button", { name: /Tạo lại/ }).click();
    await expect(page.getByRole("button", { name: "Thử lại" })).toBeVisible({
      timeout: 5000,
    });
  });
});
