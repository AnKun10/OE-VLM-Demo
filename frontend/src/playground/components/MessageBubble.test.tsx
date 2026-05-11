import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MessageBubble } from "./MessageBubble";
import type { Message } from "../types";

const stubActions = {
  isStreaming: false,
  editingId: null as string | null,
  onStartEdit: () => {},
  onSaveEdit: () => {},
  onCancelEdit: () => {},
  onRegenerate: () => {},
};

describe("MessageBubble XSS posture (A5.9)", () => {
  it("strips <script> tags and inline event handlers from assistant content", () => {
    const malicious = `Hello

<script>window.__pwned = true</script>

<img src="x" onerror="window.__pwned_img = true" alt="x">

End of content.`;

    const msg: Message = {
      id: "m1",
      role: "assistant",
      text: malicious,
      status: "done",
      createdAt: 0,
    };

    const { container } = render(
      <MessageBubble msg={msg} isLast={true} actions={stubActions} />,
    );

    // 1. No script element rendered.
    expect(container.querySelector("script")).toBeNull();
    // 2. No img with onerror handler preserved (rehype-raw + react drop event handlers).
    expect(container.querySelector("img[onerror]")).toBeNull();
    // 3. No global side-effects from the inert HTML.
    expect((window as unknown as Record<string, unknown>).__pwned).toBeUndefined();
    expect((window as unknown as Record<string, unknown>).__pwned_img).toBeUndefined();
    // 4. The benign markdown content around the payload still renders.
    expect(container.textContent).toContain("Hello");
    expect(container.textContent).toContain("End of content.");
  });

  it("renders the engine's <details> thinking-log block as expandable HTML", () => {
    const text = `<details><summary>🧠 Compressor reasoning</summary>
Step 1 — cache hit
</details>

Actual model answer here.`;

    const msg: Message = {
      id: "m2",
      role: "assistant",
      text,
      status: "done",
      createdAt: 0,
    };

    const { container } = render(
      <MessageBubble msg={msg} isLast={true} actions={stubActions} />,
    );

    // rehype-raw renders <details>/<summary> as real HTML elements.
    expect(container.querySelector("details")).not.toBeNull();
    expect(container.querySelector("summary")?.textContent).toContain(
      "Compressor reasoning",
    );
    // And the model answer after the </details> is rendered too.
    expect(container.textContent).toContain("Actual model answer here.");
  });
});
