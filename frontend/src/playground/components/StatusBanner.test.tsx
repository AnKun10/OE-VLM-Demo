import { render, screen, act } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { StatusBanner } from "./StatusBanner";

describe("StatusBanner", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("T5.5a — renders message with spinner when status is pending", () => {
    render(<StatusBanner status={{ message: "Captioning...", done: false }} />);
    expect(screen.getByText("Captioning...")).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("T5.5b — calls onClear after 1500ms when done=true", () => {
    const onClear = vi.fn();
    render(
      <StatusBanner
        status={{ message: "Done", done: true }}
        onClear={onClear}
      />,
    );
    expect(onClear).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1499));
    expect(onClear).not.toHaveBeenCalled();
    act(() => vi.advanceTimersByTime(1));
    expect(onClear).toHaveBeenCalledTimes(1);
  });

  it("T5.5c — replacing status before timeout cancels the prior timer", () => {
    const onClear = vi.fn();
    const { rerender } = render(
      <StatusBanner
        status={{ message: "Done A", done: true }}
        onClear={onClear}
      />,
    );
    act(() => vi.advanceTimersByTime(1000));
    rerender(
      <StatusBanner
        status={{ message: "Working B", done: false }}
        onClear={onClear}
      />,
    );
    act(() => vi.advanceTimersByTime(2000));
    // The first timer was cancelled by re-render; new status is non-done.
    expect(onClear).not.toHaveBeenCalled();
  });

  it("T5.5d — renders nothing when status is null", () => {
    const { container } = render(<StatusBanner status={null} />);
    expect(container.firstChild).toBeNull();
  });
});
