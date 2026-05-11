import { ImagePlus } from "lucide-react";

/**
 * Full-screen translucent overlay shown while user drags files over
 * the page. Pointer events pass through except for the drop target
 * itself (which is the page wrapper that mounts this overlay).
 */
export function DropOverlay({ visible }: { visible: boolean }) {
  if (!visible) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none"
      style={{ background: "rgba(1,94,159,0.18)", backdropFilter: "blur(2px)" }}
      role="presentation"
    >
      <div
        className="flex flex-col items-center gap-3 px-8 py-6 rounded-2xl"
        style={{ background: "white", border: "2px dashed #015e9f" }}
      >
        <ImagePlus size={32} color="#015e9f" />
        <span className="text-sm font-medium" style={{ color: "#015e9f" }}>
          Thả ảnh để tải lên
        </span>
      </div>
    </div>
  );
}
