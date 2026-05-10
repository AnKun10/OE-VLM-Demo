import type { ModelInfo } from "../types";

export function ModelDropdown({
  models,
  value,
  onChange,
}: {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
}) {
  if (models.length === 0) return null;
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="text-xs rounded-lg px-2 py-1.5 outline-none cursor-pointer transition-colors"
      style={{
        color: "#6b7280",
        background: "transparent",
        border: "1px solid #e5e7eb",
        maxWidth: 200,
      }}
      aria-label="Chọn mô hình"
    >
      {models.map((m) => (
        <option key={m.id} value={m.id}>
          {m.capabilities.vision ? "👁 " : ""}
          {m.name}
        </option>
      ))}
    </select>
  );
}
