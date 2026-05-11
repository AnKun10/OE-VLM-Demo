import type { ComponentProps } from "react";

/**
 * <a> wrapper used inside react-markdown. External links open in a new
 * tab with rel=noopener,noreferrer. Same-origin / hash links keep
 * default behavior.
 */
export function SafeLink(props: ComponentProps<"a">) {
  const href = props.href ?? "";
  const isExternal = /^https?:\/\//.test(href);
  if (!isExternal) return <a {...props} />;
  return <a {...props} target="_blank" rel="noopener noreferrer" />;
}
