"use client";

import { useEffect, useState } from "react";

import { CheckGlyph, CopyGlyph } from "@/components/ui/glyphs";

/** Copies text to the clipboard and says so for a moment. */
export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const timer = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(timer);
  }, [copied]);

  return (
    <button
      type="button"
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(
          () => setCopied(true),
          () => setCopied(false),
        );
      }}
      className="inline-flex min-h-11 items-center gap-1.5 rounded-control px-2 text-[0.8125rem] text-muted transition-colors duration-(--dur-fast) hover:bg-raised hover:text-ink sm:min-h-8"
    >
      {copied ? <CheckGlyph size={14} /> : <CopyGlyph size={14} />}
      <span aria-live="polite">{copied ? "Copied" : label}</span>
    </button>
  );
}
