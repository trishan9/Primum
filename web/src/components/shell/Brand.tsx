import Link from "next/link";

import { PrimumMark } from "@/components/ui/glyphs";

/** The product's mark and name. In the rail, the mark alone. */
export function Brand({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <Link
      href="/"
      onClick={onNavigate}
      className="flex min-h-11 items-center gap-2.5 rounded-control px-1.5"
      aria-label="Primum, Ask"
      data-rail-center=""
    >
      <span className="flex size-7 items-center justify-center rounded-[7px] bg-surface ring-1 ring-line">
        <PrimumMark size={18} />
      </span>
      <span data-rail-hide="" className="min-w-0 leading-tight">
        <span className="block text-[0.9375rem] font-semibold tracking-[-0.01em] text-ink">Primum</span>
        <span className="block text-xs text-muted">Clinical QA research</span>
      </span>
    </Link>
  );
}
