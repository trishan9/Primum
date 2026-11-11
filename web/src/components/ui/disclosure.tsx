import type { ReactNode } from "react";

import { cn } from "./cn";
import { ChevronGlyph } from "./glyphs";

type DisclosureVariant = "plain" | "framed" | "dashed";

const ROOT: Record<DisclosureVariant, string> = {
  plain: "",
  framed: "rounded-card border border-line bg-surface",
  dashed: "rounded-card border border-dashed border-frame",
};

/**
 * Secondary detail kept one click away: provenance, caveats, a withheld draft, the raw file.
 * A native `<details>`, so it works without script and the keyboard gets it for free. Compose it:
 *
 *   <Disclosure variant="framed">
 *     <Disclosure.Summary meta="3 notes">About this run</Disclosure.Summary>
 *     <Disclosure.Body>…</Disclosure.Body>
 *   </Disclosure>
 *
 * `plain` is a quiet text row; `framed` draws a card edge; `dashed` marks something uncertified or empty.
 */
function DisclosureRoot({
  variant = "plain",
  defaultOpen,
  className,
  children,
}: {
  variant?: DisclosureVariant;
  defaultOpen?: boolean;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details
      open={defaultOpen}
      data-variant={variant}
      className={cn("group/disclosure min-w-0", ROOT[variant], className)}
    >
      {children}
    </details>
  );
}

function DisclosureSummary({
  meta,
  size = "sm",
  className,
  children,
}: {
  meta?: ReactNode;
  /** `md` is a section-sized row in ink; `sm` a quiet secondary line. */
  size?: "sm" | "md";
  className?: string;
  children: ReactNode;
}) {
  return (
    <summary
      className={cn(
        "flex cursor-pointer list-none items-center gap-2 rounded-control transition-colors duration-(--dur-fast) [&::-webkit-details-marker]:hidden",
        size === "md"
          ? "min-h-12 text-sm font-semibold text-ink hover:bg-raised/60"
          : "min-h-11 text-meta text-muted hover:text-ink sm:min-h-9",
        "group-data-[variant=framed]/disclosure:px-4 group-data-[variant=dashed]/disclosure:px-4 sm:group-data-[variant=framed]/disclosure:px-5 sm:group-data-[variant=dashed]/disclosure:px-5",
        className,
      )}
    >
      <ChevronGlyph
        size={14}
        className="shrink-0 -rotate-90 text-muted transition-transform duration-(--dur) ease-(--ease-out) group-open/disclosure:rotate-0"
      />
      <span className={cn("min-w-0 flex-1", size === "sm" && "font-medium")}>{children}</span>
      {meta ? <span className="num shrink-0 truncate text-xs font-normal text-muted">{meta}</span> : null}
    </summary>
  );
}

function DisclosureBody({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={cn(
        "text-meta leading-relaxed text-muted",
        "group-data-[variant=plain]/disclosure:pb-3 group-data-[variant=plain]/disclosure:pl-[22px] group-data-[variant=plain]/disclosure:pt-1",
        "group-data-[variant=framed]/disclosure:border-t group-data-[variant=framed]/disclosure:border-rule",
        "group-data-[variant=dashed]/disclosure:border-t group-data-[variant=dashed]/disclosure:border-dashed group-data-[variant=dashed]/disclosure:border-frame",
        "group-data-[variant=dashed]/disclosure:px-4 group-data-[variant=dashed]/disclosure:py-4 group-data-[variant=framed]/disclosure:px-4 group-data-[variant=framed]/disclosure:py-3",
        "sm:group-data-[variant=dashed]/disclosure:px-5 sm:group-data-[variant=framed]/disclosure:px-5",
        className,
      )}
    >
      {children}
    </div>
  );
}

export const Disclosure = Object.assign(DisclosureRoot, { Summary: DisclosureSummary, Body: DisclosureBody });
