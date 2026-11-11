/**
 * The interface's own small glyph set: one 16px grid, one 1.5px stroke, round joins.
 * Each glyph is decorative next to a text label, so all are hidden from assistive tech.
 */

interface GlyphProps {
  className?: string;
  size?: number;
}

function Svg({ className, size = 16, children }: GlyphProps & { children: React.ReactNode }) {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={`shrink-0 ${className ?? ""}`}
    >
      {children}
    </svg>
  );
}

export function CheckGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M3 8.5 6.5 12 13 4.5" />
    </Svg>
  );
}

export function CrossGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M4 4l8 8M12 4l-8 8" />
    </Svg>
  );
}

/** An open ring: looked for, nothing found. */
export function RingGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="4.75" />
    </Svg>
  );
}

export function ExternalGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M6.5 3.5h-3v9h9v-3M9 3.5h3.5V7M12.5 3.5 7.5 8.5" />
    </Svg>
  );
}

export function ChevronGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M4 6l4 4 4-4" />
    </Svg>
  );
}

export function SunGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="2.75" />
      <path d="M8 1.5v1.5M8 13v1.5M1.5 8H3M13 8h1.5M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M3.4 12.6l1.1-1.1M11.5 4.5l1.1-1.1" />
    </Svg>
  );
}

export function MoonGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M13 9.5A5.5 5.5 0 0 1 6.5 3 5.5 5.5 0 1 0 13 9.5Z" />
    </Svg>
  );
}

export function StopGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <rect x="4.5" y="4.5" width="7" height="7" rx="1" />
    </Svg>
  );
}

export function WarningGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M8 2.5 14 13H2L8 2.5ZM8 6.5v3M8 11.25v.25" />
    </Svg>
  );
}

/** The product mark: a pointer resting on a ruled scale, to the right of a gate. */
export function PrimumMark({ className, size = 22 }: GlyphProps) {
  return (
    <svg aria-hidden="true" focusable="false" width={size} height={size} viewBox="0 0 22 22" className={`shrink-0 ${className ?? ""}`}>
      <path d="M2 15.5h18" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      <path d="M4 15.5v-3M7.5 15.5v-2M14.5 15.5v-2M18 15.5v-3" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      <path d="M11 18.5V8" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" />
      <path d="M12.9 3.5h6.2L16 9.2Z" fill="var(--accent)" />
    </svg>
  );
}

/* ---------- Navigation and shell ---------- */

/** Ask: a speech balloon. */
export function ChatGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M2.75 4.25a1.5 1.5 0 0 1 1.5-1.5h7.5a1.5 1.5 0 0 1 1.5 1.5v5a1.5 1.5 0 0 1-1.5 1.5H7l-3 2.5v-2.5h.25a1.5 1.5 0 0 1-1.5-1.5Z" />
    </Svg>
  );
}

/** Retrieval lab: ranked lists side by side. */
export function ColumnsGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <rect x="2.5" y="2.75" width="4.5" height="10.5" rx="1" />
      <rect x="9" y="2.75" width="4.5" height="10.5" rx="1" />
      <path d="M4 5.5h1.5M4 8h1.5M10.5 5.5H12M10.5 8H12" />
    </Svg>
  );
}

/** Experiments: an estimate with its interval. */
export function IntervalGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M2.5 13.5h11M3.5 4.5h6M5 9h7.5" />
      <circle cx="6.5" cy="4.5" r="1.25" fill="currentColor" />
      <circle cx="9" cy="9" r="1.25" fill="currentColor" />
    </Svg>
  );
}

/** Method: an open book. */
export function BookGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M8 4.25C6.75 3.25 4.75 3 2.5 3.25v9c2.25-.25 4.25 0 5.5 1M8 4.25c1.25-1 3.25-1.25 5.5-1v9c-2.25-.25-4.25 0-5.5 1M8 4.25v9" />
    </Svg>
  );
}

export function MenuGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M2.5 4.5h11M2.5 8h11M2.5 11.5h11" />
    </Svg>
  );
}

/** Collapse or expand the sidebar: a panel with its left column. */
export function SidebarGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <rect x="2.25" y="2.75" width="11.5" height="10.5" rx="1.75" />
      <path d="M6.25 2.75v10.5" />
    </Svg>
  );
}

export function RefreshGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M13 8a5 5 0 1 1-1.46-3.54M13 2.75v2.5h-2.5" />
    </Svg>
  );
}

export function InfoGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="5.75" />
      <path d="M8 7.25V11M8 5v.25" />
    </Svg>
  );
}

/* ---------- Content ---------- */

export function SearchGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="7" cy="7" r="4.25" />
      <path d="m10.25 10.25 3.25 3.25" />
    </Svg>
  );
}

/** Send: an arrow pointing up, as in a composer. */
export function ArrowUpGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M8 13V3M3.75 7.25 8 3l4.25 4.25" />
    </Svg>
  );
}

export function ArrowRightGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M3 8h10M8.75 3.75 13 8l-4.25 4.25" />
    </Svg>
  );
}

/** A stage's elapsed time. */
export function ClockGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="8" cy="8" r="5.75" />
      <path d="M8 4.75V8l2.25 1.5" />
    </Svg>
  );
}

/** A source passage. */
export function DocumentGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M4 2.5h5.25L12 5.25v8.25H4ZM9 2.5v3h3M6 8.25h4M6 10.75h4" />
    </Svg>
  );
}

/** Verification: a shield with a tick. */
export function ShieldGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M8 2.25 3 4v4c0 3 2.25 5 5 5.75 2.75-.75 5-2.75 5-5.75V4Z" />
      <path d="m5.75 8 1.6 1.6L10.5 6.5" />
    </Svg>
  );
}

/** Confidence: a gauge with its needle. */
export function GaugeGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M2.75 11.5a5.25 5.25 0 1 1 10.5 0" />
      <path d="M8 11.5 10.5 7" />
    </Svg>
  );
}

/** Concepts: linked nodes. */
export function LinkNodesGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <circle cx="4" cy="4.5" r="1.75" />
      <circle cx="12" cy="4.5" r="1.75" />
      <circle cx="8" cy="12" r="1.75" />
      <path d="M5.75 4.5h4.5M5 6l2.1 4.4M11 6l-2.1 4.4" />
    </Svg>
  );
}

/** Generation: a pen nib on a line. */
export function PenGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M10.5 2.75 13.25 5.5 6 12.75H3.25V10Z" />
      <path d="M9 4.25 11.75 7" />
    </Svg>
  );
}

export function CopyGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <rect x="5.25" y="5.25" width="8" height="8" rx="1.5" />
      <path d="M10.75 5.25V4.25a1.5 1.5 0 0 0-1.5-1.5h-5a1.5 1.5 0 0 0-1.5 1.5v5a1.5 1.5 0 0 0 1.5 1.5h1" />
    </Svg>
  );
}

/** Settings: two sliders on their tracks. */
export function SlidersGlyph(props: GlyphProps) {
  return (
    <Svg {...props}>
      <path d="M2.75 5h10.5M2.75 11h10.5" />
      <circle cx="5.75" cy="5" r="1.75" fill="var(--surface)" />
      <circle cx="10.25" cy="11" r="1.75" fill="var(--surface)" />
    </Svg>
  );
}
