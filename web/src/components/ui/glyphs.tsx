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
