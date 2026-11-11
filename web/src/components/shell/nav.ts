import type { ComponentType } from "react";

import { BookGlyph, ChatGlyph, ColumnsGlyph, IntervalGlyph } from "@/components/ui/glyphs";

export interface NavItem {
  href: string;
  label: string;
  /** What the destination is for, in one line: the sidebar's tooltip and the mobile sheet's subline. */
  hint: string;
  Glyph: ComponentType<{ size?: number; className?: string }>;
}

export interface NavGroup {
  label: string;
  items: readonly NavItem[];
}

/** The four destinations, in two groups: working with the system, and reading how it was tested. */
export const NAV: readonly NavGroup[] = [
  {
    label: "Workspace",
    items: [
      { href: "/", label: "Ask", hint: "Ask a clinical question", Glyph: ChatGlyph },
      { href: "/retrieval", label: "Retrieval lab", hint: "Compare retrievers on one query", Glyph: ColumnsGlyph },
    ],
  },
  {
    label: "Research",
    items: [
      { href: "/experiments", label: "Experiments", hint: "RQ1 to RQ3 from the results files", Glyph: IntervalGlyph },
      { href: "/method", label: "Method", hint: "How the threshold is certified", Glyph: BookGlyph },
    ],
  },
];

export function isActive(href: string, pathname: string): boolean {
  return href === "/" ? pathname === "/" : pathname === href || pathname.startsWith(`${href}/`);
}

export function currentItem(pathname: string): NavItem | undefined {
  return NAV.flatMap((group) => group.items).find((item) => isActive(item.href, pathname));
}

export const DISCLAIMER_SHORT = "Research prototype, not medical advice.";
export const DISCLAIMER_FULL =
  "Research prototype, not medical advice. Answers are generated from retrieved passages and can be wrong.";

/** The wide-screen sidebar's id, for the collapse toggle's `aria-controls`. */
export const SIDEBAR_ID = "app-sidebar";
