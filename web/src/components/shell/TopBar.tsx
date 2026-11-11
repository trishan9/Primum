import Link from "next/link";

import { InfoGlyph, PrimumMark } from "@/components/ui/glyphs";

import { Breadcrumb } from "./Breadcrumb";
import { MobileNav } from "./MobileNav";
import { DISCLAIMER_FULL, DISCLAIMER_SHORT } from "./nav";
import { SidebarToggle } from "./SidebarToggle";
import { ThemeToggle } from "./ThemeToggle";

/**
 * The slim bar over the sheet, on the chrome tint with no rule under it. It names where you are,
 * keeps the research-prototype notice in view on every page, and holds the theme switch.
 * Below 1024px it is the whole navigation: menu button, product name, theme. A server component:
 * the breadcrumb, the toggles and the menu are client leaves.
 */
export function TopBar() {
  return (
    <header className="sticky top-0 z-30 flex h-(--topbar-h) shrink-0 items-center gap-2 bg-chrome px-2 max-lg:h-auto max-lg:min-h-14 max-lg:border-b max-lg:border-line lg:gap-3 lg:pr-3">
      <MobileNav />
      <SidebarToggle />

      {/* Wide: where you are. Narrow: the product, with the notice as its subline. */}
      <div className="flex min-w-0 flex-1 items-center gap-3">
        <Breadcrumb />
        <Link href="/" className="flex min-h-11 min-w-0 items-center gap-2.5 rounded-control py-1.5 lg:hidden">
          <PrimumMark size={20} />
          <span className="min-w-0 leading-tight">
            <span className="block text-[0.9375rem] font-semibold tracking-[-0.01em] text-ink">Primum</span>
            <span className="block text-xs text-muted">{DISCLAIMER_SHORT}</span>
          </span>
        </Link>
      </div>

      <p className="hidden min-w-0 items-center gap-1.5 text-xs text-muted lg:flex" title={DISCLAIMER_FULL}>
        <InfoGlyph size={14} />
        <span className="truncate xl:hidden">{DISCLAIMER_SHORT}</span>
        <span className="hidden truncate xl:inline">{DISCLAIMER_FULL}</span>
      </p>
      <ThemeToggle />
    </header>
  );
}
