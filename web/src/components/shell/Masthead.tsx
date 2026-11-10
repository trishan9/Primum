"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { PrimumMark } from "@/components/ui/glyphs";

import { ApiStatus } from "./ApiStatus";
import { ThemeToggle } from "./ThemeToggle";

const VIEWS = [
  { href: "/", label: "Ask" },
  { href: "/retrieval", label: "Retrieval lab", short: "Retrieval" },
  { href: "/experiments", label: "Experiments" },
  { href: "/method", label: "Method" },
] as const;

export function Masthead() {
  const pathname = usePathname();

  return (
    <>
    <header className="z-30 border-b border-line bg-ground sm:sticky sm:top-0">
      <div className="mx-auto flex w-full max-w-[1320px] flex-wrap items-center gap-x-6 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex min-h-12 items-center gap-2 rounded-control" aria-label="Primum, home">
          <PrimumMark />
          <span className="font-serif text-[1.3125rem] font-semibold leading-none tracking-[-0.01em]">Primum</span>
        </Link>

        <nav aria-label="Views" className="order-3 -mx-1 w-full sm:order-2 sm:mx-0 sm:w-auto">
          <ul className="flex items-stretch sm:gap-1">
            {VIEWS.map((view) => {
              const active = view.href === "/" ? pathname === "/" : pathname.startsWith(view.href);
              return (
                <li key={view.href} className="flex-1 sm:flex-none">
                  <Link
                    href={view.href}
                    aria-current={active ? "page" : undefined}
                    className={`relative flex min-h-11 items-center justify-center whitespace-nowrap px-2 text-sm sm:min-h-12 sm:px-3 ${
                      active ? "font-medium text-ink" : "text-muted hover:text-ink"
                    }`}
                  >
                    {"short" in view ? (
                      <>
                        <span className="sm:hidden">{view.short}</span>
                        <span className="hidden sm:inline">{view.label}</span>
                      </>
                    ) : (
                      view.label
                    )}
                    {active ? <span aria-hidden="true" className="absolute inset-x-2 bottom-0 h-0.5 bg-ink sm:inset-x-3" /> : null}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="order-2 ml-auto flex items-center gap-1 sm:order-3 sm:gap-3">
          <ApiStatus />
          <ThemeToggle />
        </div>
      </div>
    </header>
    {/* Stays in view while the page scrolls, on every view. */}
    <p className="sticky top-0 z-20 border-b border-rule bg-ground sm:top-12">
      <span className="mx-auto block w-full max-w-[1320px] px-4 py-1.5 text-xs leading-snug text-muted sm:px-6 lg:px-8">
        Research prototype, not medical advice. Answers are generated from retrieved passages and can be wrong.
      </span>
    </p>
    </>
  );
}
