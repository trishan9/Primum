"use client";

import { useEffect, useState } from "react";

import { cn } from "@/components/ui/cn";
import { ChevronGlyph } from "@/components/ui/glyphs";

interface Section {
  id: string;
  label: string;
}

/** The section whose heading has most recently passed under the top bar. */
function useCurrentSection(key: string): string {
  const [current, setCurrent] = useState(key.split(" ")[0] ?? "");

  useEffect(() => {
    const ids = key.split(" ");
    let frame = 0;
    const measure = () => {
      frame = 0;
      // A little below the sticky top bar, where the reader's eye lands after a jump.
      const line = 120;
      let found = ids[0] ?? "";
      for (const id of ids) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= line) found = id;
      }
      // At the very end of the page the last short sections never reach the line.
      if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 2) {
        found = ids[ids.length - 1] ?? found;
      }
      setCurrent(found);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, [key]);

  return current;
}

/**
 * The page's table of contents. On wide screens a sticky rail that marks the section being read;
 * below 1024px a collapsible "On this page" above the text.
 */
export function OnThisPage({ sections }: { sections: readonly Section[] }) {
  // A string key, so the scroll subscription is made once rather than on every render.
  const current = useCurrentSection(sections.map((s) => s.id).join(" "));

  return (
    <>
      <nav aria-label="On this page" className="sticky top-[calc(var(--topbar-h)+16px)] hidden lg:block">
        <div>
          <p className="px-3 text-meta font-semibold text-ink">On this page</p>
          <ol className="relative mt-2 border-l border-line">
            {sections.map((section) => {
              const active = section.id === current;
              return (
                <li key={section.id}>
                  <a
                    href={`#${section.id}`}
                    aria-current={active ? "location" : undefined}
                    className={cn(
                      "relative -ml-px block border-l-2 py-1.5 pr-2 pl-3 text-sm transition-colors duration-(--dur-fast)",
                      active
                        ? "border-accent font-medium text-ink"
                        : "border-transparent text-muted hover:border-control hover:text-ink",
                    )}
                  >
                    {section.label}
                  </a>
                </li>
              );
            })}
          </ol>
        </div>
      </nav>

      <details className="group rounded-card border border-line bg-surface lg:hidden">
        <summary className="flex min-h-11 cursor-pointer list-none items-center justify-between gap-3 px-4 text-sm font-medium text-ink [&::-webkit-details-marker]:hidden">
          On this page
          <ChevronGlyph className="text-muted transition-transform duration-(--dur) group-open:rotate-180" />
        </summary>
        <ol className="grid grid-cols-2 gap-x-2 border-t border-rule px-2 py-2">
          {sections.map((section) => (
            <li key={section.id}>
              <a
                href={`#${section.id}`}
                className="flex min-h-11 items-center rounded-control px-2 text-sm text-muted hover:bg-raised hover:text-ink"
              >
                {section.label}
              </a>
            </li>
          ))}
        </ol>
      </details>
    </>
  );
}
