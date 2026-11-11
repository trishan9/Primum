"use client";

import { useEffect, useRef, type ReactNode } from "react";

import { cn } from "./cn";

/**
 * A horizontally scrolling region that says it scrolls: while there is more to the right, its
 * trailing edge fades out (a 24px mask). It is focusable and named, so a keyboard can scroll it.
 * The edge state is written to a data attribute from the scroll listener, never into React state,
 * so scrolling does not re-render the table inside.
 */
export function ScrollArea({
  labelledBy,
  label,
  className,
  children,
}: {
  labelledBy?: string;
  label?: string;
  className?: string;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const update = () => {
      const more = element.scrollLeft + element.clientWidth < element.scrollWidth - 1;
      if (more) element.dataset.more = "";
      else delete element.dataset.more;
    };
    update();
    element.addEventListener("scroll", update, { passive: true });
    const observer = new ResizeObserver(update);
    observer.observe(element);
    if (element.firstElementChild) observer.observe(element.firstElementChild);
    return () => {
      element.removeEventListener("scroll", update);
      observer.disconnect();
    };
  }, []);

  return (
    <div
      ref={ref}
      tabIndex={0}
      role="group"
      aria-labelledby={labelledBy}
      aria-label={labelledBy ? undefined : label}
      className={cn("scroll-fade relative overflow-x-auto", className)}
    >
      {children}
    </div>
  );
}
