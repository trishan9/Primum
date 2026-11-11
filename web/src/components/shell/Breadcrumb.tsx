"use client";

import { usePathname } from "next/navigation";

import { currentItem } from "./nav";

/** Wide screens: where you are, read from the path. The only part of the top bar that needs it. */
export function Breadcrumb() {
  const here = currentItem(usePathname());
  return (
    <p className="hidden min-w-0 items-center gap-2 text-sm lg:flex">
      <span className="text-muted">Primum</span>
      <span aria-hidden="true" className="text-control">
        /
      </span>
      <span className="truncate font-medium text-ink" aria-current="page">
        {here?.label ?? "Not found"}
      </span>
    </p>
  );
}
