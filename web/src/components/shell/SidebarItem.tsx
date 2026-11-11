"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { cn } from "@/components/ui/cn";

import { isActive } from "./nav";

/**
 * One destination: the active one is a neutral pill with the label in ink, never a coloured bar.
 * The glyph arrives rendered (`icon`), so a server parent can pass it across the client boundary.
 */
export function SidebarItem({
  href,
  label,
  icon,
  onNavigate,
}: {
  href: string;
  label: string;
  icon: ReactNode;
  onNavigate?: () => void;
}) {
  const active = isActive(href, usePathname());
  return (
    <li>
      <Link
        href={href}
        onClick={onNavigate}
        aria-current={active ? "page" : undefined}
        data-tip={label}
        data-rail-center=""
        className={cn(
          "flex min-h-11 items-center gap-2.5 rounded-control px-2.5 text-sm transition-colors duration-150 lg:min-h-9",
          active ? "bg-active font-medium text-ink" : "text-muted hover:bg-active/60 hover:text-ink",
        )}
      >
        {icon}
        <span data-rail-label="">{label}</span>
      </Link>
    </li>
  );
}
