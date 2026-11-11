"use client";

import { useEffect } from "react";

import { IconButton } from "@/components/ui/button";
import { SidebarGlyph } from "@/components/ui/glyphs";

import { SIDEBAR_ID } from "./nav";
import { useSidebarCollapsed } from "./sidebar-state";

/**
 * Collapses the wide-screen sidebar to a rail. It also lets Esc dismiss the rail's hover and
 * focus tooltips until the pointer or focus moves again (WCAG 1.4.13).
 */
export function SidebarToggle() {
  const [collapsed, toggle] = useSidebarCollapsed();

  useEffect(() => {
    const root = document.documentElement;
    const hide = (event: KeyboardEvent) => {
      if (event.key === "Escape") root.dataset.tips = "off";
    };
    const show = () => {
      if (root.dataset.tips) delete root.dataset.tips;
    };
    document.addEventListener("keydown", hide);
    document.addEventListener("pointermove", show, { passive: true });
    document.addEventListener("focusin", show);
    return () => {
      document.removeEventListener("keydown", hide);
      document.removeEventListener("pointermove", show);
      document.removeEventListener("focusin", show);
    };
  }, []);

  return (
    <IconButton
      label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      aria-expanded={collapsed === null ? undefined : !collapsed}
      aria-controls={SIDEBAR_ID}
      onClick={toggle}
      className="max-lg:hidden"
    >
      <SidebarGlyph size={16} />
    </IconButton>
  );
}
