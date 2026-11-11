"use client";

import { useId, useRef, useState } from "react";

import { IconButton } from "@/components/ui/button";
import { CrossGlyph, MenuGlyph } from "@/components/ui/glyphs";

import { DISCLAIMER_FULL, NAV } from "./nav";
import { Brand } from "./Brand";
import { SidebarItem } from "./SidebarItem";
import { SystemStatus } from "./StatusCard";

/**
 * Below 1024px: a menu button that opens the navigation as a sheet from the left. It is a native
 * modal <dialog>, so focus stays inside, Esc closes it, and the page behind is inert.
 */
export function MobileNav() {
  const dialog = useRef<HTMLDialogElement>(null);
  const id = useId();
  const [open, setOpen] = useState(false);
  const close = () => dialog.current?.close();

  return (
    <>
      <IconButton
        label="Open navigation"
        onClick={() => {
          dialog.current?.showModal();
          setOpen(true);
        }}
        className="lg:hidden"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls={id}
      >
        <MenuGlyph size={18} />
      </IconButton>
      <dialog
        ref={dialog}
        id={id}
        aria-label="Navigation"
        onClose={() => setOpen(false)}
        className="nav-sheet"
        onClick={(event) => {
          // A click on the backdrop lands on the dialog element itself.
          if (event.target === event.currentTarget) close();
        }}
      >
        <div className="flex h-full flex-col overflow-y-auto">
          <div className="flex h-14 shrink-0 items-center justify-between gap-2 px-3">
            <Brand onNavigate={close} />
            <IconButton label="Close navigation" onClick={close}>
              <CrossGlyph size={18} />
            </IconButton>
          </div>
          <nav aria-label="Views" className="flex flex-col gap-5 px-3 pt-2">
            {NAV.map((group) => (
              <div key={group.label}>
                <p className="mb-1 px-2.5 text-xs font-medium text-muted">{group.label}</p>
                <ul className="flex flex-col gap-0.5">
                  {group.items.map((item) => (
                    <SidebarItem
                      key={item.href}
                      href={item.href}
                      label={item.label}
                      icon={<item.Glyph size={16} />}
                      onNavigate={close}
                    />
                  ))}
                </ul>
              </div>
            ))}
          </nav>
          <div className="mt-auto flex flex-col gap-3 p-3">
            <SystemStatus />
            <p className="px-1 text-xs leading-relaxed text-muted">{DISCLAIMER_FULL}</p>
          </div>
        </div>
      </dialog>
    </>
  );
}
