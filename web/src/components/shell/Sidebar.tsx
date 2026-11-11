import type { ReactNode } from "react";

import { Brand } from "./Brand";
import { NAV, SIDEBAR_ID } from "./nav";
import { SidebarItem } from "./SidebarItem";
import { RailStatus, SystemStatus } from "./StatusCard";

/* ---------- Primitives ----------
   A server component: only the active-state item and the status widgets are client leaves. */

/** The sidebar column on wide screens: sticky, full height, on the chrome tint. Hidden below 1024px. */
function SidebarRoot({ label = "Primary", children }: { label?: string; children: ReactNode }) {
  return (
    <aside
      id={SIDEBAR_ID}
      aria-label={label}
      className="app-sidebar sticky top-0 z-40 hidden h-dvh shrink-0 flex-col overflow-y-auto bg-chrome lg:flex [scrollbar-width:thin] [:root[data-sidebar=collapsed]_&]:overflow-visible"
    >
      {children}
    </aside>
  );
}

function SidebarHeader({ children }: { children: ReactNode }) {
  return <div className="flex h-(--topbar-h) shrink-0 items-center px-3">{children}</div>;
}

function SidebarNav({ label = "Views", children }: { label?: string; children: ReactNode }) {
  return (
    <nav aria-label={label} className="flex flex-col gap-5 px-3 pt-3">
      {children}
    </nav>
  );
}

function SidebarGroup({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <p data-rail-hide="" className="mb-1 px-2.5 text-xs font-medium text-muted">
        {label}
      </p>
      <ul className="flex flex-col gap-0.5">{children}</ul>
    </div>
  );
}

function SidebarFooter({ children }: { children: ReactNode }) {
  return <div className="mt-auto flex flex-col gap-2 p-3">{children}</div>;
}

export const Sidebar = Object.assign(SidebarRoot, {
  Header: SidebarHeader,
  Nav: SidebarNav,
  Group: SidebarGroup,
  Item: SidebarItem,
  Footer: SidebarFooter,
});

/* ---------- The app's sidebar ---------- */

export function AppSidebar() {
  return (
    <Sidebar>
      <Sidebar.Header>
        <Brand />
      </Sidebar.Header>
      <Sidebar.Nav>
        {NAV.map((group) => (
          <Sidebar.Group key={group.label} label={group.label}>
            {group.items.map((item) => (
              <Sidebar.Item key={item.href} href={item.href} label={item.label} icon={<item.Glyph size={16} />} />
            ))}
          </Sidebar.Group>
        ))}
      </Sidebar.Nav>
      <Sidebar.Footer>
        <div data-rail-hide="">
          <SystemStatus />
        </div>
        <div data-rail-only="" className="justify-center">
          <RailStatus />
        </div>
      </Sidebar.Footer>
    </Sidebar>
  );
}
