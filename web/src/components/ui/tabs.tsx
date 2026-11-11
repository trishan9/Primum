"use client";

import { createContext, useCallback, useContext, useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

import { cn } from "./cn";

interface TabsContextValue {
  value: string;
  select: (value: string) => void;
  baseId: string;
}

const TabsContext = createContext<TabsContextValue | null>(null);

function useTabs(part: string): TabsContextValue {
  const context = useContext(TabsContext);
  if (!context) throw new Error(`<${part}> must be inside <Tabs>`);
  return context;
}

const slug = (value: string) => value.replace(/[^a-zA-Z0-9_-]/g, "_");

/**
 * Underline tabs on a full-width rule, following the WAI-ARIA tabs pattern (arrow keys, Home,
 * End; only the selected tab is in the tab order). Controlled with `value` + `onValueChange`, or
 * uncontrolled with `defaultValue`.
 *
 *   <Tabs defaultValue="rq1">
 *     <TabsList label="Research question">
 *       <Tab value="rq1">Retrieval</Tab>
 *       <Tab value="rq2">Calibration</Tab>
 *     </TabsList>
 *     <TabPanel value="rq1">…</TabPanel>
 *     <TabPanel value="rq2">…</TabPanel>
 *   </Tabs>
 *
 * Import the named parts (TabsList, Tab, TabPanel); `Tabs.List` etc. also work inside client files.
 */
function TabsRoot({
  value: controlled,
  defaultValue,
  onValueChange,
  className,
  children,
}: {
  value?: string;
  defaultValue?: string;
  onValueChange?: (value: string) => void;
  className?: string;
  children: ReactNode;
}) {
  const [uncontrolled, setUncontrolled] = useState(defaultValue ?? "");
  const value = controlled ?? uncontrolled;
  const baseId = useId();
  const select = useCallback(
    (next: string) => {
      if (controlled === undefined) setUncontrolled(next);
      onValueChange?.(next);
    },
    [controlled, onValueChange],
  );
  return (
    <TabsContext.Provider value={{ value, select, baseId }}>
      <div className={className}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ label, className, children }: { label: string; className?: string; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const tabs = Array.from(ref.current?.querySelectorAll<HTMLButtonElement>('[role="tab"]:not(:disabled)') ?? []);
    const index = tabs.indexOf(document.activeElement as HTMLButtonElement);
    if (index < 0) return;
    const target =
      event.key === "ArrowRight"
        ? tabs[(index + 1) % tabs.length]
        : event.key === "ArrowLeft"
          ? tabs[(index - 1 + tabs.length) % tabs.length]
          : event.key === "Home"
            ? tabs[0]
            : event.key === "End"
              ? tabs[tabs.length - 1]
              : null;
    if (!target) return;
    event.preventDefault();
    target.focus();
    target.click();
  };

  return (
    // The rule is the row's own border; the list scrolls sideways inside it on a phone, never the page.
    <div className="overflow-x-auto border-b border-line [scrollbar-width:none]">
      <div ref={ref} role="tablist" aria-label={label} onKeyDown={onKeyDown} className={cn("flex min-w-max gap-1", className)}>
        {children}
      </div>
    </div>
  );
}

export function Tab({ value, disabled, className, children }: { value: string; disabled?: boolean; className?: string; children: ReactNode }) {
  const { value: selected, select, baseId } = useTabs("Tab");
  const active = selected === value;
  return (
    <button
      type="button"
      role="tab"
      id={`${baseId}-tab-${slug(value)}`}
      aria-selected={active}
      aria-controls={`${baseId}-panel-${slug(value)}`}
      tabIndex={active ? 0 : -1}
      disabled={disabled}
      onClick={() => select(value)}
      className={cn(
        "relative -mb-px flex min-h-11 items-center gap-2 whitespace-nowrap border-b-2 px-3 text-sm transition-colors duration-150 sm:min-h-10",
        active ? "border-accent font-medium text-ink" : "border-transparent text-muted hover:border-control hover:text-ink",
        "disabled:cursor-not-allowed disabled:text-faint",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function TabPanel({ value, className, children }: { value: string; className?: string; children: ReactNode }) {
  const { value: selected, baseId } = useTabs("TabPanel");
  if (selected !== value) return null;
  return (
    <div
      role="tabpanel"
      id={`${baseId}-panel-${slug(value)}`}
      aria-labelledby={`${baseId}-tab-${slug(value)}`}
      tabIndex={0}
      className={cn("pt-5 focus-visible:outline-offset-4", className)}
    >
      {children}
    </div>
  );
}

export const Tabs = Object.assign(TabsRoot, { List: TabsList, Tab, Panel: TabPanel });
