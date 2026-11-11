import type { ReactNode } from "react";

/**
 * The frame every view sits in. Slots, not flags:
 *
 *   <AppShell sidebar={<AppSidebar />} topBar={<TopBar />}>{page}</AppShell>
 *
 * Wide screens: a tinted L of sidebar and top bar, and the page on a sheet inset from the
 * window's right and bottom edges. Below 1024px the frame folds away: the top bar carries a menu
 * button and the sheet becomes the screen. The window scrolls, not the sheet, so sticky elements
 * and anchors in the views behave as on any page.
 */
export function AppShell({ sidebar, topBar, children }: { sidebar: ReactNode; topBar: ReactNode; children: ReactNode }) {
  return (
    <div className="flex min-h-dvh">
      {sidebar}
      <div className="flex min-w-0 flex-1 flex-col">
        {topBar}
        <main
          id="main"
          tabIndex={-1}
          className="flex-1 bg-ground outline-none lg:mb-(--sheet-inset) lg:mr-(--sheet-inset) lg:rounded-sheet lg:border lg:border-line"
        >
          <div className="mx-auto w-full max-w-[1320px] px-4 pb-20 pt-6 sm:px-6 lg:px-8 lg:pt-7">{children}</div>
        </main>
      </div>
    </div>
  );
}
