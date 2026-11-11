# Primum web: design direction

This file is the contract for every view. It says what the interface is, the tokens it is built
from, the shared components and their APIs, and how each view is composed. Judge a change against
it; change it when a decision changes, with the reason.

## Design read

Primum is an **agent workspace for clinical question answering that knows when not to answer**.
People use it in working sessions, not glances: a researcher or clinically literate reviewer asks,
watches the pipeline run step by step, reads the evidence, and checks the experiments that justify
the threshold. So it looks like the AI tools they already work in (a sidebar of destinations, a
slim top bar, a chat thread with a visible agent trace, cards on a calm canvas) and keeps the
precision of a research instrument: every figure tabular and sourced, colour only where it means
something, nothing decorative.

- Mode: **Operate** (Ask, Retrieval lab, Experiments) and **Read** (Method).
- Dials: energy 2 (calm), density medium, motion 1 (feedback and state changes only).
- Style: *framed sheet*. One tinted L of sidebar and top bar; the page sits on a sheet inset 8px
  from the window's right and bottom edges, 16px corners, one hairline, no shadow.
- The face: the frame is tinted a cool **green-grey from the clinic** (scrubs, examination-room
  walls), not the grey of a SaaS kit; the one accent is **petrol**, a deep blue-teal that reads
  as medical without being the green of "supported".
- The element only this product has: **the gate**. A confidence pointer on a scale against a
  certified threshold; left of the gate is hatched and withheld. The product mark is that pointer.

## Integrity rules (non-negotiable)

- No result number is typed into the code. Every figure comes from the API or a results file at
  runtime; where one is missing, show a grey dash (`–`) and say which command produces it.
- Keep the honesty copy wherever it applies: the guarantee holds only for PubMedQA held-out
  questions answered by the same generator; a chat answer gets an *operating point*, not a
  guarantee; abstention is not an error; "Research prototype, not medical advice." (The shell
  shows the last on every page; views keep the rest.)
- The Ask view's fixture replay is always bannered on screen and named in the sidebar status.

## Tokens

All tokens are CSS variables in `src/app/globals.css`, defined for light, for dark under
`prefers-color-scheme` (unless `data-theme="light"`), and for `data-theme="dark"`. Tailwind names
in brackets. Never write a hex value in a component.

### Colour

| Role | Light | Dark | Tailwind | Use |
| --- | --- | --- | --- | --- |
| `--chrome` | #eef1ef | #0b0c0c | `bg-chrome` | Sidebar, top bar, mobile sheet |
| `--ground` | #f9fbfa | #111312 | `bg-ground` | The sheet the views draw on; also a hover tint inside cards |
| `--surface` | #ffffff | #161918 | `bg-surface` | Cards, inputs, popovers |
| `--raised` | #f1f4f2 | #1e2120 | `bg-raised` | Wells, segmented tracks, code chips, row hover |
| `--active` | #e2e7e4 | #232726 | `bg-active` | Selected nav item, pressed state, skeleton bars |
| `--border` | #e1e6e3 | #272b2a | `border-line` | Card and sheet edges |
| `--rule` | #ecefed | #1f2221 | `border-rule` | Row rules inside a card |
| `--control` | #868d89 | #6a716e | `border-control` | Input and control edges (3:1 or more, WCAG 1.4.11) |
| `--frame` | #c3c9c5 | #3a3f3d | `border-frame` | Dashed frames of empty states; decorative only, never a field edge |
| `--text` | #111614 | #eef1f0 | `text-ink` | Primary text, figures |
| `--muted` | #545b57 | #b0b6b3 | `text-muted` | Secondary text, labels (6:1 or more) |
| `--faint` | #5f6662 | #8d9491 | `text-faint` | Placeholders, disabled (4.5:1 on chrome, ground, surface and `--active`) |
| `--primary` / `--on-primary` | #121715 / #fff | #eef1f0 / #0b0c0c | `bg-primary text-on-primary` | The one filled action per screen |
| `--accent` | #1d6a87 | #4fa6cc | `text-accent` etc. | Confidence pointer, lead chart series, selected tab, focus ring, progress |
| `--accent-soft` / `--accent-ink` | #e4f0f4 / #11475b | #132a33 / #a9d6ea | | Accent badge, highlighted source |
| `--link` | #185a74 | #8ec8e2 | `text-link` | Inline links |
| verdicts | `--supported*`, `--unsupported*`, `--contradicted*` | | `text-supported-ink` … | Claim verdicts only, always with a glyph and a word |
| `--emergency*` | = contradicted | | `bg-emergency-bg` … | The emergency triage state only |
| `--hatch` | #a3a9a6 | #5a5f5d | `.hatch` | Everything withheld. Abstention is neutral and hatched, never red |
| data | `--grid`, `--axis`, `--series-1..4`, `--series-rest` | | | Charts; `--series-1` is the accent |

Colour budget: under about 3% of a screen's pixels. The accent goes on data and on at most one
selected control; status colours on status; everything else is ink and greys. No gradients, glass,
glows, violet or indigo. Amber (`unsupported`) doubles as the warning tone (over budget, fixture
replay), always in words.

### Type

| Face | Variable | Role |
| --- | --- | --- |
| IBM Plex Sans 400/500/600 | `font-sans` | The interface, labels and **every figure** (true tabular figures, Greek for α, δ, λ̂) |
| STIX Two Text | `font-serif` | Prose the system *says*: the answer, claims, passage text, the Method page. Nothing else |
| IBM Plex Mono | `font-mono` | Commands, identifiers, model names, PMIDs. Nothing else |

Scale (px): 12 `text-xs` (meta, axis) · 13 `text-meta` (secondary lines) · 14 `text-sm` (body,
controls, cells) · 15 card titles `text-[0.9375rem]` · 16 `text-base` · 19 `text-reading`
(serif prose, `.reading`) · 22 `text-title` (page h1, 600, tracking -0.012em) · 28 `text-kpi`
(figures, 400). Nothing above 32px; no figure at 48px. Sentence case everywhere; no tracked capitals,
no eyebrow labels above headings. Add `.num` (tabular, lining) to every number in a column, a
reading, or anything that updates.

### Space, radius, elevation, motion

- Spacing on a 4px base. Canvas padding 16 (phone) / 24 (tablet) / 32 (desktop). One gap per
  grid: 16px between cards in dense grids, 24px between major blocks. Card padding 16 / 20.
- Radius steps down as things nest: sheet 16 (`rounded-sheet`), card 12 (`rounded-card`),
  control 8 (`rounded-control`), chip 6 (`rounded-chip`).
- Elevation: none at rest. A border *or* a tone step, never both plus a shadow. `shadow-float`
  only for what floats (tooltip, popover, dialog); `shadow-dock` (one soft 16px layer) on the
  docked composer only; `shadow-control` (1px) on raised buttons and the selected segment.
- Motion: `--dur-fast` 120ms (hover), `--dur` 180ms (state), `--dur-slow` 280ms (sheets),
  `--ease-out`. Motion answers an action or shows a state change (the pointer settling, a caret
  while text streams, the running step's pulse, the mobile sheet sliding in). Nothing performs an
  entrance. All of it stops under `prefers-reduced-motion` (global rule in globals.css).

## Layout system (the shell)

`src/app/layout.tsx` composes the shell with slots:

```tsx
<ApiProvider>
  <AppShell sidebar={<AppSidebar />} topBar={<TopBar />}>{children}</AppShell>
</ApiProvider>
```

- **≥1024px**: sidebar 240px (`--sidebar-w`), sticky, on `--chrome`: brand, two nav groups
  (Workspace: Ask, Retrieval lab; Research: Experiments, Method), and the **System** status card
  (API reachability with refresh, generator from `/api/meta`, deployment bundle from
  `/api/health`, spend against budget as a true-ratio meter with any overrun in words, and the
  fixture-replay notice). Collapsible to a 68px rail (labels become tooltips; state is
  `html[data-sidebar]`, stamped before paint so nothing shifts). Top bar 52px on `--chrome`, no
  rule: sidebar toggle, breadcrumb `Primum / <view>`, the disclaimer, theme toggle.
- **<1024px**: the frame folds away. A sticky top bar (menu button, mark, "Primum" over the short
  disclaimer, theme) and a full-height navigation sheet (native `<dialog>`: focus trapped, Esc
  closes, backdrop click closes) holding the nav, the status card and the full disclaimer.
- The window scrolls, not the sheet: sticky elements inside views use `top-[calc(var(--topbar-h)+16px)]`
  (or `lg:top-20`); anchors get `scroll-mt-20`. The content column is `max-w-[1320px]` centred
  in the sheet; views do not add their own outer padding.
- No horizontal page scroll at 375px. Wide tables and rank columns scroll inside their own
  container: `ScrollArea` (`ui/scroll-area.tsx`), which is focusable, named, and fades its trailing
  24px while there is more to the right, so a cut column reads as "scrolls", not as a bug. First
  column sticky on `bg-surface`.
- `html` has `scroll-padding-bottom: calc(var(--composer-h) + 1rem)`. The Ask view publishes its
  docked composer's measured height as `--composer-h` (`useDockHeight`), so focus and scrolled-to
  cards always land above the composer (WCAG 2.4.11).
- Client boundaries sit at the leaves: `Sidebar.tsx`, `TopBar.tsx` and `Brand.tsx` are Server
  Components; `SidebarItem` (active state), `Breadcrumb` (pathname), `SidebarToggle`,
  `MobileNav`, `ThemeToggle` and the status widgets are the client leaves composed into them.
  Glyphs cross the boundary rendered (`icon={<item.Glyph size={16} />}`), never as functions.

## Component inventory

Import each primitive from its own module (`@/components/ui/card`, `…/button`, …); there is no
barrel. Components without
`"use client"` work in Server and Client Components; their dot-parts (`Card.Header`) work in both.
`Tabs` is a client component: in a Server Component import its named parts.

### `Card` (`ui/card.tsx`)
Every framed block uses it; do not hand-roll `rounded-card border bg-surface`.
`<Card as? variant? className? ...htmlProps>` where `as`: `section` (default) | `div` | `article`
| `aside` | `li` | `ul` | `figure`; `variant`: `default` (surface + hairline) | `inset` (raised well, no edge) |
`dashed` (empty frame) | `accent` (the one card holding the screen's answer).
Parts: `Card.Header` (flex row, wraps), `Card.Heading` (left group), `Card.Title as?="h2"|"h3"|"h4"|"p"`,
`Card.Description`, `Card.Actions` (right group), `Card.Body flush?` (flush = no side padding,
for tables and lists), `Card.Footer` (rule above, muted 13px).

### `Disclosure` (`ui/disclosure.tsx`)
`<Disclosure variant?="plain"|"framed"|"dashed" defaultOpen?>` with `<Disclosure.Summary meta? size?="sm"|"md">`
and `<Disclosure.Body>`. A native `<details>`: every expandable block (provenance, the raw file,
"How answers are gated", the withheld draft) uses it. `dashed` marks something uncertified.

### `Picker` (`ui/picker.tsx`, client)
`<Picker legend value options onChange showLabel? segmentedFrom?="sm"|"md"|"lg"|"never">`: a
`Segmented` track where it fits, a native select below that width; only one is ever displayed.

### `ScrollArea` (`ui/scroll-area.tsx`, client)
`<ScrollArea labelledBy|label className?>`: the horizontal scroller for wide tables (see Layout).

### `Button`, `IconButton`, `buttonClass` (`ui/button.tsx`)
`<Button variant? size? ...buttonProps>`: `variant` `primary` (near-black, one per screen) |
`secondary` (default; surface + control edge) | `ghost` (no edge) | `quiet` (underlined text
action); `size` `sm` (32px) | `md` (36px); both 44px on touch screens.
`<IconButton label ...buttonProps>{glyph}</IconButton>`: `label` is required (accessible name and
tooltip). `buttonClass(variant, size, className)` styles a `<Link>` as a button.

### `Badge`, `StatusDot` (`ui/badge.tsx`)
`<Badge tone?>`: `neutral` | `accent` | `supported` | `unsupported` | `contradicted` | `emergency`
| `outline`. Pale tint, dark text, 22px, never wraps. Put a glyph and a word inside; never colour
alone. `<StatusDot tone>`: `ok` | `warn` | `down` | `idle` | `busy` (pulses), decorative.

### `StatGroup`, `Stat` (`ui/stat.tsx`)
`<StatGroup columns?={2|3|4|5|6}>` is one container split by hairlines; two columns from 480px to
640px with an odd last cell spanning, and below 480px one row per stat (label and note left,
value right-aligned), so values never wrap into cramped half-width cells. Inside: `<Stat><Stat.Label/><Stat.Value unit? lead?/><Stat.Note/></Stat>`.
`lead` makes the screen's answer figure larger (40px from 1024px; once per screen); give its cell
`sm:col-span-2` so it reads first. No icons, no sparklines,
no colour on a KPI; a change is coloured only by whether it is good or bad, and grey at zero.

### `Segmented` (`ui/segmented.tsx`)
Unchanged API: `legend, name, value, options: {value,label}[], onChange, disabled?, hideLegend?`
plus `fill?` (stretch to the container below 480px) and `className?`. A radio group: arrow keys
move the selection. Use for 2 to 5 mutually exclusive settings (risk level, run picker on wide
screens).

### `Tabs` (`ui/tabs.tsx`, client)
`<Tabs value? defaultValue? onValueChange? className?>` with `<TabsList label>`, `<Tab value disabled?>`,
`<TabPanel value>` (also `Tabs.List`, `Tabs.Tab`, `Tabs.Panel`). Underline tabs on a full-width
rule, accent indicator, WAI-ARIA keyboard pattern; the list scrolls sideways on a phone. Use for
switching between views of one thing (RQ1/RQ2/RQ3, evidence/claims on mobile).

### `PageHeader` (`ui/page-header.tsx`)
`<PageHeader><PageHeader.Heading><PageHeader.Title id?/><PageHeader.Description/></PageHeader.Heading><PageHeader.Actions/></PageHeader>`.
A noun h1 at 22px in the sans, one grey sentence about the data, real actions only.

### `EmptyState`, `EmptyBlock` (`ui/empty-state.tsx`)
`<EmptyState><EmptyState.Main><EmptyState.Title/><EmptyState.Description/><EmptyState.Actions/></EmptyState.Main><EmptyState.Aside/></EmptyState>`;
with an Aside it becomes two columns on wide screens. `EmptyBlock {title, children?, action?, aside?}`
is the earlier one-call form, unchanged.

### `Skeleton`, `SkeletonText` (`ui/skeleton.tsx`)
`<Skeleton className="h-7 w-24"/>` at the real size of the value it stands for;
`<SkeletonText lines?/>`. Flat, gently pulsing, `aria-hidden`; put `aria-busy` on the region.

### `Code`, `Kbd`, glyphs
`<Code wrap?>` for commands and identifiers. `<Kbd>` for a key ("Enter"). `ui/glyphs.tsx` is the
one icon set (16px grid, 1.5px stroke, `aria-hidden`; pass `size`, `className`): Check, Cross,
Ring, External, Chevron, Sun, Moon, Stop, Warning, PrimumMark, Chat, Columns, Interval, Book,
Menu, Sidebar, Refresh, Info, Search, ArrowUp, ArrowRight, Clock, Document, Shield, Gauge,
LinkNodes, Pen, Copy, Sliders. Add new glyphs there in the same style; no icon packages, no emoji.

### Shell (`components/shell/*`)
`AppShell {sidebar, topBar, children}` · `Sidebar` compound (`Sidebar.Header/Nav/Group/Item/Footer`)
and `AppSidebar` (server) · `SidebarItem {href, label, icon}` (client) · `Brand` · `TopBar` (server)
with `Breadcrumb`, `SidebarToggle` (also lets Esc dismiss rail tooltips) · `MobileNav` · `StatusCard {title, action?}` with
`StatusCard.Row {label, stacked?}` and `SystemStatus` · `ThemeToggle` · `ApiProvider`/`useApi()`
· `ApiDown {what, aside?}` · `nav.ts` (`NAV`, `isActive`, `currentItem`, disclaimers).

### Charts (`components/charts/*`, `lib/charts/*`)
Libraries: nivo 0.99.0 (`@nivo/bar`, `line`, `heatmap`, `bump`, `swarmplot`, `theming`) and the d3
submodules `d3-scale`, `d3-shape`, `d3-array`, `d3-format`, all pinned; transitive d3 modules are
pinned in `pnpm-workspace.yaml` until their November 2026 republishes are two weeks old.

- **Data**: every plotted value comes from a pure function in `lib/charts/<area>.ts` (`pubmedqa`,
  `rq1`, `rq2`, `rq3`, `ask`, `retrieval`, shared `common`), unit-tested with `node --test`.
  Takeaway sentences are templated there too (`accuracyTakeaway`). Interval kinds are read from
  the file (`intervalInfo(data, key)`): bootstrap CI, split range, Clopper-Pearson; say which.
- **Theme**: `useChartTheme()` → `{ tokens, theme, animate, colors }` or null before mount (draw
  `ChartSkeleton`). `theme` is the one nivo theme; `animate` is false under reduced motion;
  `colors.categorical(i)` / `entity(domain)` / `emphasis(focus)` / `sequential([lo,hi])` /
  `diverging([lo,hi], mid?)` / `status(name)`. Tokens are resolved hex, re-read on theme change.
- **Frame**: `ChartCard` (figure) with `Header`, `Heading`, `Title as?`, `Takeaway`, `Actions`,
  `Legend items`, `Body height`, `Data columns rows` (the "View data" table; put `Note` and
  `Source` inside it), and `Footer` for a chart without a table. `ChartFrame` is its one-call form.
- **Layout helpers**: `Measured height` (render-prop width, for tick counts; use non-responsive
  nivo components inside), `proportionTicks(width)`, `HorizontalBarIntervals` /
  `VerticalBarIntervals` (nivo bar layers drawing `lo`–`hi` whiskers and `valueLabel`).
- **Loading**: nivo figures live in their own module and are imported with
  `next/dynamic(..., { ssr: false, loading: () => <ChartSkeleton height={h} /> })`; the height
  helper sits in a tiny separate module so the page can size the skeleton without the chart
  (`experiments/figures/accuracyChartHeight.ts`).

## Per-view composition

Each view starts with a `PageHeader` (or, for Ask, the thread itself) and composes cards from the
primitives above. Shared rules: one primary action per screen; the biggest thing answers the
screen's question; colour on data and status only; every block has a loading, empty and error
state inside its own frame; rows end level within a grid row.

### Ask (`/`): a chat-style agent workspace
Question: *what did the agent conclude, on what evidence, and is it allowed to say it?*

- **Structure**: `AskView` holds only the run (reducer, stream, stop) and three layout slots.
  `AskEmpty` (intro, readiness, suggestions, benchmark picker, recent readings, the "How answers
  are gated" disclosure), `Readiness`, `Reply` (`Question`, the agent trace and one outcome:
  `PendingDecision`, `Released`, `Withheld` + `WithheldDraft`, `Emergency`, `Failure`) and
  `Composer` are colocated in `components/ask/`. The composer's α is derived, never synced:
  `draft.alpha ?? meta.alpha ?? SPEC_DEFAULT_ALPHA`.
- **Empty state**: one sentence (what it does, that it may withhold), the mark inline at 20px in
  the h1, suggestions in a grid whose row partners end level (a note sits in the card's foot), on
  a phone two suggestions plus "more", the benchmark picker, and the threshold/operating-point
  explanation folded into "How answers are gated".
- **Thread** (centre, max ~760px of reading width): a visually hidden h1 "Ask" once a thread
  exists; the question as a **user message** (right-aligned bubble on `bg-raised`, sans
  15-16px); the answer as an **assistant message** (no bubble, mark avatar, its name row is the
  h2 and every part under it is an h3; serif `.reading` prose that streams with a caret). The
  stream is **not** a live region: one sr-only `role="status"` line announces the outcome once.
  Citation markers are superscript pills; a run of markers is grouped with the word before it in
  a nowrap span (`groupCitations` in `lib/markdown.ts`), so "cancers¹ ³." never breaks or floats
  before its full stop. When a run settles, the decision block (`[data-decision]`) is scrolled
  into view once (`block: "nearest"`, smooth only without reduced motion, `lib/motion.ts`).
  Under the answer: the **decision** (released /
  withheld / emergency) as a card: released uses the accent pointer, withheld uses the hatch and
  neutral copy ("Withheld: confidence below the certified threshold"), emergency uses the
  emergency tone with the triage message first.
- **Agent trace**: the pipeline steps (Safety triage, Concepts, Retrieval, Generation, Verification,
  Confidence) as tool-call rows: glyph (Shield, LinkNodes, Search, Pen, Check, Gauge), name, one-line
  result, timing (`fmtMs`, ClockGlyph) right-aligned in tabular figures, a status dot (busy pulse
  while running, check when done, ring when not run, with words). While running it is one compact
  strip (the running step, its live result, six progress segments and "n of 6"), so the streaming
  answer stays on screen under it; it expands on request and folds to one summary line after.
- **Evidence / sources panel** (right column ≥1280px, below the thread on narrower screens). The
  column is reserved as soon as a run starts, so the thread never moves sideways mid-stream; an
  emergency keeps it with "Nothing was retrieved". Its list is a focusable, named scroll region. numbered source cards (title,
  PMID in mono, the passage in serif, an External link); citation markers in the answer highlight
  their source. **Claims** with verdict Badges (glyph + word).
- **Confidence gauge**: the scale with the certified threshold as a dashed gate, the hatch left of
  it, the pointer in the accent settling on the reading, the released/withheld state in words, and
  the "operating point, not a guarantee" note for chat questions.
- **Composer**: a card docked at the bottom of the thread (sticky), textarea, risk level as
  `Segmented`, retriever select (the select fills its 44px chip), send (ArrowUpGlyph) as the
  primary; `Kbd` hint for Enter. One focus treatment: accent edge plus a soft 3px ring inside the
  radius, never a second outline. A 24px fade to `--ground` above the dock (the one gradient in
  the app) says the thread runs underneath. Below 640px the settings fold behind one "α 0.10"
  button, so the dock is a single ~64px row. The fixture bar is an `<aside aria-label="Fixture
  data">`, not a live region.
- **Errors** (no API key, budget, no index, server error) use the neutral card with an amber
  warning glyph; red is kept for emergencies and contradicted claims.

### Retrieval lab (`/retrieval`): a playground
Question: *where do lexical, dense and fused retrieval agree and part?*
Query bar card on top (input with a 3:1 `--control` edge, retriever chips as toggle buttons with
`aria-pressed` that wrap at every width and show "on" as ink on `bg-active` with a filled check,
not the accent, depth and corpus selects, Compare as the primary). Results as one column card per retriever, side by side
in their own horizontal scroller, rank changes drawn between columns, the selected passage's
detail in a card below (serif text). Before a query: suggested queries and the retriever glossary
as two cards that end level.

### Experiments (`/experiments`): an analytics dashboard
Question: *does the certified threshold hold, and what does it cost in coverage?*
`PageHeader` with the run picker on the right (`Segmented` on wide screens, `fill` on a phone)
and a provenance line from the index. RQ tabs show a status dot only when a file is missing,
loading or failed. A `StatGroup` of headline figures **read from the results
files** (e.g. worst violation rate of `certified` against δ, coverage range, retrieval nDCG of the
default retriever), each with a `Stat.Note` naming its source; a missing file prints a dash. RQ1-3
as `Tabs` (or sections with an on-page nav) of chart cards: `Card.Header` with the title and the
legend, the chart, a footer with n and the file. Tables keep right-aligned tabular figures, scroll
in their own container, and never use zebra rows or uppercase headers.

### Method (`/method`): documentation
Question: *how does the system decide to answer?*
Docs layout: sticky table of contents on the left (≥1024px; a collapsible "On this page" on a
phone), serif prose column at 65-72ch, analogies as margin notes on wide screens, the interactive
threshold explainer as a full-width card (`variant="accent"` is allowed for it) with its run picker
in the card header. Headings in the sans or serif consistently (serif h2 is fine here: it is the
reading surface).

## Checks

`pnpm lint`, `pnpm typecheck`, `pnpm test` must pass. Look at every route at 1440×900 and 375×812
in light and dark before calling a change done; check keyboard focus, the mobile sheet, and that
nothing scrolls the page sideways.
