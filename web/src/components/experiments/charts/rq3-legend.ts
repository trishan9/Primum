import type { LegendItem } from "@/components/charts/ChartCard";

/*
 * Legend items for the RQ3 figures, as CSS values, in a module apart from the (lazy) figures so a
 * section can draw its legend without importing them. They mirror `methodStyle`, `runStyle` and
 * `setStyle` in rq3-figures.tsx: orange is "plug-in, uncorrected" only; the certified rules wear
 * the accent; generators are told apart by mark and line style in ink and grey.
 */

export function methodLegendItem(method: string, name: string): LegendItem {
  if (method === "plug_in") return { name, color: "var(--series-2)", mark: "diamond", joined: true };
  if (method === "certified") return { name, color: "var(--series-1)", mark: "dot", joined: true };
  if (method === "certified_hb") return { name, color: "var(--series-1)", mark: "square", joined: true, dashed: true };
  return { name, color: "var(--series-rest)", mark: "dot", joined: true };
}

const RUN_MARKS: readonly LegendItem["mark"][] = ["dot", "diamond-open", "square"];

export function runLegendItem(name: string, i: number): LegendItem {
  return { name, color: i === 0 ? "var(--text)" : "var(--series-rest)", mark: RUN_MARKS[i] ?? "dot", joined: true, dashed: i > 0 };
}
