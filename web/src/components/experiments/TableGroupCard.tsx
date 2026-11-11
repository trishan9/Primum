"use client";

import { useState } from "react";

import type { MetricTable } from "@/lib/results";

import { MetricTableBlock } from "./MetricTableBlock";
import { Picker } from "@/components/ui/picker";

const DATASET_TITLES: Record<string, string> = {
  nfcorpus: "NFCorpus test queries",
  pubmedqa: "PubMedQA gold-abstract retrieval",
};

const SHORT_NAMES: Record<string, string> = {
  nfcorpus: "NFCorpus",
  pubmedqa: "PubMedQA",
  lac: "LAC",
  aps: "APS",
};

/**
 * A heading from where the table sits in the file: `paired_tests/nfcorpus/vs_bm25` reads
 * "Paired tests against bm25, NFCorpus test queries". A name after `vs_` is kept as written.
 */
function tableTitle(path: readonly string[], fallback: string): string {
  if (path.length === 0) return fallback;
  const known = path.map((key) => DATASET_TITLES[key.toLowerCase()]).find(Boolean);
  let joined = "";
  for (const key of path) {
    if (DATASET_TITLES[key.toLowerCase()]) continue;
    const vs = key.match(/^vs[_-](.+)$/);
    joined += vs ? ` against ${vs[1]}` : `${joined ? ", " : ""}${key.replace(/_/g, " ")}`;
  }
  joined = [joined.trim(), known].filter(Boolean).join(", ");
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

/** A short switch label: `nfcorpus/vs_bm25` reads "NFCorpus vs bm25", `platt` reads "Platt". */
function variantLabel(path: readonly string[], fallback: string): string {
  if (path.length === 0) return fallback;
  const words = path.map((key) => {
    const vs = key.match(/^vs[_-](.+)$/);
    if (vs) return `vs ${vs[1]}`;
    return SHORT_NAMES[key.toLowerCase()] ?? key.replace(/_/g, " ");
  });
  const label = words.join(" ");
  return label.charAt(0).toUpperCase() + label.slice(1);
}

const keyOf = (table: MetricTable) => table.path.join("/");

interface TableGroupCardProps {
  tables: MetricTable[];
  /** The group's heading; a table on its own gets its own title from its path when this is omitted. */
  title?: string;
  /** Used when the path says nothing. */
  fallback: string;
  rowLabel: string;
  /** What the switch between variants picks: "Calibration", "Dataset". */
  variantLegend: string;
  /** The variant to open on, by the last key of its path, when it exists. */
  preferred?: string | null;
  reference?: string;
  source?: string;
  /** Passed to MetricTableBlock: draw the plotted column, which column first, card or "All metrics" disclosure. */
  plot?: boolean;
  defaultMetric?: string;
  variant?: "card" | "disclosure";
}

/**
 * Variants of one table (the same estimators under four calibrators, the same retrievers on two
 * datasets) in one card, with a switch in its header. The numbers are the file's; the switch only
 * chooses which table is drawn.
 */
export function TableGroupCard({ tables, title, fallback, rowLabel, variantLegend, preferred, reference, source, plot, defaultMetric, variant }: TableGroupCardProps) {
  const fallbackKey = (preferred ? tables.find((t) => t.path[t.path.length - 1] === preferred) : undefined) ?? tables[0];
  const [picked, setPicked] = useState<string | null>(null);
  const table = tables.find((t) => keyOf(t) === picked) ?? fallbackKey;
  if (!table) return null;

  // A variant is named by what sets it apart within the group (the keys after the shared first one).
  const prefix = tables.length > 1 && tables.every((t) => t.path.length > 1) ? 1 : 0;
  const options = tables.map((t) => ({ value: keyOf(t), label: variantLabel(t.path.slice(prefix), fallback) }));
  const longest = Math.max(...options.map((o) => o.label.length));
  const fits = options.length <= 4 && longest * options.length <= 64;

  return (
    <MetricTableBlock
      table={table}
      title={title ?? tableTitle(table.path, fallback)}
      rowLabel={rowLabel}
      reference={reference}
      source={source}
      plot={plot}
      defaultMetric={defaultMetric}
      variant={variant}
      actions={
        tables.length > 1 ? (
          <Picker legend={variantLegend} showLabel value={keyOf(table)} options={options} onChange={setPicked} segmentedFrom={fits ? "sm" : "never"} />
        ) : null
      }
    />
  );
}
