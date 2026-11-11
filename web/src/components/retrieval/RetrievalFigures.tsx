"use client";

import dynamic from "next/dynamic";
import { memo, useMemo } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { overlapMatrix, overlapPairs, overlapTakeaway, scoreProfiles, scoreTakeaway } from "@/lib/charts/retrieval";
import { fmtPct, fmtProb, fmtScore } from "@/lib/format";
import type { Hit } from "@/lib/types";

import { OverlapMatrix } from "./figures/OverlapMatrix";
import { SCORE_PANEL_HEIGHT } from "./figures/sizes";
import { scoreKind } from "./retrievers";

// d3 figures load after the first ranking arrives, so the lab's first paint does not wait on them.
const ScoreProfiles = dynamic(() => import("./figures/ScoreProfiles").then((m) => m.ScoreProfiles), {
  ssr: false,
  loading: () => <ChartSkeleton height={SCORE_PANEL_HEIGHT * 2} label="Loading score profiles" />,
});

const SOURCE = "POST /api/retrieve · this query";

interface Linked {
  retrievers: string[];
  results: Record<string, Hit[]>;
  k: number;
  /** the passage pointed at, here or in the rank columns */
  activeId: string | null;
  pinnedId: string | null;
  onHover: (id: string | null) => void;
  onPin: (id: string) => void;
}

/** How many of their top k each pair of retrievers shares. */
export const OverlapCard = memo(function OverlapCard({ retrievers, results, k, className }: Pick<Linked, "retrievers" | "results" | "k"> & { className?: string }) {
  const columns = useMemo(() => retrievers.filter((r) => Array.isArray(results[r])), [results, retrievers]);
  const pairs = useMemo(() => overlapPairs(overlapMatrix(results, columns, k), columns), [results, columns, k]);
  const takeaway = overlapTakeaway(pairs, k);
  const rows = pairs.map((p) => ({ pair: `${p.a} · ${p.b}`, shared: `${p.shared} of ${k}`, jaccard: fmtProb(p.jaccard) }));

  return (
    <ChartCard className={className}>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Shared passages by pair</ChartCard.Title>
          <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway>
        </ChartCard.Heading>
      </ChartCard.Header>
      <ChartCard.Body>
        <OverlapMatrix pairs={pairs} columns={columns} k={k} />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "pair", label: "Pair" },
          { key: "shared", label: "Shared", numeric: true },
          { key: "jaccard", label: "Jaccard", numeric: true },
        ]}
        rows={rows}
      >
        <ChartCard.Note>Count of passages in both top-{k} lists</ChartCard.Note>
        <ChartCard.Source>{SOURCE}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
});

/** Each retriever's scores down its own top k, on its own scale. */
export function ScoreProfilesCard({ retrievers, results, k, activeId, pinnedId, onHover, onPin }: Linked) {
  const profiles = useMemo(() => scoreProfiles(results, retrievers, k), [results, retrievers, k]);
  const takeaway = useMemo(() => scoreTakeaway(profiles), [profiles]);
  const rows = profiles.map((p) => ({
    retriever: p.retriever,
    kind: scoreKind(p.retriever),
    top: fmtScore(p.max),
    bottom: fmtScore(p.min),
    step: p.steepest ? `${p.steepest.afterRank} → ${p.steepest.afterRank + 1}` : "–",
    share: p.steepest ? fmtPct(p.steepest.share) : "–",
  }));

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">How scores fall down each list</ChartCard.Title>
          <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway>
        </ChartCard.Heading>
        <ChartCard.Legend
          items={[
            { name: "Score at rank", color: "var(--series-rest)", mark: "dot" },
            { name: "Largest step", color: "var(--muted)", mark: "dash" },
            { name: "Traced passage", color: "var(--accent)", mark: "dot" },
          ]}
        />
      </ChartCard.Header>
      <ChartCard.Body>
        <ScoreProfiles profiles={profiles} k={k} activeId={activeId} pinnedId={pinnedId} onHover={onHover} onPin={onPin} />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "retriever", label: "Retriever" },
          { key: "kind", label: "Score" },
          { key: "top", label: "Highest", numeric: true },
          { key: "bottom", label: "Lowest", numeric: true },
          { key: "step", label: "Largest step", numeric: true },
          { key: "share", label: "Of range", numeric: true },
        ]}
        rows={rows}
      >
        <ChartCard.Note>Each panel has its own scale, from the list&apos;s lowest score, not zero; scores are not comparable across retrievers</ChartCard.Note>
        <ChartCard.Source>{SOURCE}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}
