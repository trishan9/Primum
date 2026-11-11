"use client";

import dynamic from "next/dynamic";
import { useMemo } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Code } from "@/components/ui/code";
import { accuracyByCondition, accuracyTakeaway, conditionLabel, pairedConditionDiffs, PUBMEDQA_INTERVAL } from "@/lib/charts/pubmedqa";
import { fmtInt, fmtP, fmtPct } from "@/lib/format";
import { findMetricTables, isRecord, runPathPrefix } from "@/lib/results";

import { ConfidenceCard, ConfusionCard } from "./charts/benchmark-cards";
import { accuracyChartHeight } from "./figures/accuracyChartHeight";
import { MetricTableBlock } from "./MetricTableBlock";
import { metaOf } from "./readers";
import { useExperiment } from "./useExperiment";

/*
 * The lazy-loading pattern for every nivo figure: the chart module (and nivo with it) is split
 * out of the page bundle, never rendered on the server, and a skeleton at the chart's real height
 * holds its place. The skeleton's height must match the chart's, so the height helper lives in
 * a tiny module the page can import without pulling the chart in.
 */
const AccuracyByConditionChart = dynamic(() => import("./figures/AccuracyByConditionChart").then((m) => m.AccuracyByConditionChart), {
  ssr: false,
  loading: () => <ChartSkeleton height="100%" />,
});

/** A difference of shares in percentage points: "+16.9 pts". */
const points = (v: number) => `${v >= 0 ? "+" : "\u2212"}${(Math.abs(v) * 100).toFixed(1)} pts`;

const CONDITION_NAMES: Record<string, string> = {
  closed_book: "closed book",
  rag: "with retrieval (rag)",
  oracle: "with the gold abstract (oracle)",
};

/**
 * The end-to-end runs RQ2 and RQ3 are built on (`results/<run>/pubmedqa_runs.json`): answer
 * accuracy closed-book, with retrieval, and with the gold abstract. The accuracy chart reads its
 * summary and paired tests (both in its data table); the confusion matrix and the
 * stated-confidence histogram are counted from its per-question records. The summary's other
 * columns sit in a closed "All metrics" table at the end.
 */
export function BenchmarkSection({ run }: { run: string }) {
  const runs = useExperiment("pubmedqa_runs", run);
  const prefix = runPathPrefix(run);
  const data = runs.status === "ready" ? runs.data : null;
  const accuracy = useMemo(() => (data ? accuracyByCondition(data) : []), [data]);
  const diffs = useMemo(() => (data ? pairedConditionDiffs(data) : []), [data]);

  if (runs.status === "loading") {
    return <div aria-busy="true" className="h-48 animate-[skeleton-pulse_1.4s_ease-in-out_infinite] rounded-card bg-raised" />;
  }
  if (runs.status !== "ready") {
    return runs.status === "missing" ? (
      <p className="text-sm text-muted">
        The end-to-end runs these estimators are trained on have not been written yet: run <Code>primum exp pubmedqa</Code>.
      </p>
    ) : null;
  }
  // A run with one condition is still a table (one row), so pass minRows = 1 here.
  const summary = isRecord(data) && isRecord(data.summary) ? findMetricTables(data.summary, 5, 1) : [];
  const meta = metaOf(data);
  const incomplete = meta.complete === false;
  // The conditions this run answered under, read from the file; the ones it did not are named
  // so that an absent row reads as "not run", not as a gap in the table.
  const pubmedqa = isRecord(data) && isRecord(data.summary) && isRecord(data.summary.pubmedqa) ? data.summary.pubmedqa : {};
  const conditions = Object.keys(pubmedqa);
  const notRun = Object.keys(CONDITION_NAMES).filter((c) => !conditions.includes(c));
  const conditionList = conditions.map((c) => CONDITION_NAMES[c] ?? c).join(", ");
  const source = `results/${prefix}pubmedqa_runs.json`;
  const charted = accuracy.filter((r) => r.accuracy);
  const takeaway = accuracyTakeaway(accuracy, diffs);

  return (
    <div className="space-y-4">
      {incomplete || (conditions.length > 0 && notRun.length > 0) ? (
        <div className="space-y-1 rounded-card bg-raised px-4 py-3 text-sm leading-relaxed text-ink sm:px-5">
          {incomplete ? (
            <p className="font-medium">
              This run is marked incomplete{typeof meta.stopped === "string" ? ` (${meta.stopped})` : ""}: its numbers cover only the questions
              answered before it stopped.
            </p>
          ) : null}
          {conditions.length > 0 && notRun.length > 0 ? (
            <p className="text-muted">
              This run answered only {conditionList}. It has no {notRun.map((c) => c.replace("_", "-")).join(" or ")} condition, so there is no
              accuracy comparison between conditions for this generator.
            </p>
          ) : null}
        </div>
      ) : null}

      <div className="grid items-start gap-4 xl:grid-cols-2">
        {charted.length > 0 ? (
          <ChartCard>
            <ChartCard.Header>
              <ChartCard.Heading>
                <ChartCard.Title as="h3">How often is the answer right, by evidence condition?</ChartCard.Title>
                {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
              </ChartCard.Heading>
            </ChartCard.Header>
            <ChartCard.Body height={accuracyChartHeight(charted.length)}>
              <AccuracyByConditionChart rows={charted} ariaLabel={`Answer accuracy by evidence condition. ${takeaway ?? ""}`} />
            </ChartCard.Body>
            <ChartCard.Data
              columns={[
                { key: "condition", label: "Condition or comparison" },
                { key: "accuracy", label: "Accuracy or difference", numeric: true },
                { key: "ci", label: "95% bootstrap CI", numeric: true },
                { key: "p", label: "p (paired)", numeric: true },
                { key: "n", label: "Questions", numeric: true },
              ]}
              rows={[
                ...charted.map((r) => ({
                  condition: r.label,
                  accuracy: fmtPct(r.accuracy?.value, 1),
                  ci: r.accuracy && r.accuracy.lo !== null && r.accuracy.hi !== null ? `${fmtPct(r.accuracy.lo, 1)} to ${fmtPct(r.accuracy.hi, 1)}` : "–",
                  p: "–",
                  n: fmtInt(r.n),
                })),
                ...diffs.map((d) => ({
                  condition: `${conditionLabel(d.a)} minus ${conditionLabel(d.b).toLowerCase()}`,
                  accuracy: d.diff ? points(d.diff.value) : "–",
                  ci: d.diff && d.diff.lo !== null && d.diff.hi !== null ? `${points(d.diff.lo)} to ${points(d.diff.hi)}` : "–",
                  p: fmtP(d.p),
                  n: d.nPairs === null ? "–" : `${fmtInt(d.nPairs)} pairs`,
                })),
              ]}
            >
              <ChartCard.Note>Whiskers: {PUBMEDQA_INTERVAL.text}. Differences in percentage points, same questions under both conditions.</ChartCard.Note>
              <ChartCard.Source>{source}</ChartCard.Source>
            </ChartCard.Data>
          </ChartCard>
        ) : null}
        <ConfusionCard data={data} source={source} />
      </div>

      <ConfidenceCard data={data} source={source} />

      {summary.map((table) => (
        <MetricTableBlock
          key={`summary/${table.path.join("/")}`}
          table={table}
          title="Every summary metric, by evidence condition"
          rowLabel="Condition"
          plot={false}
          variant="disclosure"
          source={source}
        />
      ))}
    </div>
  );
}
