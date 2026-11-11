"use client";

import dynamic from "next/dynamic";
import { useMemo, useState, type ReactNode } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Picker } from "@/components/ui/picker";
import {
  ablation,
  ablationGroup,
  ablationTakeaway,
  brierTakeaway,
  estimatorKind,
  estimatorLabel,
  estimatorMetric,
  estimatorRanking,
  learnedNames,
  learnedVsStated,
  mainCalibrator,
  rankingTakeaway,
  reliabilityBins,
  reliabilityTakeaway,
  riskCoverageSeries,
  riskCoverageTakeaway,
  rq2Interval,
  signed,
  statedConfidence,
  weightedGap,
} from "@/lib/charts/rq2";
import { fmtInt, fmtMetric, fmtP, fmtPct } from "@/lib/format";
import { findMetricTables, readUnmeasured, runPathPrefix } from "@/lib/results";

import { ablationHeight, brierHeight, RELIABILITY_HEIGHT, rankingHeight, RISK_COVERAGE_HEIGHT } from "./charts/rq2-heights";
import { RawFile, Unrecognised } from "./parts";
import { groupTables, metaOf } from "./readers";
import { TableGroupCard } from "./TableGroupCard";

/* The figures (nivo, d3) load on the client only, each behind a skeleton at its own height. */
/* Each ChartCard.Body below is given the chart's real height, so the skeleton just fills it. */
const fill = () => <ChartSkeleton height="100%" />;
const RankingChart = dynamic(() => import("./charts/rq2-ranking-chart").then((m) => m.RankingChart), { ssr: false, loading: fill });
const BrierDiffChart = dynamic(() => import("./charts/rq2-brier-chart").then((m) => m.BrierDiffChart), { ssr: false, loading: fill });
const AblationChart = dynamic(() => import("./charts/rq2-ablation-chart").then((m) => m.AblationChart), { ssr: false, loading: fill });
const RiskCoverageChart = dynamic(() => import("./charts/rq2-risk-coverage-chart").then((m) => m.RiskCoverageChart), { ssr: false, loading: fill });
const ReliabilityChart = dynamic(() => import("./charts/rq2-reliability-chart").then((m) => m.ReliabilityChart), { ssr: false, loading: fill });

const range3 = (lo: number | null, hi: number | null) => (lo !== null && hi !== null ? `${fmtMetric(lo)} to ${fmtMetric(hi)}` : "–");
const diff4 = (v: number | null | undefined) => (v === null || v === undefined ? "–" : signed(v, 4));

/** RQ2 drops features a run did not measure (constant columns); say which, from its `meta`. */
function Unmeasured({ data }: { data: unknown }) {
  const { features, featureSets } = readUnmeasured(data);
  if (features.length === 0 && featureSets.length === 0) return null;
  return (
    <p className="rounded-card border border-dashed border-frame px-4 py-3 text-sm leading-relaxed text-muted sm:px-5">
      <span className="font-medium text-ink">Not measured in this run. </span>
      {features.length > 0 ? (
        <>
          {features.join(", ")} {features.length === 1 ? "was" : "were"} not measured, so {features.length === 1 ? "it is" : "they are"} left out
          of every model and of the ablation.{" "}
        </>
      ) : null}
      {featureSets.length > 0 ? `The ${featureSets.join(", ")} feature set${featureSets.length === 1 ? " was" : "s were"} skipped.` : null}
    </p>
  );
}

/** The note-and-source row every RQ2 card ends in. */
function Foot({ note, source }: { note: ReactNode; source: string }) {
  return (
    <>
      <ChartCard.Note>{note}</ChartCard.Note>
      <ChartCard.Source>{source}</ChartCard.Source>
    </>
  );
}

/**
 * RQ2, confidence: which signals predict a correct answer, and whether a learned combination beats
 * the generator's stated confidence. Every figure is derived in `lib/charts/rq2.ts` from the file;
 * the takeaways are templated there too. `benchmark` is the slot for the end-to-end runs the
 * estimators are trained on (`BenchmarkSection`), drawn above the raw file.
 */
export function ConfidenceSection({ data, run, benchmark }: { data: unknown; run: string; benchmark?: ReactNode }) {
  const file = `${runPathPrefix(run)}rq2_confidence`;
  const source = `results/${file}.json`;

  const d = useMemo(() => {
    const calibrator = mainCalibrator(data);
    const ranking = calibrator ? estimatorRanking(data, calibrator, "auroc") : [];
    const comparison = learnedVsStated(data);
    const abl = ablation(data, "auroc");
    const riskCoverage = riskCoverageSeries(data);
    const reliability = reliabilityBins(data);
    const stated = statedConfidence(data);
    const reference = stated?.name ?? [...riskCoverage, ...reliability].find((s) => estimatorKind(s.name) === "stated")?.name ?? null;
    const learned = learnedNames([...new Set([...riskCoverage, ...reliability].map((s) => s.name))]);
    const tables = findMetricTables(data).filter((table) => table.path[0] !== "comparisons");
    return {
      tables,
      groups: groupTables(tables),
      brierRows: comparison?.rows.filter((r) => r.brier) ?? [],
      calibrator,
      ranking,
      comparison,
      abl,
      riskCoverage,
      reliability,
      reference,
      learned,
      intervals: {
        estimators: rq2Interval(data, "estimators"),
        brier: rq2Interval(data, "brier_diff"),
        auroc: rq2Interval(data, "auroc_diff"),
        reliability: rq2Interval(data, "reliability"),
        riskCoverage: rq2Interval(data, "risk_coverage"),
      },
    };
  }, [data]);

  // The learned model set against stated confidence in the two curve figures; derived, not synced.
  const [picked, setPicked] = useState<string | null>(null);
  const focus = picked !== null && d.learned.includes(picked) ? picked : (d.learned[0] ?? null);

  const { tables, groups, brierRows } = d;
  const nothing = d.ranking.length === 0 && d.riskCoverage.length === 0 && d.reliability.length === 0 && tables.length === 0;
  const calibratorNote = d.calibrator ? `, ${d.calibrator} calibration` : "";

  const rankingText = rankingTakeaway(d.ranking, d.comparison, d.intervals.auroc.label);
  const brierText = brierTakeaway(d.comparison);
  const ablationReference = d.abl.rows.find((r) => r.name === d.abl.reference) ?? null;
  const ablationText = ablationTakeaway(d.abl);
  const rcText = d.reference ? riskCoverageTakeaway(d.riskCoverage, d.reference, focus) : null;
  const relText = d.reference
    ? reliabilityTakeaway(d.reliability, [d.reference, ...(focus ? [focus] : [])], (name) => estimatorMetric(data, d.calibrator, name, "ece"))
    : null;
  const curveLegend = [
    ...(d.reference ? [{ name: estimatorLabel(d.reference), color: "var(--series-1)", mark: "line" as const }] : []),
    ...(focus ? [{ name: focus, color: "var(--series-2)", mark: "line" as const }] : []),
    { name: "other estimators", color: "var(--series-rest)", mark: "line" as const, faint: true },
  ];
  const meta = metaOf(data);
  const estimatorGroup = groups.find((g) => g.key === "estimators");
  const ablationTable = groups.find((g) => g.key === "ablation")?.tables[0];
  const otherGroups = groups.filter((g) => g !== estimatorGroup && g.key !== "ablation");

  const ablationBars = d.abl.rows.filter((r) => r.deltaFromAll !== null).length;

  return (
    <div className="space-y-4">
      {nothing ? <Unrecognised file={file} /> : null}
      <Unmeasured data={data} />

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
        {d.ranking.length > 0 ? (
          <div className="grid min-w-0 gap-3">
            <ChartCard>
              <ChartCard.Header>
                <ChartCard.Heading>
                  <ChartCard.Title as="h3">Which signals separate right answers from wrong ones?</ChartCard.Title>
                  {rankingText ? <ChartCard.Takeaway>{rankingText}</ChartCard.Takeaway> : null}
                </ChartCard.Heading>
                <ChartCard.Legend
                  items={[
                    { name: "stated confidence (reference)", color: "var(--series-1)", mark: "dot" },
                    { name: "learned combination", color: "var(--series-2)", mark: "diamond" },
                    { name: "single signal", color: "var(--series-rest)", mark: "ring" },
                  ]}
                />
              </ChartCard.Header>
              <ChartCard.Body height={rankingHeight(d.ranking.length)}>
                <RankingChart
                  ranking={d.ranking}
                  intervalLabel={d.intervals.estimators.label}
                  ariaLabel={`AUROC of ${fmtInt(d.ranking.length)} confidence estimators, best mean first. ${rankingText ?? ""}`}
                />
              </ChartCard.Body>
              <ChartCard.Data
                columns={[
                  { key: "name", label: "Estimator" },
                  { key: "kind", label: "Kind" },
                  { key: "auroc", label: "Mean AUROC", numeric: true },
                  { key: "range", label: d.intervals.estimators.label, numeric: true },
                ]}
                rows={d.ranking.map((r) => ({ name: r.name, kind: r.kind, auroc: fmtMetric(r.estimate.value), range: range3(r.estimate.lo, r.estimate.hi) }))}
              >
                <Foot
                  note={`Whiskers: ${d.intervals.estimators.label}${calibratorNote}. A split range shows how much one split varies, not a CI of the mean. signal:verbal_conf is the stated confidence itself, so it is not drawn twice.`}
                  source={source}
                />
              </ChartCard.Data>
            </ChartCard>
            {estimatorGroup ? (
              <TableGroupCard
                tables={estimatorGroup.tables}
                title="Every metric for every estimator"
                fallback="Confidence estimators"
                rowLabel="Estimator"
                variantLegend="Calibration"
                preferred={d.calibrator}
                reference={d.reference ?? undefined}
                defaultMetric="ece"
                variant="disclosure"
                source={source}
              />
            ) : null}
          </div>
        ) : null}

        <div className="grid min-w-0 gap-4">
          {brierRows.length > 0 && d.comparison ? (
            <ChartCard>
              <ChartCard.Header>
                <ChartCard.Heading>
                  <ChartCard.Title as="h3">Does a learned model give better probabilities than stated confidence?</ChartCard.Title>
                  {brierText ? <ChartCard.Takeaway>{brierText}</ChartCard.Takeaway> : null}
                </ChartCard.Heading>
              </ChartCard.Header>
              <ChartCard.Body height={brierHeight(brierRows.length)}>
                <BrierDiffChart
                  rows={brierRows}
                  reference={d.comparison.reference}
                  intervalLabel={d.intervals.brier.label}
                  ariaLabel={`Brier loss of each learned model minus that of ${d.comparison.reference}. ${brierText ?? ""}`}
                />
              </ChartCard.Body>
              <ChartCard.Data
                columns={[
                  { key: "model", label: "Learned model" },
                  { key: "brier", label: "Brier difference", numeric: true },
                  { key: "brierCi", label: d.intervals.brier.label, numeric: true },
                  { key: "p", label: "p", numeric: true },
                  { key: "auroc", label: "AUROC difference", numeric: true },
                  { key: "aurocRange", label: d.intervals.auroc.label, numeric: true },
                  { key: "share", label: "Splits with higher AUROC", numeric: true },
                ]}
                rows={(d.comparison.rows ?? []).map((r) => ({
                  model: r.model,
                  brier: diff4(r.brier?.value),
                  brierCi: r.brier && r.brier.lo !== null && r.brier.hi !== null ? `${diff4(r.brier.lo)} to ${diff4(r.brier.hi)}` : "–",
                  p: fmtP(r.brierP),
                  auroc: r.auroc ? signed(r.auroc.value) : "–",
                  aurocRange: r.auroc && r.auroc.lo !== null && r.auroc.hi !== null ? `${signed(r.auroc.lo)} to ${signed(r.auroc.hi)}` : "–",
                  share: r.aurocBetterShare === null ? "–" : fmtPct(r.aurocBetterShare),
                }))}
              >
                <Foot note={`Whiskers: ${d.intervals.brier.label}, learned minus ${d.comparison.reference}${calibratorNote}.`} source={source} />
              </ChartCard.Data>
            </ChartCard>
          ) : null}

          {ablationReference && ablationBars > 0 ? (
            <div className="grid min-w-0 gap-3">
              <ChartCard>
                <ChartCard.Header>
                  <ChartCard.Heading>
                    <ChartCard.Title as="h3">Which feature groups does the learned model need?</ChartCard.Title>
                    {ablationText ? <ChartCard.Takeaway>{ablationText}</ChartCard.Takeaway> : null}
                  </ChartCard.Heading>
                </ChartCard.Header>
                <ChartCard.Body height={ablationHeight(ablationBars)}>
                  <AblationChart
                    rows={d.abl.rows}
                    reference={ablationReference}
                    ariaLabel={`Change in mean AUROC when each feature group is dropped, against ${fmtMetric(ablationReference.estimate.value)} with all features. ${ablationText ?? ""}`}
                  />
                </ChartCard.Body>
                <ChartCard.Data
                  columns={[
                    { key: "set", label: "Feature set" },
                    { key: "auroc", label: "Mean AUROC", numeric: true },
                    { key: "range", label: d.intervals.estimators.label, numeric: true },
                    { key: "delta", label: "Change from all features", numeric: true },
                  ]}
                  rows={d.abl.rows.map((r) => ({
                    set: r.deltaFromAll === null ? r.name : `without ${ablationGroup(r.name)}`,
                    auroc: fmtMetric(r.estimate.value),
                    range: range3(r.estimate.lo, r.estimate.hi),
                    delta: r.deltaFromAll === null ? "reference" : signed(r.deltaFromAll),
                  }))}
                >
                  <Foot note="Bars: difference of mean AUROC from the full model, left lowers it, right raises it. The file has no interval for the difference, so none is drawn." source={source} />
                </ChartCard.Data>
              </ChartCard>
              {ablationTable ? (
                <TableGroupCard
                  tables={[ablationTable]}
                  title="Every metric for each feature set"
                  fallback="Feature sets"
                  rowLabel="Feature set"
                  variantLegend="Table"
                  plot={false}
                  variant="disclosure"
                  source={source}
                />
              ) : null}
            </div>
          ) : null}
        </div>
      </div>

      {d.riskCoverage.length > 0 || d.reliability.length > 0 ? (
        <div className="space-y-3 pt-2">
          {d.learned.length > 1 && d.reference ? (
            <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
              <Picker
                legend="Set against stated confidence"
                showLabel
                value={focus ?? ""}
                options={d.learned.map((name) => ({ value: name, label: name }))}
                onChange={setPicked}
                segmentedFrom="md"
              />
            </div>
          ) : null}
          <div className="grid gap-4 lg:grid-cols-2">
            {d.riskCoverage.length > 0 && d.reference ? (
              <ChartCard>
                <ChartCard.Header>
                  <ChartCard.Heading>
                    <ChartCard.Title as="h3">How does the error rate rise as the system answers more?</ChartCard.Title>
                    {rcText ? <ChartCard.Takeaway>{rcText}</ChartCard.Takeaway> : null}
                  </ChartCard.Heading>
                  <ChartCard.Legend items={curveLegend} />
                </ChartCard.Header>
                <ChartCard.Body height={RISK_COVERAGE_HEIGHT}>
                  <RiskCoverageChart series={d.riskCoverage} reference={d.reference} focus={focus} ariaLabel={`Risk-coverage curves. ${rcText ?? ""}`} />
                </ChartCard.Body>
                <ChartCard.Data
                  columns={[
                    { key: "coverage", label: "Coverage", numeric: true },
                    ...d.riskCoverage.map((s) => ({ key: s.name, label: s.name, numeric: true })),
                  ]}
                  rows={(d.riskCoverage[0]?.points ?? [])
                    .filter((_, i, all) => i % 10 === 9 || i === all.length - 1)
                    .map((p) => ({
                      coverage: fmtPct(p.coverage),
                      ...Object.fromEntries(d.riskCoverage.map((s) => [s.name, fmtPct(s.points.find((q) => Math.abs(q.coverage - p.coverage) < 1e-9)?.risk, 1)])),
                    }))}
                >
                  <Foot note={`Curves: ${d.intervals.riskCoverage.text ?? "mean over splits"}; no interval drawn. Table: every tenth grid point.`} source={source} />
                </ChartCard.Data>
              </ChartCard>
            ) : null}

            {d.reliability.length > 0 && d.reference ? (
              <ChartCard>
                <ChartCard.Header>
                  <ChartCard.Heading>
                    <ChartCard.Title as="h3">When it says 70%, is it right 70% of the time?</ChartCard.Title>
                    {relText ? <ChartCard.Takeaway>{relText}</ChartCard.Takeaway> : null}
                  </ChartCard.Heading>
                  <ChartCard.Legend
                    items={[
                      { name: estimatorLabel(d.reference), color: "var(--series-1)", mark: "dot", joined: true },
                      ...(focus ? [{ name: focus, color: "var(--series-2)", mark: "diamond" as const, joined: true }] : []),
                    ]}
                  />
                </ChartCard.Header>
                <ChartCard.Body height={RELIABILITY_HEIGHT}>
                  <ReliabilityChart series={d.reliability} reference={d.reference} focus={focus} ariaLabel={`Reliability diagram. ${relText ?? ""}`} />
                </ChartCard.Body>
                <ChartCard.Data
                  columns={[
                    { key: "model", label: "Estimator" },
                    { key: "bin", label: "Bin", numeric: true },
                    { key: "conf", label: "Mean predicted", numeric: true },
                    { key: "acc", label: "Observed correct", numeric: true },
                    { key: "n", label: "Predictions", numeric: true },
                  ]}
                  rows={d.reliability.flatMap((s) =>
                    s.bins.map((b) => ({ model: s.name, bin: `${fmtPct(b.lo)}–${fmtPct(b.hi)}`, conf: fmtPct(b.confidence, 1), acc: fmtPct(b.accuracy, 1), n: fmtInt(b.count) })),
                  )}
                  caption={`Weighted distance from the diagonal, pooled over repeats: ${d.reliability.map((s) => `${s.name} ${fmtPct(weightedGap(s), 1)}`).join(", ")}.`}
                >
                  <Foot
                    note={`Points: ${d.intervals.reliability.text ?? "pooled test predictions"}${typeof meta.ece_bins === "number" ? `, ${meta.ece_bins} bins` : ""}; empty bins left out. Strip: share of predictions per bin. Pooled, so it differs from the per-split ECE in the table.`}
                    source={source}
                  />
                </ChartCard.Data>
              </ChartCard>
            ) : null}
          </div>
        </div>
      ) : null}

      {otherGroups.map((group) =>
        group.tables.map((table) => (
          <TableGroupCard
            key={table.path.join("/")}
            tables={[table]}
            fallback="Confidence estimators"
            rowLabel="Estimator"
            variantLegend="Table"
            variant="disclosure"
            source={source}
          />
        )),
      )}

      {benchmark}
      <RawFile data={data} name={file} />
    </div>
  );
}
