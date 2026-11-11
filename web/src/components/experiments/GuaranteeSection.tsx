"use client";

import dynamic from "next/dynamic";
import { useMemo, type ReactNode } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { cn } from "@/components/ui/cn";
import { Disclosure } from "@/components/ui/disclosure";
import { CrossGlyph } from "@/components/ui/glyphs";
import { getExperiment, getExperimentsIndex } from "@/lib/api";
import {
  alphaLevels,
  certifiedSplitViolations,
  compareRuns,
  compareTakeaway,
  coverageTakeaway,
  guaranteeSeries,
  methodLabel,
  predictionSets,
  predictionSetTakeaway,
  rq3Interval,
  selectiveDomain,
  violationTakeaway,
} from "@/lib/charts/rq3";
import type { Estimate } from "@/lib/charts/common";
import { fmtAlpha, fmtInt, fmtPct } from "@/lib/format";
import { findMetricTables, levelsAsRows, runLabel, runPathPrefix, type LevelRecord } from "@/lib/results";

import { COMPARE_HEIGHT, COVERAGE_HEIGHT, SETS_HEIGHT, VIOLATION_HEIGHT } from "./charts/rq3-heights";
import { methodLegendItem, runLegendItem } from "./charts/rq3-legend";
import { RawFile, Unrecognised } from "./parts";
import { groupTables, neverCertified, type Rq3Summary } from "./readers";
import { TableGroupCard } from "./TableGroupCard";
import { useLoaded } from "./useExperiment";

/* The figures load on the client only, each behind a skeleton at its own height. */
const figures = () => import("./charts/rq3-figures");
const ViolationChart = dynamic(() => figures().then((m) => m.ViolationChart), { ssr: false, loading: () => <ChartSkeleton height={VIOLATION_HEIGHT} /> });
const CoverageChart = dynamic(() => figures().then((m) => m.CoverageChart), { ssr: false, loading: () => <ChartSkeleton height={COVERAGE_HEIGHT} /> });
const CompareRunsPanel = dynamic(() => figures().then((m) => m.CompareRunsPanel), { ssr: false, loading: () => <ChartSkeleton height={COMPARE_HEIGHT} /> });
const PredictionSetPanel = dynamic(() => figures().then((m) => m.PredictionSetPanel), { ssr: false, loading: () => <ChartSkeleton height={SETS_HEIGHT} /> });

const setLegendColor = (method: string) => (/lac/i.test(method) ? "var(--series-1)" : "var(--series-rest)");

const pct1 = (e: Estimate | null | undefined) => (e ? fmtPct(e.value, 1) : "–");
const range1 = (e: Estimate | null | undefined) => (e && e.lo !== null && e.hi !== null ? `${fmtPct(e.lo, 1)} to ${fmtPct(e.hi, 1)}` : "–");

/** The note-and-source row every RQ3 card ends in. */
function Foot({ note, source }: { note: ReactNode; source: string }) {
  return (
    <>
      <ChartCard.Note>{note}</ChartCard.Note>
      <ChartCard.Source>{source}</ChartCard.Source>
    </>
  );
}

/**
 * What the file says about the certified method, read from its records rather than written in.
 * Levels where no split certified a threshold abstain on everything, so their zero violation rate
 * is trivial; they are named separately and left out of the summary ranges.
 */
function Finding({ summary, alpha }: { summary: Rq3Summary; alpha: number | null }) {
  const { records, delta } = summary;
  const allCertified = records.filter((r) => r.method === "certified");
  const uncertifiedAlphas = allCertified.filter(neverCertified).map((r) => r.alpha);
  const certified = allCertified.filter((r) => !neverCertified(r));
  const rates = certified.flatMap((r) => (r.certifiedRate !== null ? [r.certifiedRate] : []));
  const violations = certified.flatMap((r) => (r.violation ? [r.violation.value] : []));
  const coverage = certified.flatMap((r) => (r.coverage ? [r.coverage.value] : []));
  if (violations.length === 0 || coverage.length === 0 || delta === null) return null;
  const worst = Math.max(...violations);

  return (
    <section
      aria-labelledby="rq3-finding"
      className="grid gap-x-8 gap-y-5 rounded-card border border-accent/40 bg-surface px-4 py-4 sm:px-5 xl:grid-cols-[minmax(0,1fr)_minmax(0,26rem)]"
    >
      <div className="min-w-0">
      <h3 id="rq3-finding" className="text-[0.9375rem] font-semibold leading-6 text-ink">
        What the certified threshold did on held-out questions
      </h3>
      <div className="mt-1.5 max-w-[78ch] space-y-2 text-sm leading-relaxed text-ink">
        {uncertifiedAlphas.length > 0 ? (
          <p>
            At α = {uncertifiedAlphas.map(fmtAlpha).join(", ")} no threshold was certified in any split, so the system abstained on everything there
            and its zero violation rate says nothing about the certificate.
          </p>
        ) : null}
        <p>
          {uncertifiedAlphas.length > 0 ? "At the levels that did certify, the" : "For the certified threshold, the"} share of splits in which test
          risk exceeded α was at most <span className="num font-semibold">{fmtPct(worst, 1)}</span>, {worst <= delta ? "within" : "above"} δ ={" "}
          <span className="num">{fmtAlpha(delta)}</span>. A split that certifies nothing counts as no violation
          {rates.length > 0 ? `, and a threshold was certified in only ${fmtPct(Math.min(...rates), 1)} to ${fmtPct(Math.max(...rates), 1)} of splits` : ""}.
          Mean coverage ranged from <span className="num">{fmtPct(Math.min(...coverage), 1)}</span> to{" "}
          <span className="num">{fmtPct(Math.max(...coverage), 1)}</span> of test questions
          {Math.max(...coverage) < 0.5 ? `: ${worst <= delta ? "consistent with the certificate, but " : ""}at every level tested most questions are withheld.` : "."}
        </p>
      </div>
      <p className="mt-2 text-[0.8125rem] leading-[1.35rem] text-muted">Abstaining is not an error: a withheld answer is the system declining to guess.</p>
      {summary.scope ? (
        <Disclosure className="mt-1">
          <Disclosure.Summary>Scope of the guarantee, as recorded in the results file</Disclosure.Summary>
          <Disclosure.Body className="max-w-[78ch]">{summary.scope}</Disclosure.Body>
        </Disclosure>
      ) : null}
      </div>
      <ByLevel records={allCertified} delta={delta} alpha={alpha} />
    </section>
  );
}

/** The certified method level by level: the rows the finding above summarises. */
function ByLevel({ records, delta, alpha }: { records: LevelRecord[]; delta: number; alpha: number | null }) {
  return (
    <div className="min-w-0 self-start overflow-x-auto rounded-control bg-raised" tabIndex={0} role="group" aria-labelledby="rq3-bylevel">
      <table className="num w-full text-[0.8125rem]">
        <caption id="rq3-bylevel" className="px-3 pt-2.5 text-left text-xs font-medium text-muted">
          Certified threshold, level by level
        </caption>
        <thead>
          <tr className="text-xs text-muted">
            <th scope="col" className="px-3 py-1.5 text-left font-normal">α</th>
            <th scope="col" className="px-2 py-1.5 text-right font-normal">Certified in</th>
            <th scope="col" className="px-2 py-1.5 text-right font-normal">Violation</th>
            <th scope="col" className="px-3 py-1.5 text-right font-normal">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => {
            const trivial = neverCertified(r);
            const over = !trivial && r.violation !== null && r.violation.value > delta;
            return (
              <tr key={r.alpha} aria-current={r.alpha === alpha ? "true" : undefined} className={cn("border-t border-line", r.alpha === alpha && "bg-accent-soft font-medium [&>*]:text-accent-ink")}>
                <th scope="row" className="px-3 py-1.5 text-left font-[inherit] text-ink">
                  {fmtAlpha(r.alpha)}
                </th>
                <td className="px-2 py-1.5 text-right text-ink">{r.certifiedRate !== null ? fmtPct(r.certifiedRate, 1) : "–"}</td>
                <td className={cn("px-2 py-1.5 text-right", over ? "text-contradicted-ink" : "text-ink")}>
                  {trivial ? (
                    <span className="text-muted">all withheld</span>
                  ) : r.violation ? (
                    <span className="inline-flex items-center gap-1">
                      {over ? <CrossGlyph size={12} /> : null}
                      {fmtPct(r.violation.value, 1)}
                      {over ? <span className="sr-only">, above δ</span> : null}
                    </span>
                  ) : (
                    "–"
                  )}
                </td>
                <td className="px-3 py-1.5 text-right text-ink">{r.coverage ? fmtPct(r.coverage.value, 1) : "–"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

const fetchIndex = (signal: AbortSignal) => getExperimentsIndex(signal);
const fetchRq3Runs = (signal: AbortSignal, ...runs: string[]) => Promise.all(runs.map((run) => getExperiment("rq3_conformal", signal, run)));

/**
 * The certified rule across generators, side by side: one panel for violations, one for coverage,
 * a series per run that has RQ3. Runs without RQ3 are named. Descriptive only: no test pairs runs.
 */
function CompareGenerators({ run, delta, alphas }: { run: string; delta: number | null; alphas: readonly number[] }) {
  const index = useLoaded(fetchIndex, [] as const);
  const runs = index.status === "ready" ? (index.data.runs ?? []) : [];
  const withRq3 = runs.filter((r) => r.available.includes("rq3_conformal"));
  const without = runs.filter((r) => !r.available.includes("rq3_conformal"));
  const loaded = useLoaded(fetchRq3Runs, withRq3.map((r) => r.run));
  const compared = useMemo(() => {
    if (index.status !== "ready" || loaded.status !== "ready") return [];
    const all = index.data.runs ?? [];
    const named = all.filter((r) => r.available.includes("rq3_conformal"));
    return compareRuns(named.flatMap((r, i) => (loaded.data[i] === undefined ? [] : [{ run: r.run, label: runLabel(r, all), data: loaded.data[i] }])));
  }, [index, loaded]);
  if (index.status !== "ready" || withRq3.length < 2) return null;
  if (loaded.status !== "ready") {
    return loaded.status === "loading" ? <ChartSkeleton height={COMPARE_HEIGHT + 120} label="Loading the other runs" /> : null;
  }
  const takeaway = compareTakeaway(compared, delta);
  const levels = alphaLevels(compared.map((r) => ({ method: r.run, group: "selective" as const, points: r.points })));
  const shownAlphas = levels.length > 0 ? levels : alphas;

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Does the certificate behave the same for another generator?</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Legend items={compared.map((r, i) => runLegendItem(`${r.label}${r.run === run ? " (picked run)" : ""}`, i))} />
      </ChartCard.Header>
      <ChartCard.Body>
        <div className="grid gap-x-6 gap-y-4 md:grid-cols-2">
          {(["violation", "coverage"] as const).map((metric) => (
            <div key={metric} className="min-w-0">
              <p className="mb-1 px-1 text-xs font-medium text-muted">{metric === "violation" ? "Splits with test risk above α (certified rule)" : "Held-out questions answered (certified rule)"}</p>
              <CompareRunsPanel
                runs={compared}
                metric={metric}
                delta={delta}
                alphas={shownAlphas}
                plotHeight={170}
                ariaLabel={`Certified rule, ${metric === "violation" ? "violation rate" : "coverage"} by α for ${compared.map((r) => r.label).join(" and ")}.`}
              />
            </div>
          ))}
        </div>
        {without.length > 0 ? (
          <p className="mt-3 px-1 text-[0.8125rem] leading-[1.35rem] text-muted">
            {without.map((r) => runLabel(r, runs)).join(", ")} {without.length === 1 ? "has" : "have"} no RQ3 results, so {without.length === 1 ? "it is" : "they are"} not shown.
          </p>
        ) : null}
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "run", label: "Generator" },
          { key: "alpha", label: "α", numeric: true },
          { key: "violation", label: "Violation rate", numeric: true },
          { key: "ci", label: "95% Clopper-Pearson CI", numeric: true },
          { key: "coverage", label: "Coverage", numeric: true },
          { key: "range", label: "Split range", numeric: true },
          { key: "certified", label: "Threshold found in", numeric: true },
        ]}
        rows={compared.flatMap((r) =>
          r.points.map((p) => ({
            run: r.label,
            alpha: fmtAlpha(p.alpha),
            violation: p.trivial ? "all withheld" : pct1(p.violation),
            ci: range1(p.violation),
            coverage: pct1(p.coverage),
            range: range1(p.coverage),
            certified: p.certifiedRate === null ? "–" : fmtPct(p.certifiedRate, 1),
          })),
        )}
      >
        <Foot note="Descriptive: each run is split on its own questions; no test pairs them. Hollow marks: nothing certified in any split." source={withRq3.map((r) => `results/${runPathPrefix(r.run)}rq3_conformal.json`).join(" · ")} />
      </ChartCard.Data>
    </ChartCard>
  );
}

export function GuaranteeSection({ data, run, summary, alpha }: { data: unknown; run: string; summary: Rq3Summary; alpha: number | null }) {
  const file = `${runPathPrefix(run)}rq3_conformal`;
  const source = `results/${file}.json`;
  const { records, delta, deltaSource } = summary;

  const d = useMemo(() => {
    const selective = guaranteeSeries(data, ["selective"]);
    const sets = predictionSets(data);
    return {
      selective,
      domain: selectiveDomain(selective),
      alphas: alphaLevels(selective),
      sets,
      setAlphas: [...new Set(sets.flatMap((s) => s.points.map((p) => p.alpha)))].sort((a, b) => a - b),
      intervals: { violation: rq3Interval(data, "violation_ci"), coverage: rq3Interval(data, "coverage"), sets: rq3Interval(data, "set metrics") },
    };
  }, [data]);
  const ordered = useMemo(() => d.domain.flatMap((m) => d.selective.filter((s) => s.method === m)), [d]);

  const tables = findMetricTables(data)
    .filter((table) => !table.path.some((key) => /^(?:alpha|a|α)?[_=\s]*0?\.\d+$/i.test(key)))
    .map(levelsAsRows);
  const otherTables = groupTables(tables).filter((g) => !(g.key === "prediction_sets" && d.sets.length > 0));
  const violationText = violationTakeaway(ordered, delta);
  const coverageText = coverageTakeaway(ordered);
  const setsText = predictionSetTakeaway(d.sets);
  const methodLegend = ordered.map((s) => methodLegendItem(s.method, methodLabel(s.method)));
  const anyTrivial = ordered.some((s) => s.points.some((p) => p.trivial));

  const levelRows = (show: "violation" | "coverage") =>
    ordered.flatMap((s) =>
      s.points.map((p) => ({
        method: methodLabel(s.method),
        alpha: fmtAlpha(p.alpha),
        value: p.trivial && show === "violation" ? "all withheld" : pct1(p[show]),
        interval: range1(p[show]),
        certified: p.certifiedRate === null ? "–" : fmtPct(p.certifiedRate, 1),
        extra: show === "violation" ? fmtInt(p.nRepeats) : pct1(p.risk),
        conditional: (() => {
          const c = certifiedSplitViolations(p);
          return c === null ? "–" : c.rate === null ? "none certified" : `${fmtInt(c.violations)} of ${fmtInt(c.certified)} (${fmtPct(c.rate, 0)})`;
        })(),
      })),
    );

  return (
    <div className="space-y-4">
      {records.length === 0 && tables.length === 0 ? <Unrecognised file={file} /> : null}
      <Finding summary={summary} alpha={alpha} />

      {ordered.length > 0 ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <ChartCard>
            <ChartCard.Header>
              <ChartCard.Heading>
                <ChartCard.Title as="h3">How often did test risk exceed α?</ChartCard.Title>
                {violationText ? <ChartCard.Takeaway>{violationText}</ChartCard.Takeaway> : null}
              </ChartCard.Heading>
              <ChartCard.Legend items={methodLegend} />
            </ChartCard.Header>
            <ChartCard.Body>
              <ViolationChart
                series={ordered}
                alphas={d.alphas}
                delta={delta}
                highlight={alpha}
                ariaLabel={`Violation rate by target risk α for ${ordered.map((s) => methodLabel(s.method)).join(", ")}. ${violationText ?? ""}`}
              />
            </ChartCard.Body>
            <ChartCard.Data
              columns={[
                { key: "method", label: "Method" },
                { key: "alpha", label: "α", numeric: true },
                { key: "value", label: "Violation rate", numeric: true },
                { key: "interval", label: d.intervals.violation.label, numeric: true },
                { key: "certified", label: "Threshold found in", numeric: true },
                { key: "extra", label: "Splits", numeric: true },
                { key: "conditional", label: "Violations among certifying splits", numeric: true },
              ]}
              rows={levelRows("violation")}
            >
              <Foot
                note={
                  <>
                    Rate over all splits, as the guarantee is stated; a split that certifies nothing counts as no violation. Whiskers: {d.intervals.violation.label}.{" "}
                    {delta !== null ? `Dashed line: δ = ${fmtAlpha(delta)}, from ${deltaSource}.` : "δ is not recorded, so no line is drawn."}
                    {anyTrivial ? " Open rings: nothing certified in any split." : ""}
                  </>
                }
                source={source}
              />
            </ChartCard.Data>
          </ChartCard>

          <ChartCard>
            <ChartCard.Header>
              <ChartCard.Heading>
                <ChartCard.Title as="h3">How many questions still get an answer?</ChartCard.Title>
                {coverageText ? <ChartCard.Takeaway>{coverageText}</ChartCard.Takeaway> : null}
              </ChartCard.Heading>
              <ChartCard.Legend items={methodLegend} />
            </ChartCard.Header>
            <ChartCard.Body>
              <CoverageChart
                series={ordered}
                alphas={d.alphas}
                highlight={alpha}
                plotHeight={220}
                ariaLabel={`Mean coverage by target risk α for ${ordered.map((s) => methodLabel(s.method)).join(", ")}. ${coverageText ?? ""}`}
              />
            </ChartCard.Body>
            <ChartCard.Data
              columns={[
                { key: "method", label: "Method" },
                { key: "alpha", label: "α", numeric: true },
                { key: "value", label: "Mean coverage", numeric: true },
                { key: "interval", label: d.intervals.coverage.label, numeric: true },
                { key: "certified", label: "Threshold found in", numeric: true },
                { key: "extra", label: "Risk when answering", numeric: true },
              ]}
              rows={levelRows("coverage")}
            >
              <Foot note={`Whiskers: ${d.intervals.coverage.label}. Rows under the axis: share of splits in which each rule found a threshold.`} source={source} />
            </ChartCard.Data>
          </ChartCard>
        </div>
      ) : null}

      <CompareGenerators run={run} delta={delta} alphas={d.alphas} />

      {d.sets.length > 0 ? (
        <ChartCard>
          <ChartCard.Header>
            <ChartCard.Heading>
              <ChartCard.Title as="h3">Prediction sets: do they cover the gold answer as often as promised?</ChartCard.Title>
              {setsText ? <ChartCard.Takeaway>{setsText}</ChartCard.Takeaway> : null}
            </ChartCard.Heading>
            <ChartCard.Legend items={d.sets.map((s) => ({ name: methodLabel(s.method), color: setLegendColor(s.method), mark: /lac/i.test(s.method) ? ("dot" as const) : ("diamond" as const), joined: true }))} />
          </ChartCard.Header>
          <ChartCard.Body>
            <div className="grid gap-x-6 gap-y-4 md:grid-cols-2">
              {(["setCoverage", "meanSetSize"] as const).map((metric) => (
                <div key={metric} className="min-w-0">
                  <p className="mb-1 px-1 text-xs font-medium text-muted">{metric === "setCoverage" ? "Coverage against the 1 − α target (dashed line)" : "Mean set size (of yes, no, maybe)"}</p>
                  <PredictionSetPanel
                    sets={d.sets}
                    metric={metric}
                    alphas={d.setAlphas}
                    plotHeight={190}
                    ariaLabel={`${metric === "setCoverage" ? "Set coverage" : "Mean set size"} by α for ${d.sets.map((s) => methodLabel(s.method)).join(" and ")}.`}
                  />
                </div>
              ))}
            </div>
          </ChartCard.Body>
          <ChartCard.Data
            columns={[
              { key: "method", label: "Method" },
              { key: "alpha", label: "α", numeric: true },
              { key: "target", label: "Target", numeric: true },
              { key: "coverage", label: "Set coverage", numeric: true },
              { key: "covRange", label: d.intervals.sets.label, numeric: true },
              { key: "size", label: "Mean set size", numeric: true },
              { key: "empty", label: "Empty sets", numeric: true },
            ]}
            rows={d.sets.flatMap((s) =>
              s.points.map((p) => ({
                method: methodLabel(s.method),
                alpha: fmtAlpha(p.alpha),
                target: fmtPct(1 - p.alpha),
                coverage: pct1(p.setCoverage),
                covRange: range1(p.setCoverage),
                size: p.meanSetSize ? p.meanSetSize.value.toFixed(2) : "–",
                empty: pct1(p.emptySetRate),
              })),
            )}
          >
            <Foot note={`Whiskers: ${d.intervals.sets.label}. Sets over the generator's stated yes/no/maybe probabilities.`} source={source} />
          </ChartCard.Data>
        </ChartCard>
      ) : null}

      {otherTables.map((group) => (
        <TableGroupCard
          key={group.key}
          tables={group.tables}
          title={group.key === "prediction_sets" ? "Prediction sets by risk level" : undefined}
          fallback="Prediction sets"
          rowLabel={group.tables.every((t) => t.rows.every((row) => row.name.startsWith("α = "))) ? "Risk level" : "Method"}
          variantLegend="Set method"
          source={source}
        />
      ))}
      <RawFile data={data} name={file} />
    </div>
  );
}
