"use client";

import { useMemo } from "react";

import { useChartTheme, type ChartTheme } from "@/components/charts/theme";
import { methodLabel, type LevelPoint, type MethodSeries, type RunSeries, type SetPoint } from "@/lib/charts/rq3";
import type { Estimate } from "@/lib/charts/common";
import { fmtAlpha, fmtPct } from "@/lib/format";

import type { MarkShape } from "./rq2-forest";
import { AlphaPlot, type AlphaSeries } from "./rq3-alpha-plot";
import { TipRow } from "./rq2-tooltip";
import { VIOLATION_CERTIFIED_PLOT, VIOLATION_PLUGIN_PLOT } from "./rq3-heights";

const pct = (e: Estimate | null | undefined, digits = 1) => (e ? fmtPct(e.value, digits) : "–");
const range = (e: Estimate | null | undefined, digits = 1) => (e && e.lo !== null && e.hi !== null ? `${fmtPct(e.lo, digits)} to ${fmtPct(e.hi, digits)}` : null);

/**
 * The colour and mark of each selective method. On this tab orange means "plug-in, uncorrected"
 * and nothing else; the certified rules wear the accent (Hoeffding-Bentkus dashed, with square
 * marks, since an open mark already means "nothing certified"); any other method is grey.
 */
export function methodStyle(kit: ChartTheme) {
  return (method: string): { color: string; shape: MarkShape; dashed: boolean } =>
    method === "plug_in"
      ? { color: kit.tokens["series-2"], shape: "diamond", dashed: false }
      : method === "certified"
        ? { color: kit.tokens["series-1"], shape: "dot", dashed: false }
        : method === "certified_hb"
          ? { color: kit.tokens["series-1"], shape: "square", dashed: true }
          : { color: kit.tokens["series-rest"], shape: "dot", dashed: false };
}

/** Generators in the cross-run figure: told apart by mark and line, in ink and grey, never by a series hue. */
export const RUN_SHAPES: readonly MarkShape[] = ["dot", "diamond-open", "square"];
export function runStyle(kit: ChartTheme, i: number): { color: string; shape: MarkShape; dashed: boolean } {
  return { color: i === 0 ? kit.tokens.text : kit.tokens["series-rest"], shape: RUN_SHAPES[i] ?? "dot", dashed: i > 0 };
}

/** Prediction-set methods: LAC in the accent, APS (and any other) in grey. */
export function setStyle(kit: ChartTheme, method: string): { color: string; shape: MarkShape } {
  const lac = /lac/i.test(method);
  return { color: lac ? kit.tokens["series-1"] : kit.tokens["series-rest"], shape: lac ? "dot" : "diamond" };
}

function points(series: MethodSeries, metric: "violation" | "coverage") {
  return series.points.flatMap((p) => {
    const e = p[metric];
    return e ? [{ alpha: p.alpha, value: e.value, lo: e.lo, hi: e.hi, hollow: p.trivial }] : [];
  });
}

function LevelTip({ kit, alpha, series, show }: { kit: ChartTheme; alpha: number; series: readonly MethodSeries[]; show: "violation" | "coverage" }) {
  const style = methodStyle(kit);
  return (
    <>
      <TipRow kit={kit}>
        <strong>α = {fmtAlpha(alpha)}</strong>
      </TipRow>
      {series.map((s) => {
        const p = s.points.find((q) => q.alpha === alpha);
        if (!p) return null;
        const e = p[show];
        return (
          <p key={s.method} className="mt-1">
            <span aria-hidden="true" className="mr-1.5 inline-block size-2 rounded-full" style={{ background: style(s.method).color }} />
            {methodLabel(s.method)}
            <br />
            <span style={{ color: kit.tokens.muted }}>
              {p.trivial ? "nothing certified in any split, all withheld" : `${show === "violation" ? "risk above α in" : "answers"} ${pct(e)}${range(e) ? ` (${range(e)})` : ""}`}
              {!p.trivial && p.certifiedRate !== null ? `; threshold in ${fmtPct(p.certifiedRate, 1)} of splits` : ""}
            </span>
          </p>
        );
      })}
    </>
  );
}

/**
 * Share of splits whose test risk exceeded α, with its Clopper-Pearson CI and δ as the line to
 * stay under, as two small multiples on one α axis: the certified rules on a scale that fits
 * them and δ (they are the claim), and the plug-in rule on 0 to 100%. Never a log or broken axis.
 */
export function ViolationChart({ series, alphas, delta, highlight, ariaLabel }: { series: readonly MethodSeries[]; domain?: readonly string[]; alphas: readonly number[]; delta: number | null; highlight: number | null; ariaLabel: string }) {
  const kit = useChartTheme();
  const certified = useMemo(() => series.filter((s) => s.method !== "plug_in"), [series]);
  const plugIn = useMemo(() => series.filter((s) => s.method === "plug_in"), [series]);
  const [certifiedDrawn, plugInDrawn] = useMemo((): [AlphaSeries[], AlphaSeries[]] => {
    if (!kit) return [[], []];
    const style = methodStyle(kit);
    const draw = (list: readonly MethodSeries[]) => list.map((s) => ({ key: s.method, ...style(s.method), points: points(s, "violation") }));
    return [draw(certified), draw(plugIn)];
  }, [kit, certified, plugIn]);
  const top = Math.max(delta ?? 0, ...certified.flatMap((s) => s.points.map((p) => p.violation?.hi ?? p.violation?.value ?? 0)));
  const refs = delta !== null ? [{ value: delta, label: `δ = ${fmtAlpha(delta)}` }] : [];
  const tip = (list: readonly MethodSeries[]) =>
    function ViolationTip(alpha: number) {
      return kit ? <LevelTip kit={kit} alpha={alpha} series={list} show="violation" /> : null;
    };
  return (
    <div className="space-y-3">
      {certified.length > 0 ? (
        <div>
          <p className="mb-1 px-1 text-xs font-medium text-muted">{certified.length > 1 ? "Certified rules" : "Certified rule"}, on a scale that fits δ</p>
          <AlphaPlot
            alphas={alphas}
            series={certifiedDrawn}
            yDomain={[0, Math.min(1, top * 1.3)]}
            yFormat={(v) => fmtPct(v)}
            yLabel="Splits, risk above α"
            plotHeight={VIOLATION_CERTIFIED_PLOT}
            references={refs}
            highlight={highlight}
            tip={tip(certified)}
            ariaLabel={`${ariaLabel} Certified rules.`}
          />
        </div>
      ) : null}
      {plugIn.length > 0 ? (
        <div>
          <p className="mb-1 px-1 text-xs font-medium text-muted">Plug-in rule, uncorrected, on 0 to 100%</p>
          <AlphaPlot
            alphas={alphas}
            series={plugInDrawn}
            yDomain={[0, 1]}
            yFormat={(v) => fmtPct(v)}
            yLabel="Splits, risk above α"
            plotHeight={VIOLATION_PLUGIN_PLOT}
            references={refs}
            highlight={highlight}
            tip={tip(plugIn)}
            ariaLabel={`${ariaLabel} Plug-in rule.`}
          />
        </div>
      ) : null}
    </div>
  );
}

/** Mean share of held-out questions answered, per method, with the split range; under the axis, how often each rule found a threshold. */
export function CoverageChart({ series, alphas, highlight, plotHeight, ariaLabel }: { series: readonly MethodSeries[]; domain?: readonly string[]; alphas: readonly number[]; highlight: number | null; plotHeight: number; ariaLabel: string }) {
  const kit = useChartTheme();
  const drawn = useMemo<AlphaSeries[]>(() => {
    if (!kit) return [];
    const style = methodStyle(kit);
    return series.map((s) => ({ key: s.method, ...style(s.method), points: points(s, "coverage") }));
  }, [kit, series]);
  const rows = useMemo(
    () =>
      series
        .filter((s) => s.method === "certified" || s.method === "plug_in")
        .map((s) => ({
          label: s.method === "plug_in" ? "plug-in" : s.method,
          values: (alpha: number) => {
            const p: LevelPoint | undefined = s.points.find((q) => q.alpha === alpha);
            return p && p.certifiedRate !== null ? fmtPct(p.certifiedRate, 1) : "–";
          },
        })),
    [series],
  );
  return (
    <AlphaPlot
      alphas={alphas}
      series={drawn}
      yDomain={[0, 1]}
      yFormat={(v) => fmtPct(v)}
      yLabel="Held-out questions answered"
      plotHeight={plotHeight}
      highlight={highlight}
      rows={rows}
      tip={(alpha) => (kit ? <LevelTip kit={kit} alpha={alpha} series={series} show="coverage" /> : null)}
      ariaLabel={ariaLabel}
    />
  );
}

/**
 * One panel of the generator comparison: the certified rule's violation rate or coverage, one
 * series per run. Descriptive: the runs are split on their own questions.
 */
export function CompareRunsPanel({ runs, metric, delta, alphas, plotHeight, ariaLabel }: { runs: readonly RunSeries[]; domain?: readonly string[]; metric: "violation" | "coverage"; delta: number | null; alphas: readonly number[]; plotHeight: number; ariaLabel: string }) {
  const kit = useChartTheme();
  const drawn = useMemo<AlphaSeries[]>(() => {
    if (!kit) return [];
    return runs.map((r, i) => ({
      key: r.run,
      ...runStyle(kit, i),
      points: r.points.flatMap((p) => {
        const e = p[metric];
        return e ? [{ alpha: p.alpha, value: e.value, lo: e.lo, hi: e.hi, hollow: p.trivial }] : [];
      }),
    }));
  }, [kit, runs, metric]);
  const top = metric === "coverage" ? 1 : Math.min(1, Math.max(delta ?? 0, ...runs.flatMap((r) => r.points.map((p) => p.violation?.hi ?? 0))) * 1.08);
  return (
    <AlphaPlot
      alphas={alphas}
      series={drawn}
      yDomain={[0, top]}
      yFormat={(v) => fmtPct(v)}
      yLabel={metric === "coverage" ? "Questions answered" : "Splits with risk above α"}
      plotHeight={plotHeight}
      references={metric === "violation" && delta !== null ? [{ value: delta, label: `δ = ${fmtAlpha(delta)}` }] : []}
      tip={(alpha) =>
        kit ? (
          <>
            <TipRow kit={kit}>
              <strong>Certified rule, α = {fmtAlpha(alpha)}</strong>
            </TipRow>
            {runs.map((r) => {
              const p = r.points.find((q) => q.alpha === alpha);
              const e = p?.[metric];
              return (
                <TipRow key={r.run} kit={kit} label={`${r.label}:`}>
                  {p?.trivial ? "nothing certified" : `${pct(e)}${range(e) ? ` (${range(e)})` : ""}`}
                </TipRow>
              );
            })}
          </>
        ) : null
      }
      ariaLabel={ariaLabel}
    />
  );
}

/** One panel of the prediction-set figure: set coverage against its 1 − α target, or mean set size. */
export function PredictionSetPanel({ sets, metric, alphas, plotHeight, ariaLabel }: { sets: readonly { method: string; points: SetPoint[] }[]; metric: "setCoverage" | "meanSetSize"; alphas: readonly number[]; plotHeight: number; ariaLabel: string }) {
  const kit = useChartTheme();
  const drawn = useMemo<AlphaSeries[]>(() => {
    if (!kit) return [];
    return sets.map((s) => ({
      key: s.method,
      ...setStyle(kit, s.method),
      points: s.points.flatMap((p) => {
        const e = p[metric];
        return e ? [{ alpha: p.alpha, value: e.value, lo: e.lo, hi: e.hi }] : [];
      }),
    }));
  }, [kit, sets, metric]);
  const coverage = metric === "setCoverage";
  const values = sets.flatMap((s) =>
    s.points.flatMap((p) => {
      const e = p[metric];
      return e ? [e.lo ?? e.value, e.hi ?? e.value] : [];
    }),
  );
  // Coverage: a dot plot around the targets, so the axis spans the targets and the data, not 0 to 1.
  const yDomain: [number, number] = coverage
    ? [Math.max(0, Math.min(...values, ...alphas.map((a) => 1 - a)) - 0.02), 1]
    : [0, Math.max(...values, 1)];
  const fmt = coverage ? (v: number) => fmtPct(v) : (v: number) => v.toFixed(1);
  return (
    <AlphaPlot
      alphas={alphas}
      series={drawn}
      yDomain={yDomain}
      yFormat={fmt}
      yLabel={coverage ? "Sets containing the gold label" : "Labels per set (mean)"}
      plotHeight={plotHeight}
      references={coverage ? [{ value: (a: number) => 1 - a, label: "1 − α" }] : []}
      tip={(alpha) =>
        kit ? (
          <>
            <TipRow kit={kit}>
              <strong>α = {fmtAlpha(alpha)}</strong>
              {coverage ? <span style={{ color: kit.tokens.muted }}> target {fmtPct(1 - alpha)}</span> : null}
            </TipRow>
            {sets.map((s) => {
              const e = s.points.find((p) => p.alpha === alpha)?.[metric];
              const text = coverage ? `${pct(e)}${range(e) ? ` (${range(e)})` : ""}` : e ? `${e.value.toFixed(2)}${e.lo !== null && e.hi !== null ? ` (${e.lo.toFixed(2)} to ${e.hi.toFixed(2)})` : ""}` : "–";
              return (
                <TipRow key={s.method} kit={kit} label={`${methodLabel(s.method)}:`}>
                  {text}
                </TipRow>
              );
            })}
          </>
        ) : null
      }
      ariaLabel={ariaLabel}
    />
  );
}
