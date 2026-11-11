"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { useChartTheme } from "@/components/charts/theme";
import { Picker } from "@/components/ui/picker";
import {
  devBeforeAfter,
  estimateExtent,
  isBinaryMetric,
  lossSeries,
  lossSummary,
  lossTakeaway,
  paddedDomain,
  pairedBaselines,
  pairedDiffs,
  pairedTakeaway,
  perQueryDistribution,
  perQueryTakeaway,
  recallAtK,
  recallTakeaway,
  retrievalMetrics,
  retrievalSystems,
  rq1Interval,
} from "@/lib/charts/rq1";
import { at, isRecord } from "@/lib/charts/common";
import { fmtInt, fmtMetric, fmtP, fmtPct } from "@/lib/format";
import { metricLabel } from "@/lib/results";

import { dotRowsHeight, LOSS_CHART_HEIGHT, stripsHeight } from "./rq1-heights";
import type { DotRow, DotSeries } from "./rq1-dot-rows";

/* Each figure is its own lazily loaded module: the page paints its cards, and the plots follow. */
const DotRows = dynamic(() => import("./rq1-dot-rows").then((m) => m.DotRows), { ssr: false, loading: () => <ChartSkeleton height={dotRowsHeight(8)} /> });
const PerQueryStrips = dynamic(() => import("./rq1-perquery-strips").then((m) => m.PerQueryStrips), {
  ssr: false,
  loading: () => <ChartSkeleton height={stripsHeight(8)} />,
});
const LossChart = dynamic(() => import("./rq1-loss-chart").then((m) => m.LossChart), { ssr: false, loading: () => <ChartSkeleton height={LOSS_CHART_HEIGHT} /> });

const DATASET_NAME: Record<string, string> = { nfcorpus: "NFCorpus", pubmedqa: "PubMedQA" };
const datasetName = (key: string) => DATASET_NAME[key] ?? key;
const SOURCE = "results/rq1_retrieval.json";

const K_SHAPES = ["dot", "diamond", "square"] as const;

const pct1 = (v: number) => fmtPct(v, 1);
const pct0 = (v: number) => fmtPct(v, 0);
const signed = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
const ci = (lo: number | null, hi: number | null, f: (v: number) => string) => (lo !== null && hi !== null ? `${f(lo)} to ${f(hi)}` : "–");

/* ---------- Recall@k on PubMedQA ---------- */

/** How often the gold abstract is in the top k, per retriever, for every k the file reports. */
export function RecallAtKCard({ data, deployed }: { data: unknown; deployed: string | null }) {
  const kit = useChartTheme();
  const curves = useMemo(() => recallAtK(data, "pubmedqa"), [data]);
  const interval = rq1Interval(data);
  const ks = useMemo(() => [...new Set(curves.flatMap((c) => c.points.map((p) => p.k)))].sort((a, b) => a - b), [curves]);
  const takeaway = useMemo(() => recallTakeaway(curves), [curves]);
  const domain = useMemo(() => paddedDomain(estimateExtent(curves.flatMap((c) => c.points.map((p) => p.estimate))), 0.1, [0, 1]), [curves]);

  if (curves.length === 0) return null;
  // k is ordinal, so one hue from light to dark (in dark mode, dark to light), and a shape per k.
  const ramp = kit ? kit.colors.sequential([0, 1]) : null;
  const step = (i: number) => (ks.length > 1 ? 0.5 + (0.5 * i) / (ks.length - 1) : 1);
  const series: DotSeries[] = ks.map((k, i) => ({ key: `k${k}`, name: `k = ${k}`, color: ramp ? ramp(step(i)) : "currentColor", shape: K_SHAPES[i] ?? "dot" }));
  const rows: DotRow[] = curves.map((c) => ({
    key: c.system,
    label: c.system,
    emphasis: c.system === deployed,
    marks: c.points.map((p) => ({ series: `k${p.k}`, value: p.estimate.value, lo: p.estimate.lo, hi: p.estimate.hi })),
  }));

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Does the gold abstract make the top k? PubMedQA</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Legend items={series.map((s) => ({ name: `Recall@${s.name.replace("k = ", "")}`, color: s.color, mark: s.shape ?? "dot" }))} />
      </ChartCard.Header>
      <ChartCard.Body height={dotRowsHeight(rows.length)}>
        <DotRows
          rows={rows}
          series={series}
          domain={domain}
          format={pct1}
          tickFormat={pct0}
          connect
          intervalLabel="CI"
          ariaLabel={`Recall at ${ks.join(", ")} per retriever on PubMedQA, axis from ${pct0(domain[0])} to ${pct0(domain[1])}. ${takeaway ?? ""}`}
        />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[{ key: "system", label: "Retriever" }, ...ks.flatMap((k) => [{ key: `r${k}`, label: `Recall@${k}`, numeric: true }, { key: `c${k}`, label: "CI", numeric: true }])]}
        rows={curves.map((c) => ({
          system: c.system,
          ...Object.fromEntries(
            c.points.flatMap((p) => [
              [`r${p.k}`, pct1(p.estimate.value)],
              [`c${p.k}`, ci(p.estimate.lo, p.estimate.hi, pct1)],
            ]),
          ),
        }))}
      >
        <ChartCard.Note>
          {deployed ? `Bold: ${deployed}, the deployed retriever. ` : ""}Axis from {pct0(domain[0])}: dots are positions, not bars. Whiskers: {interval.text ?? "interval as written in the file"}; one question set, so
          the k = 20 values sit close to the ceiling.
        </ChartCard.Note>
        <ChartCard.Source>{SOURCE}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}

/* ---------- Paired differences against a baseline ---------- */

function pairedDatasets(data: unknown): string[] {
  const block = at(data, "paired_tests");
  return isRecord(block) ? Object.keys(block) : [];
}

/** Each retriever minus a baseline, per query, averaged: the paired test RQ1 is decided on. */
export function PairedDiffCard({ data, deployed }: { data: unknown; deployed: string | null }) {
  const kit = useChartTheme();
  const datasets = useMemo(() => pairedDatasets(data), [data]);
  const [pickedDataset, setDataset] = useState<string | null>(null);
  const dataset = pickedDataset && datasets.includes(pickedDataset) ? pickedDataset : (datasets[0] ?? "");
  const baselines = useMemo(() => pairedBaselines(data, dataset), [data, dataset]);
  const [pickedBaseline, setBaseline] = useState<string | null>(null);
  const baseline = pickedBaseline && baselines.includes(pickedBaseline) ? pickedBaseline : baselines.includes("bm25") ? "bm25" : (baselines[0] ?? "");
  const all = useMemo(() => pairedDiffs(data, dataset, baseline), [data, dataset, baseline]);
  const metrics = useMemo(() => {
    const present = new Set(all.map((d) => d.metric));
    const ordered = retrievalMetrics(data, dataset).filter((m) => present.has(m));
    return [...ordered, ...[...present].filter((m) => !ordered.includes(m))];
  }, [all, data, dataset]);
  const [pickedMetric, setMetric] = useState<string | null>(null);
  const metric = pickedMetric && metrics.includes(pickedMetric) ? pickedMetric : (metrics[0] ?? "");
  const order = useMemo(() => retrievalSystems(data, dataset), [data, dataset]);
  const diffs = useMemo(
    () => all.filter((d) => d.metric === metric).sort((a, b) => order.indexOf(a.system) - order.indexOf(b.system)),
    [all, metric, order],
  );
  const name = metricLabel(metric);
  const takeaway = useMemo(() => pairedTakeaway(diffs, baseline, name), [diffs, baseline, name]);
  const domain = useMemo(() => {
    const values = [...estimateExtent(diffs.map((d) => d.diff)), 0];
    const span = Math.max(...values) - Math.min(...values);
    return paddedDomain(values, span > 0.2 ? 0.05 : 0.02);
  }, [diffs]);
  const interval = rq1Interval(data);
  const test = at(data, "meta", "inference", "test");
  const adjustment = at(data, "meta", "inference", "adjustment");

  if (datasets.length === 0 || diffs.length === 0) return null;
  const color = kit ? kit.tokens["series-1"] : "currentColor";
  const rows: DotRow[] = diffs.map((d) => ({
    key: d.system,
    label: d.system,
    emphasis: d.system === deployed,
    marks: [{ series: "diff", value: d.diff.value, lo: d.diff.lo, hi: d.diff.hi }],
    aside: d.pHolm !== null ? `Holm p ${fmtP(d.pHolm)}` : `p ${fmtP(d.p)}`,
  }));

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Which retrievers beat {baseline}, query by query?</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Actions className="w-full min-w-0 shrink sm:w-auto">
          {datasets.length > 1 ? (
            <Picker legend="Dataset" value={dataset} options={datasets.map((d) => ({ value: d, label: datasetName(d) }))} onChange={setDataset} segmentedFrom="sm" />
          ) : null}
          {baselines.length > 1 ? (
            <Picker legend="Baseline" value={baseline} options={baselines.map((b) => ({ value: b, label: `vs ${b}` }))} onChange={setBaseline} segmentedFrom="never" />
          ) : null}
          {metrics.length > 1 ? (
            <Picker legend="Metric" value={metric} options={metrics.map((m) => ({ value: m, label: metricLabel(m) }))} onChange={setMetric} segmentedFrom="never" />
          ) : null}
        </ChartCard.Actions>
      </ChartCard.Header>
      <ChartCard.Body height={dotRowsHeight(rows.length)}>
        <DotRows
          rows={rows}
          series={[{ key: "diff", name: `${name} difference`, color }]}
          domain={domain}
          format={signed}
          tickFormat={(v) => (v === 0 ? "0" : signed(v).replace(/0+$/, "").replace(/\.$/, ""))}
          reference={{ value: 0, label: `no difference from ${baseline}` }}
          intervalLabel="CI"
          asideWidth={92}
          ariaLabel={`${name} of each retriever minus ${baseline} on ${datasetName(dataset)}, with intervals and Holm-adjusted p-values. ${takeaway ?? ""}`}
        />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "system", label: "Retriever" },
          { key: "diff", label: `${name} minus ${baseline}`, numeric: true },
          { key: "ci", label: "CI", numeric: true },
          { key: "p", label: "p", numeric: true },
          { key: "holm", label: "Holm p", numeric: true },
        ]}
        rows={diffs.map((d) => ({ system: d.system, diff: signed(d.diff.value), ci: ci(d.diff.lo, d.diff.hi, signed), p: fmtP(d.p), holm: fmtP(d.pHolm) }))}
      >
        <ChartCard.Note>
          {deployed ? `Bold: ${deployed}, the deployed retriever. ` : ""}Dashed line: no difference. Whiskers: unadjusted {interval.text ?? "interval as written in the file"} over
          queries. Right: p from the {typeof test === "string" ? test : "paired test"}
          {typeof adjustment === "string" ? `, Holm-adjusted (${adjustment.replace(/^Holm\s*/i, "")})` : ", Holm-adjusted"}, so a whisker can clear zero while the adjusted p does
          not.
        </ChartCard.Note>
        <ChartCard.Source>{SOURCE}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}

/* ---------- Per-query distribution ---------- */

function perQueryDatasets(data: unknown): string[] {
  const block = at(data, "per_query");
  return isRecord(block) ? Object.keys(block).filter((k) => isRecord(block[k])) : [];
}

/** The per-query scores behind each mean: how many queries a retriever fails outright. */
export function PerQueryCard({ data, deployed }: { data: unknown; deployed: string | null }) {
  const datasets = useMemo(() => perQueryDatasets(data), [data]);
  const [pickedDataset, setDataset] = useState<string | null>(null);
  // NFCorpus first: its graded judgements give a spread; PubMedQA's single gold abstract makes recall 0 or 1.
  const fallback = datasets.includes("nfcorpus") ? "nfcorpus" : (datasets[0] ?? "");
  const dataset = pickedDataset && datasets.includes(pickedDataset) ? pickedDataset : fallback;
  // Only metrics with a spread per query: recall@k with one gold abstract is 0 or 1 for every
  // question, and its strip would only restate the mean.
  const metrics = useMemo(
    () => retrievalMetrics(data, dataset).filter((m) => perQueryDistribution(data, dataset, m).some((r) => !isBinaryMetric(r.values))),
    [data, dataset],
  );
  const [pickedMetric, setMetric] = useState<string | null>(null);
  const metric = pickedMetric && metrics.includes(pickedMetric) ? pickedMetric : metrics.includes("ndcg@10") ? "ndcg@10" : (metrics[0] ?? "");
  const rows = useMemo(() => (metric ? perQueryDistribution(data, dataset, metric) : []), [data, dataset, metric]);
  const name = metricLabel(metric);
  const takeaway = useMemo(() => perQueryTakeaway(rows, name), [rows, name]);

  if (datasets.length === 0 || rows.length === 0) return null;

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">What does the mean hide? {name} for every query</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Actions className="w-full min-w-0 shrink sm:w-auto">
          {datasets.length > 1 ? (
            <Picker legend="Dataset" value={dataset} options={datasets.map((d) => ({ value: d, label: datasetName(d) }))} onChange={setDataset} segmentedFrom="sm" />
          ) : null}
          {metrics.length > 1 ? (
            <Picker legend="Metric" value={metric} options={metrics.map((m) => ({ value: m, label: metricLabel(m) }))} onChange={setMetric} segmentedFrom="never" />
          ) : null}
        </ChartCard.Actions>
      </ChartCard.Header>
      <ChartCard.Body height={stripsHeight(rows.length)}>
        <PerQueryStrips rows={rows} deployed={deployed} ariaLabel={`${name} per query on ${datasetName(dataset)}, one strip per retriever. ${takeaway ?? ""}`} />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "system", label: "Retriever" },
          { key: "n", label: "Queries", numeric: true },
          { key: "q1", label: "Lower quartile", numeric: true },
          { key: "median", label: "Median", numeric: true },
          { key: "q3", label: "Upper quartile", numeric: true },
          { key: "mean", label: "Mean", numeric: true },
          { key: "zero", label: "At zero", numeric: true },
        ]}
        rows={rows.map((r) => ({
          system: r.system,
          n: fmtInt(r.summary.n),
          q1: fmtMetric(r.summary.q1),
          median: fmtMetric(r.summary.median),
          q3: fmtMetric(r.summary.q3),
          mean: fmtMetric(r.summary.mean),
          zero: pct1(r.zeroShare),
        }))}
      >
        <ChartCard.Note>
          {deployed ? `Bold, in colour: ${deployed}, the deployed retriever. ` : ""}Dots: one query each above zero (vertical spread is jitter, not data). Bar at 0: the queries
          scoring exactly zero, its height their share, printed on the right. Band: middle half; tick: median. Only metrics that vary per query are offered.
        </ChartCard.Note>
        <ChartCard.Source>{SOURCE} · per_query</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}

/* ---------- Encoder fine-tuning ---------- */

const TRAINING_SOURCE = "results/encoder_training.json";

/** The fine-tuning loss, unsmoothed, against chance, and the dev-set retrieval before and after. */
export function LossCard({ training }: { training: unknown }) {
  const series = useMemo(() => lossSeries(training), [training]);
  const dev = useMemo(() => devBeforeAfter(training), [training]);
  const takeaway = useMemo(() => (series ? lossTakeaway(series, dev) : null), [series, dev]);
  const summary = useMemo(() => (series ? lossSummary(series) : null), [series]);
  if (!series) return null;
  const objective = at(training, "hyperparameters", "objective");

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Did fine-tuning teach the encoder anything?</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
      </ChartCard.Header>
      <ChartCard.Body height={LOSS_CHART_HEIGHT}>
        <LossChart series={series} ariaLabel={`Training loss over ${fmtInt(series.points.length)} logged steps. ${takeaway ?? ""}`} />
      </ChartCard.Body>
      {dev.rows.length > 0 ? (
        <dl className="mx-4 mb-4 grid grid-cols-1 gap-px overflow-hidden rounded-control bg-(--rule) text-sm sm:mx-5 sm:grid-cols-2">
          {dev.rows.map((r) => (
            <div key={r.metric} className="flex items-baseline justify-between gap-3 bg-raised px-3 py-2">
              <dt className="text-muted">
                Dev {metricLabel(r.metric)}
                {dev.nQueries !== null ? <span className="num text-faint"> · {fmtInt(dev.nQueries)} queries</span> : null}
              </dt>
              <dd className="num text-ink">
                <span className="text-muted">{fmtMetric(r.before)}</span>
                <span aria-hidden="true" className="px-1.5 text-faint">
                  →
                </span>
                <span className="sr-only"> before, </span>
                <span className="font-semibold">{fmtMetric(r.after)}</span>
                <span className="sr-only"> after fine-tuning</span>
              </dd>
            </div>
          ))}
        </dl>
      ) : null}
      <ChartCard.Data
        columns={[
          { key: "step", label: "Step", numeric: true },
          { key: "loss", label: "Loss", numeric: true },
        ]}
        rows={series.points.map((p) => ({ step: fmtInt(p.step), loss: p.loss.toFixed(4) }))}
      >
        <ChartCard.Note>
          {typeof objective === "string" ? `${objective}; ` : ""}every logged point, no smoothing
          {summary ? ` (takeaway averages ${summary.window} points at each end)` : ""}. Dev metrics have no interval in the file.
        </ChartCard.Note>
        <ChartCard.Source>{TRAINING_SOURCE}</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}
