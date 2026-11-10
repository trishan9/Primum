"use client";

import { useState } from "react";

import { ChartFrame } from "@/components/charts/ChartFrame";
import { IntervalChart } from "@/components/charts/IntervalChart";
import { seriesColor } from "@/components/charts/scale";
import { XYChart } from "@/components/charts/XYChart";
import { ApiDown } from "@/components/shell/ApiDown";
import { useApi } from "@/components/shell/ApiProvider";
import { Code, EmptyBlock, Segmented } from "@/components/ui/controls";
import { ChevronGlyph } from "@/components/ui/glyphs";
import { getExperimentsIndex, MAIN_RUN, type ExperimentName } from "@/lib/api";
import { fmtAlpha, fmtInt, fmtMetric, fmtP, fmtPct, fmtProb } from "@/lib/format";
import {
  findLevelRecords,
  findLossCurve,
  findMetricTables,
  findReliability,
  findRiskCoverage,
  isRecord,
  levelsAsRows,
  readProvenance,
  readStat,
  readUnmeasured,
  runLabel,
  runPathPrefix,
  type LevelRecord,
  type Stat,
  type MetricTable,
} from "@/lib/results";

import type { RunInfo } from "@/lib/types";

import { MetricTableBlock } from "./MetricTableBlock";
import { useExperiment, useLoaded, type Loaded } from "./useExperiment";

interface Question {
  id: "rq1" | "rq2" | "rq3";
  label: string;
  /** A two or three word name for the question, used as its heading. */
  topic: string;
  file: ExperimentName;
  question: string;
  command: string;
  /** What the section will hold once the experiment has run. Describes the analysis, never its outcome. */
  willShow: string;
  /** True when the results do not involve the generator, so they are read from the main run whichever is picked. */
  generatorIndependent?: boolean;
}

const QUESTIONS: Question[] = [
  {
    id: "rq1",
    label: "RQ1",
    topic: "Retrieval",
    file: "rq1_retrieval",
    question:
      "Does a domain-adapted, contrastively fine-tuned encoder, fused with lexical BM25, retrieve biomedical evidence better than either alone?",
    command: "primum exp rq1",
    generatorIndependent: true,
    willShow:
      "nDCG@10, Recall@100 and MRR@10 on the NFCorpus test queries, and Recall@1, 5 and 20 with MRR@10 for finding the gold abstract of each PubMedQA question, for every retriever, with 95% bootstrap intervals and paired permutation tests against BM25 and the base dense encoder.",
  },
  {
    id: "rq2",
    label: "RQ2",
    topic: "Confidence and calibration",
    file: "rq2_confidence",
    question:
      "Which signals predict that a generated answer is correct, and is a learned combination better calibrated than any single signal?",
    command: "primum exp rq2",
    willShow:
      "AUROC, AURC, ECE and Brier score for each single signal and for the logistic and MLP confidence models under each calibration method, over repeated splits; a feature ablation; a reliability diagram; and risk-coverage curves.",
  },
  {
    id: "rq3",
    label: "RQ3",
    topic: "Certified abstention",
    file: "rq3_conformal",
    question:
      "Does the certified threshold keep selective risk at or below α on unseen questions, and how much coverage does that cost?",
    command: "primum exp rq3",
    willShow:
      "For each certified α: how often test risk exceeded α across repeated splits, with a Clopper-Pearson interval, set against the tolerated rate δ; mean coverage; and the same for an uncorrected plug-in threshold and for single-signal confidences.",
  },
];

const DATASET_TITLES: Record<string, string> = {
  nfcorpus: "NFCorpus test queries",
  pubmedqa: "PubMedQA gold-abstract retrieval",
};

/**
 * A heading from where the table sits in the file: `paired_tests/nfcorpus/vs_bm25` reads
 * "Paired tests against bm25, NFCorpus test queries". A name after `vs_` is kept as written.
 */
function tableTitle(table: MetricTable, fallback: string): string {
  if (table.path.length === 0) return fallback;
  const known = table.path.map((key) => DATASET_TITLES[key.toLowerCase()]).find(Boolean);
  let joined = "";
  for (const key of table.path) {
    if (DATASET_TITLES[key.toLowerCase()]) continue;
    const vs = key.match(/^vs[_-](.+)$/);
    joined += vs ? ` against ${vs[1]}` : `${joined ? ", " : ""}${key.replace(/_/g, " ")}`;
  }
  joined = [joined.trim(), known].filter(Boolean).join(", ");
  return joined.charAt(0).toUpperCase() + joined.slice(1);
}

function RawFile({ data, name }: { data: unknown; name: string }) {
  const text = JSON.stringify(data, null, 2) ?? "";
  const lines = text.split("\n");
  const limit = 400;
  return (
    <details className="group rounded-card border border-line bg-surface">
      <summary className="flex min-h-11 items-center gap-1.5 px-4 text-[0.8125rem] font-medium text-ink sm:px-5">
        <ChevronGlyph size={14} className="-rotate-90 text-muted transition-transform duration-150 group-open:rotate-0" />
        Raw results file, results/{name}.json
      </summary>
      <pre
        tabIndex={0}
        className="max-h-96 relative overflow-auto border-t border-rule px-4 py-3 font-mono text-xs leading-relaxed text-ink sm:px-5"
      >
        {lines.slice(0, limit).join("\n")}
        {lines.length > limit ? `\n… ${fmtInt(lines.length - limit)} more lines in the file` : ""}
      </pre>
    </details>
  );
}

function Provenance({ data, file }: { data: unknown; file: string }) {
  const p = readProvenance(data);
  const parts = [
    `results/${file}.json`,
    p.created ? `written ${p.created}` : null,
    p.n !== null ? `n = ${fmtInt(p.n)}` : null,
    p.models.length ? p.models.join(", ") : null,
  ].filter(Boolean);
  return <p className="num mt-1.5 text-xs text-muted">{parts.join(", ")}</p>;
}

function Unrecognised({ file }: { file: string }) {
  return (
    <EmptyBlock title="The results file is here, but this page does not recognise its layout">
      <p>
        No table, curve or per-level record could be read from <Code wrap>results/{file}.json</Code>, so no figure is
        drawn. The file itself is shown below, unaltered.
      </p>
    </EmptyBlock>
  );
}

/* ---------- RQ1 ---------- */

function Rq1({ data }: { data: unknown }) {
  const tables = findMetricTables(data);
  const training = useExperiment("encoder_training");
  const loss = training.status === "ready" ? findLossCurve(training.data) : null;
  const dev = training.status === "ready" && isRecord(training.data) ? findMetricTables({ dev: training.data.dev }) : [];

  return (
    <div className="space-y-4">
      {tables.length === 0 ? <Unrecognised file="rq1_retrieval" /> : null}
      {tables.map((table) => (
        <MetricTableBlock
          key={table.path.join("/")}
          table={table}
          title={tableTitle(table, "Retrieval")}
          rowLabel="Retriever"
          reference="bm25"
        />
      ))}
      {loss ? (
        <ChartFrame
          title="Did the contrastive loss fall during fine-tuning?"
          note="In-batch-negative InfoNCE loss of the fine-tuned encoder, by optimiser step, from results/encoder_training.json."
          table={{
            columns: [
              { key: "step", label: "Step", numeric: true },
              { key: "loss", label: "Loss", numeric: true },
            ],
            rows: loss.points.map((p) => ({ step: fmtInt(p.x), loss: fmtMetric(p.y) })),
          }}
        >
          <XYChart
            series={[{ name: "Training loss", color: seriesColor(0), points: loss.points }]}
            xLabel="Optimiser step"
            yLabel="InfoNCE loss"
            ariaLabel={`Training loss over ${loss.points.length} recorded steps.`}
            xFormat={(v) => fmtInt(v)}
          />
        </ChartFrame>
      ) : null}
      {dev.map((table) => (
        <MetricTableBlock
          key="encoder-dev"
          table={table}
          title="Encoder before and after fine-tuning, on its own dev queries"
          rowLabel="Encoder"
        />
      ))}
      <RawFile data={data} name="rq1_retrieval" />
    </div>
  );
}

/* ---------- RQ2 ---------- */

interface Comparison {
  reference: string;
  model: string;
  brier: Stat | null;
  brierP: number | null;
  auroc: Stat | null;
  aurocBetterShare: number | null;
}

/** `comparisons: {vs_<reference>: {<model>: {brier_diff, brier_p, auroc_diff, auroc_better_share}}}`. */
function readComparisons(data: unknown): Comparison[] {
  if (!isRecord(data) || !isRecord(data.comparisons)) return [];
  const out: Comparison[] = [];
  for (const [key, block] of Object.entries(data.comparisons)) {
    if (!isRecord(block)) continue;
    const reference = key.replace(/^vs[_-]/, "");
    for (const [model, entry] of Object.entries(block)) {
      if (!isRecord(entry)) continue;
      out.push({
        reference,
        model,
        brier: readStat(entry.brier_diff),
        brierP: typeof entry.brier_p === "number" ? entry.brier_p : null,
        auroc: readStat(entry.auroc_diff),
        aurocBetterShare: typeof entry.auroc_better_share === "number" ? entry.auroc_better_share : null,
      });
    }
  }
  return out;
}

/** What a difference (model minus reference) says, read off its interval; lower is better for Brier. */
function brierVerdict(stat: Stat | null): string {
  if (!stat || stat.lo === null || stat.hi === null) return "no interval reported";
  if (stat.hi < 0) return "better (lower Brier)";
  if (stat.lo > 0) return "worse (higher Brier)";
  return "no clear difference";
}

/**
 * The comparison RQ2 turns on: each learned combination against the model's own stated
 * confidence. The verdict column is read from the interval in the file, so the page cannot
 * claim an improvement the results do not show.
 */
function Comparisons({ data }: { data: unknown }) {
  const rows = readComparisons(data);
  if (rows.length === 0) return null;
  const meta = isRecord(data) && isRecord(data.meta) ? data.meta : {};
  const calibrator = typeof meta.main_calibrator === "string" ? meta.main_calibrator : null;
  const references = [...new Set(rows.map((r) => r.reference))];
  const anyBetter = rows.some((r) => r.brier && r.brier.hi !== null && r.brier.hi < 0);
  const worse = [
    ...new Set(rows.filter((r) => r.brier && r.brier.lo !== null && r.brier.lo > 0).map((r) => r.model)),
  ];

  return (
    <section aria-labelledby="rq2-comparisons" className="min-w-0 rounded-card border border-line bg-surface p-4 sm:p-5">
      <h3 id="rq2-comparisons" className="text-[0.9375rem] font-semibold leading-snug text-ink">
        Does a learned combination beat the model&apos;s stated confidence?
      </h3>
      <p className="mt-1 max-w-[82ch] text-sm leading-relaxed text-muted">
        Each learned model against {references.join(" and ")}
        {calibrator ? `, under ${calibrator} calibration` : ""}. Differences are the model minus the reference: a
        negative Brier difference means the learned model&apos;s probabilities are more accurate, a positive AUROC difference means it
        ranks right and wrong answers apart better.{" "}
        {anyBetter
          ? "At least one learned model has a Brier interval entirely below zero."
          : "No learned model has a Brier interval entirely below zero, so these results show no improvement over the stated confidence."}
        {worse.length > 0
          ? ` ${worse.join(", ")} ${worse.length === 1 ? "has" : "have"} a Brier interval entirely above zero against at least one reference: worse than the stated confidence.`
          : ""}
      </p>
      <div className="relative mt-3 overflow-x-auto" tabIndex={0} role="group" aria-labelledby="rq2-comparisons">
        <table className="num w-full min-w-[44rem] text-sm">
          <thead>
            <tr className="text-[0.8125rem] text-muted">
              <th scope="col" className="pb-2 pr-3 text-left font-normal">Reference</th>
              <th scope="col" className="pb-2 pr-3 text-left font-normal">Learned model</th>
              <th scope="col" className="pb-2 pl-3 text-right font-normal">Brier difference</th>
              <th scope="col" className="pb-2 pl-3 text-right font-normal">p</th>
              <th scope="col" className="pb-2 pl-3 text-left font-normal">Reading</th>
              <th scope="col" className="pb-2 pl-3 text-right font-normal">AUROC difference</th>
              <th scope="col" className="pb-2 pl-3 text-right font-normal">Splits with higher AUROC</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={`${row.reference}/${row.model}`} className="border-t border-rule align-baseline">
                <td className="py-1.5 pr-3 text-muted">{row.reference}</td>
                <th scope="row" className="py-1.5 pr-3 text-left font-medium text-ink">{row.model}</th>
                <td className="whitespace-nowrap py-1.5 pl-3 text-right text-ink">
                  {row.brier ? fmtMetric(row.brier.value) : "–"}
                  {row.brier && row.brier.lo !== null && row.brier.hi !== null ? (
                    <span className="block text-[0.8125rem] leading-tight text-muted">
                      {fmtMetric(row.brier.lo)} to {fmtMetric(row.brier.hi)}
                    </span>
                  ) : null}
                </td>
                <td className="py-1.5 pl-3 text-right text-ink">{fmtP(row.brierP)}</td>
                <td className="py-1.5 pl-3 text-ink">{brierVerdict(row.brier)}</td>
                <td className="whitespace-nowrap py-1.5 pl-3 text-right text-ink">
                  {row.auroc ? fmtMetric(row.auroc.value) : "–"}
                  {row.auroc && row.auroc.lo !== null && row.auroc.hi !== null ? (
                    <span className="block text-[0.8125rem] leading-tight text-muted">
                      {fmtMetric(row.auroc.lo)} to {fmtMetric(row.auroc.hi)}
                    </span>
                  ) : null}
                </td>
                <td className="py-1.5 pl-3 text-right text-ink">{row.aurocBetterShare === null ? "–" : fmtPct(row.aurocBetterShare)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * The end-to-end runs RQ2 and RQ3 are built on (`results/pubmedqa_runs.json`): answer accuracy
 * closed-book, with retrieval, and with the gold abstract. The file also holds every per-question
 * record, which is far too large to print, so only its summary is drawn.
 */
const CONDITION_NAMES: Record<string, string> = {
  closed_book: "closed book",
  rag: "with retrieval (rag)",
  oracle: "with the gold abstract (oracle)",
};

function PubmedqaRuns({ run }: { run: string }) {
  const runs = useExperiment("pubmedqa_runs", run);
  const prefix = runPathPrefix(run);
  if (runs.status !== "ready") {
    return runs.status === "missing" ? (
      <p className="text-sm text-muted">
        The end-to-end runs these estimators are trained on have not been written yet: run <Code>primum exp pubmedqa</Code>.
      </p>
    ) : null;
  }
  const data = runs.data;
  // A run with one condition is still a table (one row), so pass minRows = 1 here.
  const summary = isRecord(data) && isRecord(data.summary) ? findMetricTables(data.summary, 5, 1) : [];
  const paired = isRecord(data) && isRecord(data.paired_tests) ? findMetricTables({ paired_tests: data.paired_tests }) : [];
  const meta = isRecord(data) && isRecord(data.meta) ? data.meta : {};
  const incomplete = meta.complete === false;
  // The conditions this run answered under, read from the file; the ones it did not are named
  // so that an absent row reads as "not run", not as a gap in the table.
  const pubmedqa = isRecord(data) && isRecord(data.summary) && isRecord(data.summary.pubmedqa) ? data.summary.pubmedqa : {};
  const conditions = Object.keys(pubmedqa);
  const notRun = Object.keys(CONDITION_NAMES).filter((c) => !conditions.includes(c));
  const conditionList = conditions.map((c) => CONDITION_NAMES[c] ?? c).join(", ");

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-[0.9375rem] font-semibold leading-snug text-ink">The answers being scored</h3>
        <Provenance data={data} file={`${prefix}pubmedqa_runs`} />
        {incomplete ? (
          <p className="mt-1.5 text-sm font-medium text-ink">
            This run is marked incomplete{typeof meta.stopped === "string" ? ` (${meta.stopped})` : ""}: its numbers cover only the questions answered before it stopped.
          </p>
        ) : null}
        {conditions.length > 0 && notRun.length > 0 ? (
          <p className="mt-1.5 max-w-[72ch] text-sm leading-relaxed text-muted">
            This run answered only {conditionList}. It has no {notRun.map((c) => c.replace("_", "-")).join(" or ")}{" "}
            condition, so there is no accuracy comparison between conditions for this generator.
          </p>
        ) : null}
      </div>
      {summary.map((table) => (
        <MetricTableBlock
          key={`summary/${table.path.join("/")}`}
          table={table}
          title={`PubMedQA answer accuracy${conditionList ? `: ${conditionList}` : ""}`}
          rowLabel="Condition"
        />
      ))}
      {paired.map((table) => (
        <MetricTableBlock key="paired" table={table} title="Paired accuracy differences between conditions" rowLabel="Comparison" />
      ))}
    </div>
  );
}

/** RQ2 drops features a run did not measure (constant columns); say which, from its `meta`. */
function Unmeasured({ data }: { data: unknown }) {
  const { features, featureSets } = readUnmeasured(data);
  if (features.length === 0 && featureSets.length === 0) return null;
  return (
    <p className="max-w-[72ch] rounded-card border border-dashed border-(--frame) px-4 py-3 text-sm leading-relaxed text-muted sm:px-5">
      <span className="font-medium text-ink">Not measured in this run. </span>
      {features.length > 0 ? (
        <>
          {features.join(", ")} {features.length === 1 ? "was" : "were"} not measured, so{" "}
          {features.length === 1 ? "it is" : "they are"} left out of every model.{" "}
        </>
      ) : null}
      {featureSets.length > 0 ? `The ${featureSets.join(", ")} feature set${featureSets.length === 1 ? " was" : "s were"} skipped.` : null}
    </p>
  );
}

function Rq2({ data, run }: { data: unknown; run: string }) {
  const prefix = runPathPrefix(run);
  const tables = findMetricTables(data).filter((table) => table.path[0] !== "comparisons" || readComparisons(data).length === 0);
  const reliability = findReliability(data);
  const riskCoverage = findRiskCoverage(data);
  const nothing = tables.length === 0 && reliability.length === 0 && riskCoverage.length === 0;

  return (
    <div className="space-y-4">
      {nothing ? <Unrecognised file={`${prefix}rq2_confidence`} /> : null}
      <PubmedqaRuns run={run} />
      <Unmeasured data={data} />
      <Comparisons data={data} />

      {reliability.length > 0 || riskCoverage.length > 0 ? (
        <div className="grid gap-4 lg:grid-cols-2">
          {reliability.length > 0 ? (
            <ChartFrame
              title="When the model says 70%, is it right 70% of the time?"
              note="Reliability diagram: each point is one confidence bin, placed at its mean stated confidence and its observed accuracy. Points on the dashed diagonal are perfectly calibrated."
              legend={reliability.map((curve, i) => ({ name: curve.name, color: seriesColor(i), mark: "line" as const }))}
              table={{
                columns: [
                  { key: "model", label: "Model" },
                  { key: "conf", label: "Mean confidence", numeric: true },
                  { key: "acc", label: "Accuracy", numeric: true },
                  { key: "n", label: "Answers", numeric: true },
                ],
                rows: reliability.flatMap((curve) =>
                  curve.points.map((p) => ({
                    model: curve.name,
                    conf: fmtProb(p.x, 3),
                    acc: fmtProb(p.y, 3),
                    n: p.n === null ? "–" : fmtInt(p.n),
                  })),
                ),
              }}
            >
              <XYChart
                series={reliability.map((curve, i) => ({ name: curve.name, color: seriesColor(i), points: curve.points, markers: true }))}
                xLabel="Stated confidence"
                yLabel="Observed accuracy"
                xDomain={[0, 1]}
                yDomain={[0, 1]}
                diagonal="perfect calibration"
                ariaLabel={`Reliability diagram for ${reliability.map((c) => c.name).join(", ")}.`}
                height={300}
              />
            </ChartFrame>
          ) : null}

          {riskCoverage.length > 0 ? (
            <ChartFrame
              title="How does the error rate rise as the system answers more?"
              note="Risk-coverage curve: answers are released in order of confidence, most confident first. A curve that stays low for longer means confidence separates right answers from wrong ones."
              legend={riskCoverage.map((curve, i) => ({ name: curve.name, color: seriesColor(i), mark: "line" as const }))}
              table={{
                columns: [
                  { key: "model", label: "Model" },
                  { key: "coverage", label: "Coverage", numeric: true },
                  { key: "risk", label: "Selective risk", numeric: true },
                ],
                rows: riskCoverage.flatMap((curve) => {
                  const stride = Math.max(1, Math.floor(curve.points.length / 20));
                  return curve.points
                    .filter((_, i) => i % stride === 0 || i === curve.points.length - 1)
                    .map((p) => ({ model: curve.name, coverage: fmtProb(p.x, 3), risk: fmtProb(p.y, 3) }));
                }),
              }}
            >
              <XYChart
                series={riskCoverage.map((curve, i) => ({ name: curve.name, color: seriesColor(i), points: curve.points }))}
                xLabel="Coverage (share of questions answered)"
                yLabel="Selective risk"
                xDomain={[0, 1]}
                xFormat={(v) => fmtPct(v)}
                yFormat={(v) => fmtPct(v)}
                ariaLabel={`Risk-coverage curves for ${riskCoverage.map((c) => c.name).join(", ")}.`}
                height={300}
              />
            </ChartFrame>
          ) : null}
        </div>
      ) : null}

      {tables.map((table) => (
        <MetricTableBlock
          key={table.path.join("/")}
          table={table}
          title={tableTitle(table, "Confidence estimators")}
          rowLabel="Estimator"
        />
      ))}
      <RawFile data={data} name={`${prefix}rq2_confidence`} />
    </div>
  );
}

/* ---------- RQ3 ---------- */

function Rq3({ data, run }: { data: unknown; run: string }) {
  const { meta } = useApi();
  const prefix = runPathPrefix(run);
  const records = findLevelRecords(data);
  const fileDelta = readProvenance(data).delta;
  const delta = fileDelta ?? (typeof meta?.delta === "number" ? meta.delta : null);
  const deltaSource = fileDelta !== null ? "the results file" : "the server configuration";
  const tables = findMetricTables(data)
    .filter((table) => !table.path.some((key) => /^(?:alpha|a|α)?[_=\s]*0?\.\d+$/i.test(key)))
    .map(levelsAsRows);

  const alphas = [...new Set(records.map((r) => r.alpha))].sort((a, b) => a - b);
  const allMethods = [...new Set(records.map((r) => r.method))];
  // Variants nested one level down (single signals, minimum-selection sensitivity) would crowd the
  // charts; they are drawn only when nothing else is, and are always in the table.
  const topLevel = allMethods.filter((m) => !m.includes(" / "));
  const methods = topLevel.length > 0 ? topLevel : allMethods;
  const variantsHidden = methods.length < allMethods.length;
  const at = (method: string, alpha: number): LevelRecord | undefined => records.find((r) => r.method === method && r.alpha === alpha);
  const withViolation = methods.filter((m) => records.some((r) => r.method === m && r.violation));
  const withCoverage = methods.filter((m) => records.some((r) => r.method === m && r.coverage));

  // What the file says about the certified method, read from its records rather than written in.
  // A level where no split certified a threshold abstains on everything, so its zero violation
  // rate is trivial; those levels are named separately and left out of the summary ranges.
  const allCertified = records.filter((r) => r.method === "certified");
  const neverCertified = (r: LevelRecord) =>
    r.certifiedRate !== null ? r.certifiedRate === 0 : r.coverage !== null && r.coverage.value === 0;
  const uncertifiedAlphas = allCertified.filter(neverCertified).map((r) => r.alpha);
  const certified = allCertified.filter((r) => !neverCertified(r));
  const certifiedRates = certified.flatMap((r) => (r.certifiedRate !== null ? [r.certifiedRate] : []));
  const certifiedViolations = certified.flatMap((r) => (r.violation ? [r.violation.value] : []));
  const certifiedCoverage = certified.flatMap((r) => (r.coverage ? [r.coverage.value] : []));
  const fileMeta = isRecord(data) && isRecord(data.meta) ? data.meta : {};
  const scope = typeof fileMeta.guarantee === "string" ? fileMeta.guarantee : null;

  const recordTable = {
    columns: [
      { key: "method", label: "Method" },
      { key: "alpha", label: "α", numeric: true },
      { key: "violation", label: "Violation rate", numeric: true },
      { key: "interval", label: "Interval", numeric: true },
      { key: "coverage", label: "Coverage", numeric: true },
      { key: "risk", label: "Selective risk", numeric: true },
    ],
    rows: records.map((r) => ({
      method: r.method,
      alpha: fmtAlpha(r.alpha),
      violation: r.violation ? fmtPct(r.violation.value, 1) : "–",
      interval: r.violation && r.violation.lo !== null && r.violation.hi !== null ? `${fmtPct(r.violation.lo, 1)} to ${fmtPct(r.violation.hi, 1)}` : "–",
      coverage: r.coverage ? fmtPct(r.coverage.value, 1) : "–",
      risk: r.risk ? fmtPct(r.risk.value, 1) : "–",
    })),
  };

  return (
    <div className="space-y-4">
      {records.length === 0 && tables.length === 0 ? <Unrecognised file={`${prefix}rq3_conformal`} /> : null}

      {certifiedViolations.length > 0 && certifiedCoverage.length > 0 && delta !== null ? (
        <div className="max-w-[72ch] space-y-2 rounded-card border border-line bg-surface p-4 text-sm leading-relaxed text-ink sm:p-5">
          {uncertifiedAlphas.length > 0 ? (
            <p>
              At α = {uncertifiedAlphas.map(fmtAlpha).join(", ")} no threshold was certified in any split, so the
              system abstained on everything there and its zero violation rate says nothing about the certificate.
            </p>
          ) : null}
          <p>
            {uncertifiedAlphas.length > 0 ? "At the levels that did certify, the" : "For the certified threshold, the"}{" "}
            share of splits in which test risk exceeded α was at most {fmtPct(Math.max(...certifiedViolations), 1)},{" "}
            {Math.max(...certifiedViolations) <= delta ? "within" : "above"} δ = {fmtAlpha(delta)}. A split that
            certifies nothing counts as no violation
            {certifiedRates.length > 0
              ? `, and a threshold was certified in only ${fmtPct(Math.min(...certifiedRates), 1)} to ${fmtPct(Math.max(...certifiedRates), 1)} of splits`
              : ""}
            . Mean coverage
            ranged from {fmtPct(Math.min(...certifiedCoverage), 1)} to {fmtPct(Math.max(...certifiedCoverage), 1)} of
            test questions
            {Math.max(...certifiedCoverage) < 0.5
              ? `: ${Math.max(...certifiedViolations) <= delta ? "consistent with the certificate, but " : ""}at every level tested most questions are withheld.`
              : "."}
          </p>
          {scope ? <p className="text-[0.8125rem] text-muted">Scope, as recorded in the results file: {scope}</p> : null}
        </div>
      ) : null}

      {records.length > 0 ? (
        <div className="grid gap-4 lg:grid-cols-2">
          {withViolation.length > 0 ? (
            <ChartFrame
              title="How often did test risk exceed α?"
              note={
                <>
                  Each point is the share of repeated splits in which the error rate among released answers went above
                  α on held-out questions; bars are the reported intervals.{" "}
                  {delta !== null
                    ? `The guarantee tolerates this happening in at most ${fmtPct(delta)} of splits (δ, from ${deltaSource}): a method keeps its promise when its point sits at or below the dashed line.`
                    : "The tolerated rate δ is not stated in the file or by the server, so no target line is drawn."}
                  {variantsHidden ? " Single-signal and minimum-selection-size variants are in the table, not drawn." : ""}
                </>
              }
              legend={withViolation.map((name, i) => ({ name, color: seriesColor(i), mark: "dot" as const }))}
              table={recordTable}
            >
              <IntervalChart
                categories={alphas.map(fmtAlpha)}
                series={withViolation.map((name, i) => ({
                  name,
                  color: seriesColor(i),
                  values: alphas.map((alpha) => {
                    const v = at(name, alpha)?.violation;
                    return v ? { value: v.value, lo: v.lo, hi: v.hi } : null;
                  }),
                }))}
                xLabel="Target risk α"
                yLabel="Share of splits with risk above α"
                yFormat={(v) => fmtPct(v)}
                target={delta !== null ? { y: delta, label: `δ = ${fmtAlpha(delta)}` } : undefined}
                ariaLabel={`Violation rate by target risk for ${withViolation.join(", ")}.`}
                height={300}
              />
            </ChartFrame>
          ) : null}

          {withCoverage.length > 0 ? (
            <ChartFrame
              title="How many questions still get an answer at each risk level?"
              note="Mean share of held-out questions answered. A stricter α certifies a higher threshold, so fewer answers are released: this is the price of the guarantee."
              legend={withCoverage.map((name, i) => ({ name, color: seriesColor(i), mark: "line" as const }))}
              table={recordTable}
            >
              <XYChart
                series={withCoverage.map((name, i) => ({
                  name,
                  color: seriesColor(i),
                  markers: true,
                  points: alphas.flatMap((alpha) => {
                    const c = at(name, alpha)?.coverage;
                    return c ? [{ x: alpha, y: c.value }] : [];
                  }),
                }))}
                xLabel="Target risk α"
                yLabel="Coverage"
                yDomain={[0, 1]}
                xFormat={fmtAlpha}
                yFormat={(v) => fmtPct(v)}
                ariaLabel={`Coverage by target risk for ${withCoverage.join(", ")}.`}
                height={300}
              />
            </ChartFrame>
          ) : null}
        </div>
      ) : null}

      {tables.map((table) => (
        <MetricTableBlock
          key={table.path.join("/")}
          table={table}
          title={tableTitle(table, "Prediction sets")}
          rowLabel={table.rows.every((row) => row.name.startsWith("α = ")) ? "Risk level" : "Method"}
        />
      ))}
      <RawFile data={data} name={`${prefix}rq3_conformal`} />
    </div>
  );
}

/* ---------- Page ---------- */

function statusWord(loaded: Loaded<unknown>): string {
  switch (loaded.status) {
    case "loading":
      return "Checking";
    case "ready": {
      const created = readProvenance(loaded.data).created;
      return created ? `Run, written ${created}` : "Run";
    }
    case "missing":
      return "Not run yet";
    case "down":
      return "Unknown: API unreachable";
    case "error":
      return "Could not be read";
  }
}

function Section({ question, loaded, run }: { question: Question; loaded: Loaded<unknown>; run: string }) {
  const file = `${runPathPrefix(run)}${question.file}`;
  return (
    <section id={question.id} aria-labelledby={`${question.id}-heading`} className="scroll-mt-28 border-t border-line pt-8">
      <div className="max-w-[60rem]">
        <h2 id={`${question.id}-heading`} className="font-serif text-[1.5rem] font-semibold leading-tight tracking-[-0.012em] text-ink">
          {question.label}: {question.topic}
        </h2>
        <p className="mt-1.5 max-w-[62ch] font-serif text-[1.1875rem] leading-snug text-ink">{question.question}</p>
        {loaded.status === "ready" ? <Provenance data={loaded.data} file={file} /> : null}
        {question.generatorIndependent ? (
          <p className="mt-1.5 max-w-[72ch] text-xs text-muted">
            Retrieval and encoder training involve no generator, so this section reads the main run whichever generator
            is picked above.
          </p>
        ) : null}
      </div>

      <div className="mt-5" aria-busy={loaded.status === "loading"}>
        {loaded.status === "loading" ? <p className="text-sm text-muted">Looking for results/{file}.json.</p> : null}

        {loaded.status === "missing" && run !== MAIN_RUN ? (
          <EmptyBlock title={`${question.label} was not computed for this run`}>
            <p>
              <Code wrap>results/{file}.json</Code> does not exist, so nothing is drawn here. Pick another run above to see{" "}
              {question.label}.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "missing" && run === MAIN_RUN ? (
          <EmptyBlock
            title="This experiment has not been run yet"
            aside={
              <>
                <span className="font-medium text-ink">What it will report. </span>
                {question.willShow}
              </>
            }
          >
            <p>
              Run <Code>{question.command}</Code> from the project root. It writes <Code wrap>results/{question.file}.json</Code>,
              and this section draws itself from that file.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "down" ? (
          <EmptyBlock
            title="Not loaded"
            aside={
              <>
                <span className="font-medium text-ink">What the experiment reports. </span>
                {question.willShow}
              </>
            }
          >
            <p>
              Results are read through the API, which is not answering, so this page cannot tell whether{" "}
              <Code>{question.command}</Code> has been run. No figure is shown in its place.
            </p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "error" ? (
          <EmptyBlock title="The results file could not be read">
            <p>{loaded.message}</p>
          </EmptyBlock>
        ) : null}

        {loaded.status === "ready" ? (
          question.id === "rq1" ? (
            <Rq1 data={loaded.data} />
          ) : question.id === "rq2" ? (
            <Rq2 data={loaded.data} run={run} />
          ) : (
            <Rq3 data={loaded.data} run={run} />
          )
        ) : null}
      </div>
    </section>
  );
}

/** "N failed generations (rag), excluded from scoring", from the run's own counts. */
function failedText(info: RunInfo): string | null {
  if (!info.n_failed) return null;
  const entries = Object.entries(info.n_failed).filter(([, n]) => typeof n === "number");
  const total = entries.reduce((sum, [, n]) => sum + n, 0);
  if (entries.length === 0) return null;
  if (total === 0) return "no failed generations";
  const failing = entries.filter(([, n]) => n > 0);
  const where = failing.length === 1 ? failing[0]?.[0] : failing.map(([c, n]) => `${fmtInt(n)} in ${c}`).join(", ");
  return `${fmtInt(total)} failed generation${total === 1 ? "" : "s"} (${where}), excluded from scoring`;
}

/** Which generator's answers RQ2 and RQ3 are drawn from, and what that run did and did not measure. */
function RunPicker({ runs, run, onChange }: { runs: RunInfo[]; run: string; onChange: (run: string) => void }) {
  const info = runs.find((r) => r.run === run);
  const k = info?.self_consistency_samples ?? null;
  const conditions = info?.conditions ?? null;
  const subset = info?.subset_of ?? null;
  const parent = subset ? runs.find((r) => r.run === subset.run) : undefined;
  const parentLabel = parent ? `"${runLabel(parent, runs)}"` : "larger";
  const parts = info
    ? [
        info.generator,
        k !== null ? `${fmtInt(k)} sample${k === 1 ? "" : "s"} per question` : null,
        info.n !== null ? `n = ${fmtInt(info.n)} questions` : null,
        failedText(info),
      ].filter(Boolean)
    : [];

  return (
    <section aria-labelledby="run-heading" className="mt-6 max-w-[60rem] rounded-card border border-line bg-surface p-4 sm:p-5">
      <h2 id="run-heading" className="sr-only">
        Generator run
      </h2>
      <Segmented
        legend="Generator whose answers RQ2 and RQ3 score"
        name="run"
        value={run}
        options={runs.map((r) => ({ value: r.run, label: runLabel(r, runs) }))}
        onChange={onChange}
      />
      {parts.length > 0 ? <p className="num mt-2 text-xs text-muted">{parts.join(", ")}.</p> : null}
      <ul className="mt-2 max-w-[72ch] list-disc space-y-1 pl-5 text-sm leading-relaxed text-muted">
        {k === 1 ? (
          <li>One sample per question: self-consistency could not be measured, so RQ2 leaves those signals out.</li>
        ) : null}
        {conditions && !conditions.includes("closed_book") && !conditions.includes("oracle") ? (
          <li>Retrieval condition only: no closed-book or oracle answers, so no comparison between conditions.</li>
        ) : null}
        {subset ? (
          <li>
            {subset.prefix
              ? `These are the first ${fmtInt(info?.n)} PubMedQA questions in dataset order of the ${parentLabel} run, not a random sample, so its numbers need not carry over to the full set.`
              : `These ${fmtInt(info?.n)} questions are a subset of the ${parentLabel} run; how they were chosen is not recorded, so its numbers need not carry over to the full set.`}{" "}
            On the {fmtInt(subset.answered_in_both)} questions answered in both, the first-sample decision matches the{" "}
            {parentLabel} run for {fmtInt(subset.first_sample_matches)}.
            {k !== null && k > 1
              ? ` A question counts as failed here if any of its ${fmtInt(k)} samples was invalid${
                  subset.failed_only_here > 0 ? `; ${fmtInt(subset.failed_only_here)} ${subset.failed_only_here === 1 ? "question failed here but was" : "questions failed here but were"} answered in the ${parentLabel} run` : ""
                }.`
              : null}
            {info && !info.available.includes("rq3_conformal") ? " RQ3 was not run on this subset." : null}
          </li>
        ) : null}
        <li>
          Each run is shown on its own. Differences between generators are descriptive: no test pairs the runs, and each
          run&apos;s RQ2 and RQ3 are computed on splits of its own questions.
        </li>
      </ul>
    </section>
  );
}

export function ExperimentsView() {
  const { reachability } = useApi();
  const [picked, setPicked] = useState(MAIN_RUN);
  const index = useLoaded((signal) => getExperimentsIndex(signal), "experiments-index");
  const runs = index.status === "ready" ? (index.data.runs ?? []) : [];
  // A run the server no longer lists falls back to the main one.
  const run = runs.some((r) => r.run === picked) ? picked : MAIN_RUN;
  const rq1 = useExperiment("rq1_retrieval");
  const rq2 = useExperiment("rq2_confidence", run);
  const rq3 = useExperiment("rq3_conformal", run);
  const loaded = { rq1, rq2, rq3 };

  return (
    <div>
      <div className="max-w-[60rem]">
        <h1 className="font-serif text-[1.75rem] font-semibold leading-tight tracking-[-0.015em] text-ink">Experiments</h1>
        <p className="mt-1 max-w-[68ch] text-[0.9375rem] leading-relaxed text-muted">
          Three research questions, each answered by one script and one results file. Every number on this page is
          read from those files at the moment you open it. Where a file does not exist, the page says so and shows
          nothing in its place.
        </p>
      </div>

      {runs.length > 1 ? <RunPicker runs={runs} run={run} onChange={setPicked} /> : null}

      {reachability === "down" ? (
        <div className="mt-5">
          <ApiDown
            what="Results files are served by it, so none can be shown."
            aside={
              <>
                <span className="font-medium text-ink">Nothing here is cached or typed in. </span>
                Each figure on this page is drawn from a results file at the moment the page loads. With the API
                unreachable there is no file to read, so the sections below stay empty.
              </>
            }
          />
        </div>
      ) : null}

      <div className="relative mt-6 overflow-x-auto rounded-card border border-line bg-surface">
        <table className="w-full text-sm">
          <caption className="sr-only">Status of the three experiments</caption>
          <thead>
            <tr className="text-xs text-muted">
              <th scope="col" className="px-4 py-2.5 text-left font-normal sm:px-5">
                Question
              </th>
              <th scope="col" className="hidden px-4 py-2.5 text-left font-normal sm:table-cell">
                Script
              </th>
              <th scope="col" className="px-4 py-2.5 text-left font-normal sm:pr-5">
                Results file
              </th>
            </tr>
          </thead>
          <tbody>
            {QUESTIONS.map((question) => {
              const state = loaded[question.id];
              return (
                <tr key={question.id} className="border-t border-rule align-baseline">
                  <th scope="row" className="px-4 py-3 text-left font-normal sm:px-5">
                    <a href={`#${question.id}`} className="font-medium text-ink underline decoration-control underline-offset-4 hover:decoration-ink">
                      {question.label}
                    </a>
                    <span className="ml-2 text-muted">{question.topic}</span>
                  </th>
                  <td className="hidden px-4 py-3 sm:table-cell">
                    <Code>{question.command}</Code>
                  </td>
                  <td className="px-4 py-3 sm:pr-5">
                    <span className={state.status === "ready" ? "font-medium text-ink" : "text-muted"}>{statusWord(state)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="mt-10 space-y-12">
        {QUESTIONS.map((question) => (
          <Section
            key={question.id}
            question={question}
            loaded={loaded[question.id]}
            run={question.generatorIndependent ? MAIN_RUN : run}
          />
        ))}
      </div>
    </div>
  );
}
