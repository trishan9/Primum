"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";

import { ChartCard } from "@/components/charts/ChartCard";
import { ChartSkeleton } from "@/components/charts/ChartSkeleton";
import { Picker } from "@/components/ui/picker";
import {
  confidenceTakeaway,
  confusionByCondition,
  confusionTakeaway,
  conditionLabel,
  DECISIONS,
  histogramValues,
  recordConditions,
  verbalConfidenceHistogram,
  verbalConfidenceSummaries,
  type HistogramScale,
} from "@/lib/charts/pubmedqa";
import { fmtInt, fmtPct } from "@/lib/format";

import { CONFIDENCE_HIST_HEIGHT, CONFUSION_HEIGHT } from "./benchmark-heights";

const ConfusionChart = dynamic(() => import("./benchmark-confusion-chart").then((m) => m.ConfusionChart), {
  ssr: false,
  loading: () => <ChartSkeleton height={CONFUSION_HEIGHT} />,
});
const ConfidenceHistogramChart = dynamic(() => import("./benchmark-confidence-chart").then((m) => m.ConfidenceHistogramChart), {
  ssr: false,
  loading: () => <ChartSkeleton height={CONFIDENCE_HIST_HEIGHT} />,
});

/** A condition switch, shown only when the run answered under more than one. */
function ConditionPicker({ conditions, value, onChange }: { conditions: readonly string[]; value: string; onChange: (value: string) => void }) {
  if (conditions.length < 2) return null;
  return (
    <Picker legend="Evidence condition" value={value} options={conditions.map((c) => ({ value: c, label: conditionLabel(c) }))} onChange={onChange} segmentedFrom="sm" />
  );
}

const pick = (conditions: readonly string[], picked: string | null, preferred = "rag") =>
  picked && conditions.includes(picked) ? picked : conditions.includes(preferred) ? preferred : (conditions[0] ?? "");

/* ---------- Confusion ---------- */

/** Gold yes/no/maybe against the decision, from the per-question records. */
export function ConfusionCard({ data, source }: { data: unknown; source: string }) {
  const all = useMemo(() => confusionByCondition(data), [data]);
  const conditions = useMemo(() => all.map((c) => c.condition), [all]);
  const [picked, setPicked] = useState<string | null>(null);
  const condition = pick(conditions, picked);
  const confusion = all.find((c) => c.condition === condition);
  const takeaway = useMemo(() => (confusion ? confusionTakeaway(confusion) : null), [confusion]);
  if (!confusion || confusion.n === 0) return null;

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Which labels does it confuse? {conditionLabel(condition)}</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Actions className="w-full min-w-0 shrink sm:w-auto">
          <ConditionPicker conditions={conditions} value={condition} onChange={setPicked} />
        </ChartCard.Actions>
      </ChartCard.Header>
      <ChartCard.Body height={CONFUSION_HEIGHT}>
        <ConfusionChart confusion={confusion} ariaLabel={`Gold label against decision, ${conditionLabel(condition).toLowerCase()}. ${takeaway ?? ""}`} />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "gold", label: "Gold" },
          ...DECISIONS.map((d) => ({ key: d, label: `Answered ${d}`, numeric: true })),
          { key: "total", label: "Total", numeric: true },
        ]}
        rows={DECISIONS.map((gold) => ({
          gold,
          ...Object.fromEntries(
            DECISIONS.map((d) => {
              const cell = confusion.cells.find((c) => c.gold === gold && c.decision === d);
              return [d, `${fmtInt(cell?.count)} (${fmtPct(cell?.shareOfGold, 0)})`];
            }),
          ),
          total: fmtInt(confusion.goldTotals[gold]),
        }))}
      >
        <ChartCard.Note>
          Counts with the share of each gold row; colour is that share. {fmtInt(confusion.n)} scored
          {confusion.failed > 0 ? `, ${fmtInt(confusion.failed)} failed generations left out` : ""}.
        </ChartCard.Note>
        <ChartCard.Source>{source} · records</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}

/* ---------- Stated confidence ---------- */

const BINS = 10;

/** The generator's own stated confidence, for answers that turned out right and wrong. */
export function ConfidenceCard({ data, source }: { data: unknown; source: string }) {
  const conditions = useMemo(() => recordConditions(data), [data]);
  const [picked, setPicked] = useState<string | null>(null);
  const condition = pick(conditions, picked);
  const [scale, setScale] = useState<HistogramScale>("share");
  const hist = useMemo(() => verbalConfidenceHistogram(data, condition, BINS), [data, condition]);
  const summaries = useMemo(() => verbalConfidenceSummaries(data, condition), [data, condition]);
  const takeaway = useMemo(() => confidenceTakeaway(summaries), [summaries]);
  if (!hist || hist.nCorrect + hist.nWrong === 0) return null;
  const values = histogramValues(hist, scale);
  const fmt = (v: number) => (scale === "count" ? fmtInt(v) : fmtPct(v, 1));

  return (
    <ChartCard>
      <ChartCard.Header>
        <ChartCard.Heading>
          <ChartCard.Title as="h3">Is it less sure when it is wrong? {conditionLabel(condition)}</ChartCard.Title>
          {takeaway ? <ChartCard.Takeaway>{takeaway}</ChartCard.Takeaway> : null}
        </ChartCard.Heading>
        <ChartCard.Actions className="w-full min-w-0 shrink sm:w-auto">
          <ConditionPicker conditions={conditions} value={condition} onChange={setPicked} />
          <Picker
            legend="Bar height"
            value={scale}
            options={[
              { value: "share", label: "Share of group" },
              { value: "count", label: "Count" },
            ]}
            onChange={setScale}
            segmentedFrom="sm"
          />
        </ChartCard.Actions>
      </ChartCard.Header>
      <ChartCard.Body height={CONFIDENCE_HIST_HEIGHT}>
        <ConfidenceHistogramChart
          hist={hist}
          scale={scale}
          medians={{ correct: summaries.correct, wrong: summaries.wrong }}
          ariaLabel={`Stated confidence of right and wrong answers, ${scale === "count" ? "counts" : "share of each group"}, ${conditionLabel(condition).toLowerCase()}. ${takeaway ?? ""}`}
        />
      </ChartCard.Body>
      <ChartCard.Data
        columns={[
          { key: "bin", label: "Stated confidence" },
          { key: "right", label: scale === "count" ? "Right" : "Right, share", numeric: true },
          { key: "wrong", label: scale === "count" ? "Wrong" : "Wrong, share", numeric: true },
        ]}
        rows={values
          .filter((b) => b.correct > 0 || b.wrong > 0)
          .map((b) => ({ bin: hist.distinct ? b.lo.toFixed(2) : `${b.lo.toFixed(1)} to ${b.hi.toFixed(1)}`, right: fmt(b.correct), wrong: fmt(b.wrong) }))}
      >
        <ChartCard.Note>
          {hist.distinct ? "One bin per stated value" : `${BINS} bins of 0.1 over 0 to 1`} (empty bins omitted from the table). Right above the axis, wrong
          below; ticks mark each group&rsquo;s median. Share: each bar over its own group, so the two shapes compare despite unequal group sizes.{hist.missing > 0 ? ` ${fmtInt(hist.missing)} answers stated no confidence.` : ""}
        </ChartCard.Note>
        <ChartCard.Source>{source} · records</ChartCard.Source>
      </ChartCard.Data>
    </ChartCard>
  );
}
