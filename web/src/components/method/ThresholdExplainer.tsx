"use client";

import { useId, useMemo, useState } from "react";

import { linear } from "@/components/charts/scale";
import { useWidth } from "@/components/charts/useSize";
import { useLoaded, type Loaded } from "@/components/experiments/useExperiment";
import { useApi } from "@/components/shell/ApiProvider";
import { Code, EmptyBlock, Segmented } from "@/components/ui/controls";
import { getCalibration, getExperimentsIndex, MAIN_RUN } from "@/lib/api";
import { fmtAlpha, fmtInt, fmtP, fmtPct, fmtProb } from "@/lib/format";
import { generatorName, isRecord, runLabel, runPathPrefix } from "@/lib/results";
import {
  certifyThreshold,
  DEFAULT_MIN_SELECTED,
  parseCalibration,
  selectAt,
  thresholdSweep,
  type CalibrationData,
  type CalibrationRows,
  type Selection,
} from "@/lib/stats";

const M = { top: 30, right: 16, bottom: 40, left: 46 };
const H = 320;

/** The lowest threshold whose observed risk is within alpha, with no statistical correction. */
function plugInThreshold(data: CalibrationRows, alpha: number, minSelected: number): Selection | null {
  const candidates = [...new Set(data.conf)].sort((a, b) => a - b);
  for (const threshold of candidates) {
    const selection = selectAt(data.conf, data.correct, threshold);
    if (selection.nSelected < minSelected) break;
    if (selection.risk <= alpha) return selection;
  }
  return null;
}

function Chart({
  sweep,
  alpha,
  certified,
  plugIn,
  deployed,
}: {
  sweep: Selection[];
  alpha: number;
  certified: number | null;
  plugIn: number | null;
  deployed: { alpha: number; threshold: number | null }[];
}) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const patternId = useId();
  const plotW = Math.max(0, width - M.left - M.right);
  const plotH = H - M.top - M.bottom;
  const sx = linear([0, 1], [M.left, M.left + plotW]);
  const sy = linear([0, 1], [M.top + plotH, M.top]);
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1];

  const path = (key: "risk" | "coverage") => {
    let d = "";
    let pen = false;
    for (const point of sweep) {
      const y = point[key];
      if (!Number.isFinite(y) || (key === "risk" && point.nSelected < 5)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${sx(point.threshold).toFixed(1)} ${sy(y).toFixed(1)} `;
      pen = true;
    }
    return d;
  };

  const withheldTo = certified ?? 1;
  const riskEnd = [...sweep].reverse().find((p) => Number.isFinite(p.risk) && p.nSelected >= 5);

  return (
    <div ref={ref}>
      {width > 0 ? (
        <svg
          width={width}
          height={H}
          role="img"
          aria-label={`Selective risk and coverage as the threshold moves from 0 to 1. ${
            certified === null
              ? `No threshold is certified for alpha ${fmtAlpha(alpha)}.`
              : `The certified threshold for alpha ${fmtAlpha(alpha)} is ${fmtProb(certified)}.`
          }`}
          className="block"
        >
          <defs>
            <pattern id={patternId} width="8" height="8" patternUnits="userSpaceOnUse">
              <path d="M-2 2 2 -2M0 8 8 0M6 10 10 6" stroke="var(--hatch)" strokeWidth="1.1" />
            </pattern>
          </defs>

          <rect x={sx(0)} y={M.top} width={sx(withheldTo) - sx(0)} height={plotH} fill="var(--raised)" opacity="0.6" />
          <rect x={sx(0)} y={M.top} width={sx(withheldTo) - sx(0)} height={plotH} fill={`url(#${patternId})`} opacity={certified === null ? 0.3 : 0.55} />
          {plotW > 300 && sx(withheldTo) - sx(0) > 90 ? (
            <text x={sx(0) + 8} y={M.top + plotH - 9} className="fill-(--text) text-[12px]">
              withheld
            </text>
          ) : null}

          <text x={0} y={11} className="fill-(--muted) text-[12px]">
            Share of questions
          </text>
          {ticks.map((tick) => (
            <g key={tick}>
              <line x1={M.left} x2={M.left + plotW} y1={sy(tick)} y2={sy(tick)} stroke="var(--grid)" strokeWidth="1" />
              <text x={M.left - 8} y={sy(tick)} dy="0.32em" textAnchor="end" className="num fill-(--muted) text-[12px]">
                {fmtPct(tick)}
              </text>
              <text
                x={sx(tick)}
                y={M.top + plotH + 16}
                textAnchor={tick === 0 ? "start" : tick === 1 ? "end" : "middle"}
                className="num fill-(--muted) text-[12px]"
              >
                {tick === 0 ? "0" : tick === 1 ? "1" : tick.toFixed(1)}
              </text>
            </g>
          ))}
          <line x1={M.left} x2={M.left + plotW} y1={sy(0)} y2={sy(0)} stroke="var(--axis)" strokeWidth="1" />
          <text x={M.left + plotW / 2} y={H - 4} textAnchor="middle" className="fill-(--muted) text-[12px]">
            {plotW > 420
              ? "Confidence threshold: answer only when P(correct) is at or above it"
              : "Threshold on P(correct)"}
          </text>

          {/* thresholds the server certified on its own split, for comparison */}
          {deployed.map((item) =>
            item.threshold === null ? null : (
              <g key={item.alpha}>
                <path d={`M${sx(item.threshold) - 4} ${M.top - 9}h8l-4 7Z`} fill="var(--muted)" />
                <title>{`Deployed threshold for α = ${fmtAlpha(item.alpha)}: ${fmtProb(item.threshold)}`}</title>
              </g>
            ),
          )}

          <path d={path("coverage")} fill="none" stroke="var(--series-rest)" strokeWidth="2" strokeLinejoin="round" />
          <path d={path("risk")} fill="none" stroke="var(--series-1)" strokeWidth="2" strokeLinejoin="round" />
          {sweep[0] ? (
            <text x={sx(0.02)} y={sy(sweep[0].coverage) + 16} className="fill-(--text) text-[12px]">
              coverage
            </text>
          ) : null}
          {riskEnd && Number.isFinite(sweep[0]?.risk) ? (
            <text x={sx(0.02)} y={sy(sweep[0]?.risk ?? 0) - 7} className="fill-(--text) text-[12px]">
              error rate among answered
            </text>
          ) : null}

          <line x1={M.left} x2={M.left + plotW} y1={sy(alpha)} y2={sy(alpha)} stroke="var(--text)" strokeWidth="1" strokeDasharray="5 4" />
          {/* Left end: the right end is where the certified threshold line usually lands. */}
          <text x={M.left + 6} y={sy(alpha) - 5} textAnchor="start" className="num fill-(--text) text-[12px] font-medium">
            α = {fmtAlpha(alpha)}
          </text>

          {plugIn !== null && (certified === null || Math.abs(plugIn - certified) > 0.004) ? (
            <g>
              <line x1={sx(plugIn)} x2={sx(plugIn)} y1={M.top} y2={M.top + plotH} stroke="var(--muted)" strokeWidth="1.5" strokeDasharray="2 4" />
              {/* Second row, so it never collides with the certified label when the two are close. */}
              <text x={sx(plugIn) - 5} y={M.top + (certified !== null ? 30 : 12)} textAnchor="end" className="num fill-(--muted) text-[12px]">
                {plotW > 420 ? "uncorrected " : ""}
                {fmtProb(plugIn)}
              </text>
            </g>
          ) : null}

          {certified !== null ? (
            <g>
              <line x1={sx(certified)} x2={sx(certified)} y1={M.top - 2} y2={M.top + plotH} stroke="var(--text)" strokeWidth="2" />
              <text
                x={sx(certified) + (certified > 0.72 ? -6 : 6)}
                y={M.top + 12}
                textAnchor={certified > 0.72 ? "end" : "start"}
                className="num fill-(--text) text-[13px] font-semibold"
              >
                {plotW > 420 ? "certified " : ""}
                {fmtProb(certified)}
              </text>
            </g>
          ) : null}
        </svg>
      ) : (
        <div style={{ height: H }} />
      )}
    </div>
  );
}

type Split = "calibration" | "heldout";

function Explainer({ data }: { data: CalibrationData }) {
  const id = useId();
  const { meta } = useApi();
  const [alpha, setAlpha] = useState(0.1);
  const [split, setSplit] = useState<Split>("calibration");
  const delta = data.delta ?? (typeof meta?.delta === "number" ? meta.delta : 0.1);
  const deltaKnown = data.delta !== null || typeof meta?.delta === "number";
  // The server's minimum selection size, so the walk and the plug-in rule here match its own.
  const minSelected = data.minSelected ?? DEFAULT_MIN_SELECTED;

  const heldout = data.heldout;
  const onHeldout = split === "heldout" && heldout !== null;
  // The rows the reader is looking at. The certificate itself is always computed on `data`.
  const shown: CalibrationRows = onHeldout ? heldout : data;

  const sweep = useMemo(() => thresholdSweep(shown.conf, shown.correct, 201), [shown]);
  const certificate = useMemo(
    () =>
      certifyThreshold(data.conf, data.correct, {
        alpha,
        delta,
        nGrid: data.nGrid ?? undefined,
        minSelected,
      }),
    [data, alpha, delta, minSelected],
  );
  const plugIn = useMemo(() => plugInThreshold(data, alpha, minSelected), [data, alpha, minSelected]);
  const all = useMemo(() => selectAt(shown.conf, shown.correct, 0), [shown]);

  // The certified threshold applied to the rows on screen: in-sample on the calibration split,
  // out-of-sample on the held-out one.
  const selection = certificate.threshold === null ? null : selectAt(shown.conf, shown.correct, certificate.threshold);
  const plugInShown = plugIn ? selectAt(shown.conf, shown.correct, plugIn.threshold) : null;
  const failed = certificate.steps.find((step) => !step.passed);
  const first = certificate.steps[0];
  const presets = data.certified.length > 0 ? data.certified.map((c) => c.alpha) : [0.05, 0.1, 0.15, 0.2];
  const splitWord = onHeldout ? "held-out" : "calibration";

  return (
    <div className="rounded-card border border-line bg-surface">
      <div className="border-b border-rule p-4 sm:p-5">
        <div className="flex flex-wrap items-end gap-x-6 gap-y-3">
          <div className="min-w-0 flex-1 basis-72">
            <label htmlFor={`${id}-alpha`} className="flex items-baseline justify-between gap-3 text-sm font-medium text-ink">
              <span>Tolerated error rate among released answers (α)</span>
              <output htmlFor={`${id}-alpha`} className="num text-[1.0625rem] font-semibold">
                {fmtAlpha(alpha)}
              </output>
            </label>
            <input
              id={`${id}-alpha`}
              type="range"
              min={0.02}
              max={0.4}
              step={0.01}
              value={alpha}
              onChange={(event) => setAlpha(Number(event.target.value))}
              className="mt-2 h-11 w-full accent-[var(--accent)] sm:h-6"
            />
          </div>
          <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Risk levels the deployed system offers">
            {presets.map((preset) => (
              <button
                key={preset}
                type="button"
                aria-pressed={Math.abs(alpha - preset) < 1e-9}
                onClick={() => setAlpha(preset)}
                className={`num min-h-11 min-w-12 rounded-control border px-2.5 text-sm sm:min-h-8 ${
                  Math.abs(alpha - preset) < 1e-9
                    ? "border-primary bg-primary font-medium text-on-primary"
                    : "border-control text-muted hover:border-ink hover:text-ink"
                }`}
              >
                {fmtAlpha(preset)}
              </button>
            ))}
          </div>
        </div>
        <div className="mt-4 flex flex-wrap items-end gap-x-6 gap-y-2">
          {heldout ? (
            <Segmented<Split>
              legend="Questions the threshold is evaluated on"
              name={`${id}-split`}
              value={split}
              onChange={setSplit}
              options={[
                { value: "calibration", label: `Calibration, ${fmtInt(data.conf.length)}` },
                { value: "heldout", label: `Held out, ${fmtInt(heldout.conf.length)}` },
              ]}
            />
          ) : null}
          <p className="min-w-0 flex-1 basis-72 text-[0.8125rem] leading-relaxed text-muted">
            The certificate is computed on the calibration split only.{" "}
            {heldout
              ? onHeldout
                ? "The held-out questions played no part in fitting the confidence model, calibrating it or choosing the threshold, so what you see here is out-of-sample."
                : "Switch to the held-out questions to see the same threshold applied to questions that played no part in choosing it."
              : "The server sent no held-out questions, so the threshold can only be shown on the questions it was certified on."}
          </p>
        </div>
      </div>

      <dl className="num grid grid-cols-2 gap-px border-b border-rule bg-(--rule) text-sm sm:grid-cols-4" aria-live="polite">
        <div className="bg-surface p-4 sm:px-5">
          <dt className="text-xs text-muted">Certified threshold, from calibration</dt>
          <dd className="mt-0.5 text-[1.5rem] font-semibold leading-tight text-ink">
            {certificate.threshold === null ? "none" : fmtProb(certificate.threshold)}
          </dd>
        </div>
        <div className="bg-surface p-4 sm:px-5">
          <dt className="text-xs text-muted">Questions answered, {splitWord}</dt>
          <dd className="mt-0.5 text-[1.5rem] font-semibold leading-tight text-ink">
            {selection ? fmtPct(selection.coverage) : "0%"}
            <span className="ml-1.5 text-xs font-normal text-muted">
              {selection ? fmtInt(selection.nSelected) : 0} of {fmtInt(shown.conf.length)}
            </span>
          </dd>
        </div>
        <div className="bg-surface p-4 sm:px-5">
          <dt className="text-xs text-muted">Wrong among answered, {splitWord}</dt>
          <dd className="mt-0.5 text-[1.5rem] font-semibold leading-tight text-ink">
            {selection && selection.nSelected > 0 ? fmtPct(selection.risk, 1) : "–"}
            {selection && selection.nSelected > 0 ? (
              <span className="ml-1.5 text-xs font-normal text-muted">
                {fmtInt(selection.nErrors)} {selection.nErrors === 1 ? "error" : "errors"}
              </span>
            ) : null}
          </dd>
        </div>
        <div className="bg-surface p-4 sm:px-5">
          <dt className="text-xs text-muted">p-value of the test, calibration</dt>
          <dd className="mt-0.5 text-[1.5rem] font-semibold leading-tight text-ink">
            {fmtP(certificate.pValue)}
            <span className="ml-1.5 whitespace-nowrap text-xs font-normal text-muted">limit δ = {fmtAlpha(delta)}</span>
          </dd>
        </div>
      </dl>

      <div className="p-4 sm:p-5">
        <Chart
          sweep={sweep}
          alpha={alpha}
          certified={certificate.threshold}
          plugIn={plugIn ? plugIn.threshold : null}
          deployed={data.certified}
        />

        <div className="mt-4 max-w-[72ch] space-y-2 text-sm leading-relaxed text-ink">
          {certificate.threshold === null ? (
            <p>
              No threshold can be certified at α = {fmtAlpha(alpha)}.{" "}
              {first
                ? `Even the strictest candidate tested on the calibration questions, ${fmtProb(first.threshold)}, answering ${fmtInt(first.nSelected)} of them with ${fmtInt(first.nErrors)} wrong, gives a p-value of ${fmtP(first.pValue)}, above δ.`
                : `Too few calibration questions clear any candidate threshold to run the test (at least ${fmtInt(minSelected)} must be answered).`}{" "}
              At this level the system withholds every answer.
            </p>
          ) : (
            <p>
              On the calibration questions, the walk tested {fmtInt(certificate.steps.length)} candidate{" "}
              {certificate.steps.length === 1 ? "threshold" : "thresholds"}, starting from the strictest ({first ? fmtProb(first.threshold) : "–"})
              and moving down.{" "}
              {failed
                ? `It stopped at ${fmtProb(failed.threshold)}, where ${fmtInt(failed.nErrors)} wrong out of ${fmtInt(failed.nSelected)} gave a p-value of ${fmtP(failed.pValue)}, above δ. The last threshold to pass, ${fmtProb(certificate.threshold)}, is the certified one.`
                : `Every candidate passed, so the lowest one, ${fmtProb(certificate.threshold)}, is certified.`}
            </p>
          )}
          {onHeldout && selection ? (
            <p>
              {selection.nSelected === 0
                ? `On the ${fmtInt(shown.conf.length)} held-out questions, no confidence reaches ${fmtProb(selection.threshold)}, so nothing would be answered.`
                : `On the ${fmtInt(shown.conf.length)} held-out questions, the same threshold answers ${fmtInt(selection.nSelected)} with ${fmtInt(selection.nErrors)} wrong: an error rate of ${fmtPct(selection.risk, 1)}, ${selection.risk <= alpha ? "within" : "above"} α = ${fmtAlpha(alpha)}.`}{" "}
              One held-out sample can land above α by chance. The guarantee bounds how often that happens across
              calibration sets, at most δ, not what happens on any one sample, and with few questions answered the
              observed rate is itself uncertain.
            </p>
          ) : null}
          {!onHeldout && plugIn && selection && certificate.threshold !== null && certificate.threshold - plugIn.threshold > 0.004 ? (
            <p>
              Without the test, you would pick {fmtProb(plugIn.threshold)}: the lowest threshold whose observed error
              rate is within α on these questions. The certified threshold sits{" "}
              {fmtProb(certificate.threshold - plugIn.threshold)} higher. That gap is the safety margin, and it costs{" "}
              {fmtPct(plugIn.coverage - selection.coverage, 1)} of coverage.
            </p>
          ) : null}
          {onHeldout && plugIn && plugInShown && plugInShown.nSelected > 0 && (certificate.threshold === null || Math.abs(plugIn.threshold - certificate.threshold) > 0.004) ? (
            <p>
              The uncorrected threshold chosen on the calibration questions, {fmtProb(plugIn.threshold)}, answers{" "}
              {fmtInt(plugInShown.nSelected)} held-out questions with an error rate of {fmtPct(plugInShown.risk, 1)}.
            </p>
          ) : null}
          <p className="text-[0.8125rem] text-muted">
            Answering everything on these {fmtInt(shown.conf.length)} {splitWord} questions is wrong{" "}
            {fmtPct(all.risk, 1)} of the time. The certificate is computed in your browser from the{" "}
            {fmtInt(data.conf.length)} calibration questions served by <Code>/api/calibration</Code>, with the same
            fixed-sequence binomial test as <Code>primum.uncertainty.conformal</Code>
            {data.minSelected !== null
              ? ` and the server's minimum of ${fmtInt(data.minSelected)} answered questions per tested threshold.`
              : `. The server did not send its minimum selection size, so ${fmtInt(DEFAULT_MIN_SELECTED)} is used, and the thresholds here may differ from the server's.`}{" "}
            {data.certified.some((c) => c.threshold !== null)
              ? "Small triangles above the plot mark the thresholds the deployed system applies; with the same settings they should coincide with the certified line at the same α."
              : ""}
            {deltaKnown ? "" : " The server did not state δ, so the project default of 0.10 is used."} The guarantee is
            for PubMedQA questions answered by the same pipeline; for questions asked in chat the threshold is an
            operating point.
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * The interactive threshold explainer, with a picker over the generator runs that have RQ3
 * results. It draws nothing until real calibration data exists.
 */
export function ThresholdExplainer() {
  const [picked, setPicked] = useState(MAIN_RUN);
  const index = useLoaded((signal) => getExperimentsIndex(signal), "experiments-index");
  const runs = index.status === "ready" ? (index.data.runs ?? []).filter((r) => r.available.includes("rq3_conformal")) : [];
  const run = runs.some((r) => r.run === picked) ? picked : MAIN_RUN;
  const loaded = useLoaded((signal) => getCalibration(signal, run), `calibration/${run}`);
  const llm = loaded.status === "ready" && isRecord(loaded.data) && typeof loaded.data.llm === "string" ? loaded.data.llm : null;

  return (
    <div className="space-y-3">
      {runs.length > 1 ? (
        <Segmented
          legend="Generator whose calibration rows are shown"
          name="explainer-run"
          value={run}
          options={runs.map((r) => ({ value: r.run, label: runLabel(r, runs) }))}
          onChange={setPicked}
        />
      ) : null}
      {llm ? (
        <p className="num font-sans text-xs text-muted">
          Showing {generatorName(llm)} ({llm}), from results/{runPathPrefix(run)}rq3_conformal.json, and the thresholds RQ3
          certified for that generator. Each generator has its own calibration rows and thresholds.
        </p>
      ) : null}
      <ExplainerBody key={run} loaded={loaded} />
    </div>
  );
}

function ExplainerBody({ loaded }: { loaded: Loaded<unknown> }) {

  if (loaded.status === "loading") {
    return (
      <p className="font-sans text-sm text-muted" aria-busy="true">
        Looking for calibration predictions.
      </p>
    );
  }
  if (loaded.status === "missing") {
    return (
      <EmptyBlock title="The explainer has no data yet">
        <p>
          It runs on real calibration and held-out predictions, which exist only after the third experiment has run. Run{" "}
          <Code>primum exp rq3</Code>, then reload this page. Until then nothing is drawn here: an explainer with
          made-up numbers would defeat its purpose.
        </p>
      </EmptyBlock>
    );
  }
  if (loaded.status === "down") {
    return (
      <EmptyBlock title="The explainer is not loaded">
        <p>
          It runs on calibration predictions served by the API at <Code>/api/calibration</Code>, and the API is not
          answering. Start it with <Code>primum serve</Code>.
        </p>
      </EmptyBlock>
    );
  }
  if (loaded.status === "error") {
    return (
      <EmptyBlock title="The calibration data could not be read">
        <p>{loaded.message}</p>
      </EmptyBlock>
    );
  }
  const data = parseCalibration(loaded.data);
  if (!data) {
    return (
      <EmptyBlock title="The calibration data is in a layout this page does not recognise">
        <p>
          <Code>/api/calibration</Code> answered, but without per-question <Code>p_correct</Code> and{" "}
          <Code>correct</Code> values, so nothing is drawn.
        </p>
      </EmptyBlock>
    );
  }
  return <Explainer data={data} />;
}
