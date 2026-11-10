import { fmtAlpha, fmtProb } from "@/lib/format";

interface ConfidenceScaleProps {
  /** Calibrated probability that the answer is correct; null while it is being computed. */
  p: number | null;
  /** Certified threshold; null when no threshold certifies this alpha. */
  threshold: number | null;
  alpha: number;
  answered: boolean | null;
}

const TICKS = Array.from({ length: 11 }, (_, i) => i / 10);

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/** Keeps a label that hangs off a point on the scale from running past either end. */
function anchor(x: number): string {
  if (x < 0.45) return "translate-x-0";
  if (x > 0.55) return "-translate-x-full";
  return "-translate-x-1/2";
}

/**
 * The instrument's one reading: where the calibrated confidence sits against the certified
 * threshold. Everything left of the threshold is hatched, and the system withholds any answer
 * whose pointer lands there. The hatch, the positions and the printed values carry the
 * meaning; colour only reinforces it.
 */
export function ConfidenceScale({ p, threshold, alpha, answered }: ConfidenceScaleProps) {
  const hasGate = typeof threshold === "number";
  const gate = hasGate ? clamp01(threshold) : null;
  const noneCertified = threshold === null;
  const pointer = p === null ? null : clamp01(p);

  const summary = [
    pointer === null ? "Confidence not computed yet." : `Calibrated probability of a correct answer: ${fmtProb(pointer)}.`,
    hasGate
      ? `Certified threshold for alpha ${fmtAlpha(alpha)}: ${fmtProb(gate)}.`
      : `No threshold certifies alpha ${fmtAlpha(alpha)}.`,
    answered === null ? "" : answered ? "The answer is released." : "The answer is withheld.",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div role="img" aria-label={summary} className="num select-none">
      <div className="relative h-9">
        {pointer !== null ? (
          <div
            className={`absolute bottom-3 whitespace-nowrap text-[0.8125rem] leading-none transition-[left] duration-700 ease-out ${anchor(pointer)}`}
            style={{ left: `${pointer * 100}%` }}
          >
            <span className="text-muted">P(correct) </span>
            <span className="text-[0.9375rem] font-semibold text-ink">{fmtProb(pointer)}</span>
          </div>
        ) : (
          <div className="absolute bottom-3 left-0 text-[0.8125rem] leading-none text-muted">P(correct) pending</div>
        )}
      </div>

      <div className="relative h-7 rounded-[3px] border border-control bg-surface">
        <div
          className="absolute inset-y-0 left-0 overflow-hidden rounded-l-[2px] bg-raised"
          style={{ width: `${(noneCertified ? 1 : (gate ?? 0)) * 100}%` }}
        >
          <div className="hatch absolute inset-0" />
        </div>
        {gate !== null ? (
          <div className="absolute -bottom-1.5 -top-1.5 w-0.5 -translate-x-1/2 bg-ink" style={{ left: `${gate * 100}%` }} />
        ) : null}
        {pointer !== null ? (
          <div
            className="absolute -bottom-px -top-2 -translate-x-1/2 transition-[left] duration-700 ease-out"
            style={{ left: `${pointer * 100}%` }}
          >
            <svg aria-hidden="true" width="13" height="8" viewBox="0 0 13 8" className="absolute -top-px left-1/2 -translate-x-1/2">
              <path d="M0 0h13L6.5 8Z" fill={answered === false ? "var(--text)" : "var(--accent)"} />
            </svg>
            <div className={`h-full w-[3px] ${answered === false ? "bg-ink" : "bg-accent"}`} />
          </div>
        ) : null}
      </div>

      <div className="relative h-6 text-xs text-muted">
        {TICKS.map((tick, i) => (
          <div key={tick} className="absolute top-0" style={{ left: `${tick * 100}%` }}>
            <div className={`w-px bg-control ${i % 2 === 0 ? "h-1.5" : "h-1"}`} />
            {i % 2 === 0 ? (
              <span
                className={`absolute top-2 leading-none ${i === 0 ? "translate-x-0" : i === 10 ? "-translate-x-full" : "-translate-x-1/2"}`}
              >
                {tick === 0 ? "0" : tick === 1 ? "1" : tick.toFixed(1)}
              </span>
            ) : null}
          </div>
        ))}
      </div>

      <div className="relative mt-1 h-5 text-[0.8125rem] leading-none">
        {gate !== null ? (
          <div className={`absolute top-0.5 whitespace-nowrap ${anchor(gate)}`} style={{ left: `${gate * 100}%` }}>
            <span className="text-muted">threshold </span>
            <span className="font-semibold text-ink">{fmtProb(gate)}</span>
            <span className="text-muted">
              <span className="hidden sm:inline">, certified for α = {fmtAlpha(alpha)}</span>
              <span className="sm:hidden"> (α = {fmtAlpha(alpha)})</span>
            </span>
          </div>
        ) : (
          <div className="absolute left-0 top-0.5 text-muted">
            No threshold certifies α = {fmtAlpha(alpha)}: every answer is withheld at this level.
          </div>
        )}
      </div>
    </div>
  );
}
