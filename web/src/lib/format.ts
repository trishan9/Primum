/** Number formatting. One rule per kind of quantity, used everywhere that quantity appears. */

const MISSING = "–";

/** Probabilities and thresholds: two decimals, always with the leading zero. */
export function fmtProb(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  return value.toFixed(digits);
}

/** A proportion as a percentage. */
export function fmtPct(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  return `${(value * 100).toFixed(digits)}%`;
}

/** Stage latency: milliseconds under a second, seconds above. */
export function fmtMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return MISSING;
  if (ms < 1) return "<1 ms";
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
}

/** Retrieval scores live on different scales, so the precision follows the magnitude. */
export function fmtScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  const abs = Math.abs(value);
  if (abs >= 100) return value.toFixed(0);
  if (abs >= 10) return value.toFixed(1);
  if (abs >= 0.1) return value.toFixed(2);
  return value.toFixed(3);
}

/** A metric from a results file: three decimals for values in [0, 1], fewer above. */
export function fmtMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  const abs = Math.abs(value);
  if (abs >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (abs >= 10) return value.toFixed(1);
  return value.toFixed(3);
}

export function fmtInt(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  return Math.round(value).toLocaleString("en-US");
}

export function fmtUsd(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  // Single calls cost fractions of a cent; two decimals would print them all as $0.00.
  if (value > 0 && value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

/** The risk level as it is written in the method: alpha = 0.10. */
export function fmtAlpha(alpha: number): string {
  return Number.isFinite(alpha) ? alpha.toFixed(2) : MISSING;
}

/** p-values: exact to three decimals, with a floor so tiny values do not print as zero. */
export function fmtP(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return MISSING;
  if (value < 0.001) return "<0.001";
  return value.toFixed(3);
}

export const MISSING_VALUE = MISSING;
