/**
 * The statistics behind the threshold explainer, implemented in the browser so that
 * dragging the risk level recomputes the certificate on the real held-out data instead
 * of interpolating between four precomputed points.
 *
 * This follows `primum.uncertainty.conformal.certify_threshold` (SPEC 3.5): Learn-then-Test
 * with fixed-sequence testing and an exact binomial tail p-value. The Python module is the
 * reference implementation; the thresholds it certified are drawn alongside for comparison.
 */

/** Lanczos approximation of ln Γ(x), accurate to about 1e-14 for x > 0. */
export function logGamma(x: number): number {
  const g = 7;
  const c = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ] as const;
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - logGamma(1 - x);
  const z = x - 1;
  let sum: number = c[0];
  for (let i = 1; i < g + 2; i++) sum += (c[i] as number) / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * P(Bin(n, alpha) <= k): the p-value for H0 "the true risk exceeds alpha" after seeing
 * k errors among n selected answers. Small values mean k errors would be surprisingly few
 * if the risk really were above alpha, so H0 is rejected and the threshold is certified.
 */
export function binomialPValue(k: number, n: number, alpha: number): number {
  if (n <= 0) return 1;
  if (k >= n) return 1;
  if (k < 0) return 0;
  if (alpha <= 0) return 1;
  if (alpha >= 1) return 0;
  const logA = Math.log(alpha);
  const logB = Math.log1p(-alpha);
  const logNFact = logGamma(n + 1);
  let total = 0;
  for (let i = 0; i <= k; i++) {
    total += Math.exp(logNFact - logGamma(i + 1) - logGamma(n - i + 1) + i * logA + (n - i) * logB);
  }
  return Math.min(1, total);
}

export interface Selection {
  threshold: number;
  nSelected: number;
  nErrors: number;
  /** share of questions answered */
  coverage: number;
  /** error rate among the answered; NaN when nothing is answered */
  risk: number;
}

/** What happens if the system answers exactly when confidence >= threshold. */
export function selectAt(conf: readonly number[], correct: readonly boolean[], threshold: number): Selection {
  let nSelected = 0;
  let nErrors = 0;
  for (let i = 0; i < conf.length; i++) {
    if ((conf[i] as number) >= threshold) {
      nSelected++;
      if (!correct[i]) nErrors++;
    }
  }
  return {
    threshold,
    nSelected,
    nErrors,
    coverage: conf.length ? nSelected / conf.length : 0,
    risk: nSelected ? nErrors / nSelected : Number.NaN,
  };
}

export interface Certificate {
  alpha: number;
  delta: number;
  /** null when even the most conservative candidate fails its test */
  threshold: number | null;
  selection: Selection | null;
  /** p-value of the last test that passed */
  pValue: number | null;
  /** every test performed, in the order it was performed */
  steps: { threshold: number; nSelected: number; nErrors: number; pValue: number; passed: boolean }[];
}

/** Default minimum selection size, used only when the server does not send `min_selected`. */
export const DEFAULT_MIN_SELECTED = 20;

/** Tolerance the server applies to the two ceilings, so float noise cannot shift a rank. */
const EPS = 1e-9;

/**
 * Candidate thresholds, strictest first, exactly as the server builds them: every distinct
 * confidence when there are at most `nGrid` of them, otherwise the confidences at ranks
 * ceil(j * n / nGrid), j = 1..nGrid (evenly spaced empirical quantiles ending at the minimum).
 * Thresholds are data points, so each one selects exactly the questions at or above it.
 */
function candidateThresholds(confDesc: readonly number[], nGrid: number): number[] {
  const distinct = confDesc.filter((value, i) => i === 0 || value !== confDesc[i - 1]);
  if (distinct.length <= nGrid) return distinct;
  const n = confDesc.length;
  const out: number[] = [];
  for (let j = 1; j <= nGrid; j++) {
    const value = confDesc[Math.ceil((j * n) / nGrid - EPS) - 1] as number;
    if (out[out.length - 1] !== value) out.push(value);
  }
  return out;
}

/**
 * Fixed-sequence testing, ported from `certify_threshold` so it reproduces the server's
 * thresholds when given the same `nGrid` and `minSelected` (`/api/calibration` sends both).
 * The candidates depend on the confidences only, never on the labels. Starting from the
 * strictest, each is tested at level delta and the walk stops at the first failure. Because
 * the order is fixed in advance and testing stops at the first non-rejection, the chance that
 * the returned threshold has true risk above alpha is at most delta, with no correction for
 * the number of candidates.
 *
 * A candidate selecting N questions with none wrong has p-value (1 - alpha)^N, so while
 * (1 - alpha)^N > delta no labels could make it pass. Testing it would end the walk by
 * construction, so, like the server, the walk starts at the first candidate that selects at
 * least ceil(ln delta / ln(1 - alpha)) questions (and at least `minSelected`).
 */
export function certifyThreshold(
  conf: readonly number[],
  correct: readonly boolean[],
  options: { alpha: number; delta: number; nGrid?: number; minSelected?: number },
): Certificate {
  const { alpha, delta, nGrid = 100, minSelected = DEFAULT_MIN_SELECTED } = options;
  const confDesc = [...conf].sort((a, b) => b - a);
  const minRejectable = Math.ceil(Math.log(delta) / Math.log1p(-alpha) - EPS);
  const minTestable = Math.max(minSelected, minRejectable);

  const steps: Certificate["steps"] = [];
  let best: { selection: Selection; pValue: number } | null = null;
  for (const threshold of candidateThresholds(confDesc, nGrid)) {
    const selection = selectAt(conf, correct, threshold);
    if (selection.nSelected < minTestable) continue;
    const pValue = binomialPValue(selection.nErrors, selection.nSelected, alpha);
    const passed = pValue <= delta;
    steps.push({ threshold, nSelected: selection.nSelected, nErrors: selection.nErrors, pValue, passed });
    if (!passed) break;
    best = { selection, pValue };
  }

  return {
    alpha,
    delta,
    threshold: best ? best.selection.threshold : null,
    selection: best ? best.selection : null,
    pValue: best ? best.pValue : null,
    steps,
  };
}

/** Selective risk and coverage at each of `points` evenly spaced thresholds in [0, 1]. */
export function thresholdSweep(conf: readonly number[], correct: readonly boolean[], points = 101): Selection[] {
  return Array.from({ length: points }, (_, i) => selectAt(conf, correct, i / (points - 1)));
}

/* ---------- /api/calibration ---------- */

export interface CalibrationRows {
  conf: number[];
  correct: boolean[];
}

export interface CalibrationData extends CalibrationRows {
  /**
   * Rows never used to fit, calibrate or certify (`heldout` in the payload), where a threshold
   * certified on the calibration rows is evaluated out of sample; null when the server sent none.
   */
  heldout: CalibrationRows | null;
  /** thresholds the server certified, keyed by alpha; null means none certifies that alpha */
  certified: { alpha: number; threshold: number | null }[];
  delta: number | null;
  /** the server's certification settings, when RQ3 recorded them; needed to reproduce its thresholds */
  nGrid: number | null;
  minSelected: number | null;
}

function positiveInt(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value > 0 ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toBool(value: unknown): boolean | null {
  if (typeof value === "boolean") return value;
  if (value === 1 || value === 0) return value === 1;
  return null;
}

function readCertified(raw: unknown): CalibrationData["certified"] {
  const out: CalibrationData["certified"] = [];
  const push = (alphaRaw: unknown, value: unknown) => {
    const alpha = typeof alphaRaw === "number" ? alphaRaw : Number.parseFloat(String(alphaRaw));
    if (!Number.isFinite(alpha) || alpha <= 0 || alpha >= 1) return;
    const inner = isRecord(value) ? value.threshold : value;
    if (inner === null || inner === undefined) out.push({ alpha, threshold: null });
    else if (typeof inner === "number" && Number.isFinite(inner)) out.push({ alpha, threshold: inner });
  };
  if (Array.isArray(raw)) {
    for (const item of raw) if (isRecord(item)) push(item.alpha, item);
  } else if (isRecord(raw)) {
    for (const [key, value] of Object.entries(raw)) push(key, value);
  }
  return out.sort((a, b) => a.alpha - b.alpha);
}

/** Per-item `{p_correct, correct}` in the column or the row layout; null when neither is present. */
function readRows(raw: Record<string, unknown>): CalibrationRows | null {
  const conf: number[] = [];
  const correct: boolean[] = [];

  const rows = [raw.items, raw.records, raw.data].find(Array.isArray) as unknown[] | undefined;
  if (Array.isArray(raw.p_correct) && Array.isArray(raw.correct) && raw.p_correct.length === raw.correct.length) {
    for (let i = 0; i < raw.p_correct.length; i++) {
      const p = raw.p_correct[i];
      const c = toBool(raw.correct[i]);
      if (typeof p === "number" && Number.isFinite(p) && c !== null) {
        conf.push(p);
        correct.push(c);
      }
    }
  } else if (rows) {
    for (const row of rows) {
      if (!isRecord(row)) continue;
      const c = toBool(row.correct);
      if (typeof row.p_correct === "number" && Number.isFinite(row.p_correct) && c !== null) {
        conf.push(row.p_correct);
        correct.push(c);
      }
    }
  }
  return conf.length === 0 ? null : { conf, correct };
}

/**
 * Reads the calibration payload. SPEC 3.8 fixes its content (per-item `{p_correct, correct}`
 * plus certified thresholds) but not its layout, so both the column layout
 * `{p_correct: [...], correct: [...]}` and the row layout `{items: [{p_correct, correct}]}`
 * are accepted, for the calibration rows and for an optional `heldout` block alike. Returns
 * null when the calibration rows are absent, and the page says so.
 */
export function parseCalibration(raw: unknown): CalibrationData | null {
  if (!isRecord(raw)) return null;
  const rows = readRows(raw);
  if (!rows) return null;
  const heldout = isRecord(raw.heldout) ? readRows(raw.heldout) : null;

  const certifiedRaw = raw.thresholds ?? raw.certified_thresholds ?? raw.certified ?? raw.guarantees;
  return {
    ...rows,
    heldout,
    certified: readCertified(certifiedRaw),
    delta: typeof raw.delta === "number" ? raw.delta : null,
    nGrid: positiveInt(raw.n_grid),
    minSelected: positiveInt(raw.min_selected),
  };
}
