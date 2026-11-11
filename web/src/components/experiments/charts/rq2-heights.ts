/*
 * Heights of the RQ2 figures, in a module of their own so a section can size each chart's
 * skeleton without importing the chart (and nivo with it).
 */

export const FOREST_TOP = 22;
export const FOREST_AXIS = 42;

/** The extra row under the axis label for the two end words ("← lower, better" / "worse →"). */
export const FOREST_ENDS = 16;

/** A forest plot of `rows` rows, axis included (and the end words, when it has them). */
export const forestHeight = (rows: number, rowHeight = 28, ends = false): number => FOREST_TOP + rows * rowHeight + FOREST_AXIS + (ends ? FOREST_ENDS : 0);

export const RANKING_ROW = 26;
export const rankingHeight = (rows: number): number => forestHeight(rows, RANKING_ROW);

export const BRIER_ROW = 34;
/** The Brier forest always draws its end words. */
export const brierHeight = (rows: number): number => forestHeight(rows, BRIER_ROW, true);

/** Horizontal bars, one per dropped group, plus the tick row. */
export const ablationHeight = (rows: number): number => Math.max(140, rows * 36 + 48);

export const RISK_COVERAGE_HEIGHT = 300;

/** The reliability square plus its count strip. */
export const RELIABILITY_PLOT = 260;
export const RELIABILITY_STRIP = 64;
export const RELIABILITY_HEIGHT = RELIABILITY_PLOT + RELIABILITY_STRIP;
