/*
 * Heights of the RQ1 figures, in a module of their own so a section can size each chart's
 * skeleton without importing the chart (and its libraries) before it loads.
 */

/** `DotRows`: 34px per row, 8px above, 30px for the tick row. */
export const dotRowsHeight = (rows: number): number => Math.max(96, rows * 34 + 38);

/** `PerQueryStrips`: 40px per retriever plus the tick row. */
export const stripsHeight = (rows: number): number => Math.max(120, rows * 40 + 40);

export const LOSS_CHART_HEIGHT = 260;
