/*
 * Heights of the RQ3 figures, in a module of their own so a section can size each chart's
 * skeleton without importing the chart.
 */

export const ALPHA_TOP = 16;
export const ALPHA_AXIS = 44;
export const ALPHA_ROW = 18;

/** An α plot with a `plot`-pixel drawing area and `rows` text rows under its axis. */
export const alphaPlotHeight = (plot: number, rows = 0): number => ALPHA_TOP + plot + ALPHA_AXIS + (rows > 0 ? rows * ALPHA_ROW + 8 : 0);

/** The violation figure: two small multiples on one α axis, each with a 20px caption above it. */
export const VIOLATION_CERTIFIED_PLOT = 150;
export const VIOLATION_PLUGIN_PLOT = 100;
export const VIOLATION_HEIGHT = alphaPlotHeight(VIOLATION_CERTIFIED_PLOT) + alphaPlotHeight(VIOLATION_PLUGIN_PLOT) + 2 * 20 + 12;
export const COVERAGE_HEIGHT = alphaPlotHeight(220, 2);
export const COMPARE_HEIGHT = alphaPlotHeight(170);
export const SETS_HEIGHT = alphaPlotHeight(190);
