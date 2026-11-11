/** Height of `AccuracyByConditionChart` for `rows` bars, axis included: 32px per band (a bar of about 16px) plus the tick row. */
export const accuracyChartHeight = (rows: number): number => Math.max(108, rows * 32 + 40);
