/** The few pieces of chart arithmetic every figure here shares. */

export type Scale = (value: number) => number;

export function linear(domain: readonly [number, number], range: readonly [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value) => r0 + ((value - d0) / span) * (r1 - r0);
}

/** Round tick values covering [min, max], about `count` of them, on a 1/2/5 step. */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [];
  if (min === max) return [min];
  const raw = (max - min) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const step = (fraction >= 5 ? 10 : fraction >= 2 ? 5 : fraction >= 1 ? 2 : 1) * power;
  const start = Math.ceil(min / step - 1e-9) * step;
  const ticks: number[] = [];
  for (let value = start; value <= max + step * 1e-9; value += step) ticks.push(Number(value.toPrecision(12)));
  return ticks;
}

/** A domain padded outward to round numbers, so no mark sits on the frame. */
export function niceDomain(min: number, max: number, count = 5): [number, number] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) {
    const pad = Math.abs(min) * 0.1 || 0.1;
    return [min - pad, max + pad];
  }
  const raw = (max - min) / Math.max(1, count);
  const power = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / power;
  const step = (fraction >= 5 ? 10 : fraction >= 2 ? 5 : fraction >= 1 ? 2 : 1) * power;
  return [Math.floor(min / step + 1e-9) * step, Math.ceil(max / step - 1e-9) * step];
}

export function extent(values: readonly number[]): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  return min === Infinity ? [0, 1] : [min, max];
}

/** Tick labels with just enough decimals to tell neighbouring ticks apart. */
export function tickFormat(ticks: readonly number[]): (value: number) => string {
  const step = ticks.length > 1 ? Math.abs((ticks[1] as number) - (ticks[0] as number)) : 1;
  const decimals = step >= 1 ? 0 : Math.min(4, Math.ceil(-Math.log10(step) - 1e-9));
  return (value) => (Math.abs(value) >= 1000 ? value.toLocaleString("en-US") : value.toFixed(decimals));
}

/** The first four series get the validated categorical hues, in fixed order; any further series are grey. */
export function seriesColor(index: number): string {
  return index < 4 ? `var(--series-${index + 1})` : "var(--series-rest)";
}
