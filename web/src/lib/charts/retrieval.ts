/**
 * Retrieval lab derivations, from one `/api/retrieve` response: where each passage ranks under
 * each retriever (a bump chart), how much the retrievers' top-k lists overlap, and how each
 * retriever's scores fall away down its own ranking. Pure; every plotted value comes from here.
 *
 * A rank is the passage's position in the retriever's list cut to the top `k` (1-based), the same
 * number the rank columns print, so the chart and the columns can never disagree.
 */

import type { Hit } from "../types.ts";

type Results = Readonly<Record<string, readonly Hit[] | undefined>>;

/** The lists cut to the top k, for the retrievers that answered, in the order given. */
function lists(results: Results, retrievers: readonly string[], k: number): { retriever: string; hits: readonly Hit[] }[] {
  return retrievers.filter((r) => Array.isArray(results[r])).map((retriever) => ({ retriever, hits: (results[retriever] ?? []).slice(0, k) }));
}

export interface BumpSeries {
  passageId: string;
  title: string;
  /** one entry per retriever, in column order; null where that retriever did not return it in its top k */
  ranks: { retriever: string; rank: number | null }[];
  /** best (smallest) rank across retrievers */
  bestRank: number;
  /** how many retrievers returned it */
  foundBy: number;
}

/**
 * Every passage any retriever returned in its top k, with its rank under each. Ordered by best
 * rank, then by how many retrievers found it, so the passages the retrievers agree on come first.
 */
export function rankBump(results: Results, retrievers: readonly string[] = Object.keys(results), k = Infinity): BumpSeries[] {
  const columns = lists(results, retrievers, k);
  const names = columns.map((c) => c.retriever);
  const byId = new Map<string, BumpSeries>();
  for (const { retriever, hits } of columns) {
    hits.forEach((hit, index) => {
      const rank = index + 1;
      const id = hit.passage.passage_id;
      let series = byId.get(id);
      if (!series) {
        series = { passageId: id, title: hit.passage.title, ranks: names.map((r) => ({ retriever: r, rank: null })), bestRank: Infinity, foundBy: 0 };
        byId.set(id, series);
      }
      const slot = series.ranks.find((r) => r.retriever === retriever);
      if (slot && slot.rank === null) {
        slot.rank = rank;
        series.foundBy += 1;
      }
      series.bestRank = Math.min(series.bestRank, rank);
    });
  }
  return [...byId.values()].sort((a, b) => a.bestRank - b.bestRank || b.foundBy - a.foundBy || a.passageId.localeCompare(b.passageId));
}

export interface BumpSegment {
  passageId: string;
  from: { retriever: string; rank: number };
  to: { retriever: string; rank: number };
}

/** The lines of the bump chart: one per passage per pair of neighbouring columns that both returned it. */
export function bumpSegments(series: readonly BumpSeries[]): BumpSegment[] {
  return series.flatMap((s) =>
    s.ranks.slice(1).flatMap((to, i) => {
      const from = s.ranks[i];
      return from && from.rank !== null && to.rank !== null
        ? [{ passageId: s.passageId, from: { retriever: from.retriever, rank: from.rank }, to: { retriever: to.retriever, rank: to.rank } }]
        : [];
    }),
  );
}

export interface Disagreement {
  passageId: string;
  best: { retriever: string; rank: number };
  worst: { retriever: string; rank: number };
  spread: number;
}

/** The passage, among those at least two retrievers returned, whose ranks are furthest apart. Null if none moves. */
export function widestDisagreement(series: readonly BumpSeries[]): Disagreement | null {
  let found: Disagreement | null = null;
  for (const s of series) {
    const placed = s.ranks.filter((r): r is { retriever: string; rank: number } => r.rank !== null);
    if (placed.length < 2) continue;
    const best = placed.reduce((a, b) => (b.rank < a.rank ? b : a));
    const worst = placed.reduce((a, b) => (b.rank > a.rank ? b : a));
    const spread = worst.rank - best.rank;
    if (spread > 0 && (!found || spread > found.spread)) found = { passageId: s.passageId, best, worst, spread };
  }
  return found;
}

/** One sentence on the bump chart, from the series. */
export function bumpTakeaway(series: readonly BumpSeries[], columns: number, k: number): string {
  if (series.length === 0) return "No retriever returned a passage.";
  const everywhere = series.filter((s) => s.foundBy === columns).length;
  const once = series.filter((s) => s.foundBy === 1).length;
  const plural = (n: number, one: string, many: string) => `${n} ${n === 1 ? one : many}`;
  const head = `${plural(series.length, "distinct passage", "distinct passages")} across ${columns} top-${k} lists: ${everywhere} in every list, ${once} in only one.`;
  const wide = widestDisagreement(series);
  if (!wide) return head;
  return `${head} Widest disagreement: a passage ${wide.best.retriever} ranks ${wide.best.rank} and ${wide.worst.retriever} ranks ${wide.worst.rank}.`;
}

export interface Overlap {
  a: string;
  b: string;
  shared: number;
  /** |A ∩ B| / |A ∪ B| */
  jaccard: number;
}

/** Pairwise overlap of the retrievers' top-k sets (both orders, diagonal included). */
export function overlapMatrix(results: Results, retrievers: readonly string[] = Object.keys(results), k = Infinity): Overlap[] {
  const sets = new Map(lists(results, retrievers, k).map(({ retriever, hits }) => [retriever, new Set(hits.map((h) => h.passage.passage_id))]));
  const names = [...sets.keys()];
  return names.flatMap((a) =>
    names.map((b) => {
      const A = sets.get(a) as Set<string>;
      const B = sets.get(b) as Set<string>;
      const shared = [...A].filter((id) => B.has(id)).length;
      const union = new Set([...A, ...B]).size;
      return { a, b, shared, jaccard: union > 0 ? shared / union : 0 };
    }),
  );
}

/** Each unordered pair once (a before b in column order), the diagonal left out. */
export function overlapPairs(matrix: readonly Overlap[], retrievers: readonly string[]): Overlap[] {
  const order = new Map(retrievers.map((r, i) => [r, i]));
  return matrix.filter((o) => (order.get(o.a) ?? -1) < (order.get(o.b) ?? -1) && order.has(o.a) && order.has(o.b));
}

/** One sentence on the overlap matrix: the most and the least alike pair. */
export function overlapTakeaway(pairs: readonly Overlap[], k: number): string {
  if (pairs.length === 0) return "Choose at least two retrievers to compare their lists.";
  const byShared = [...pairs].sort((x, y) => y.shared - x.shared || y.jaccard - x.jaccard);
  const most = byShared[0] as Overlap;
  // Ties go to the pair that comes first in column order, both ways.
  const least = [...pairs].sort((x, y) => x.shared - y.shared || x.jaccard - y.jaccard)[0] as Overlap;
  if (pairs.length === 1 || most.shared === least.shared) {
    return pairs.length === 1
      ? `${most.a} and ${most.b} share ${most.shared} of their top ${k}.`
      : `Every pair shares ${most.shared} of its top ${k}.`;
  }
  return `Most alike: ${most.a} and ${most.b}, sharing ${most.shared} of their top ${k}. Least alike: ${least.a} and ${least.b}, sharing ${least.shared}.`;
}

export interface ScorePoint {
  rank: number;
  score: number;
  passageId: string;
}

export interface ScoreProfile {
  retriever: string;
  points: ScorePoint[];
  min: number;
  max: number;
  /** the biggest fall between neighbouring ranks: after which rank, by how much, and as a share of the list's score range */
  steepest: { afterRank: number; drop: number; share: number } | null;
}

/**
 * Each retriever's scores down its own top k. Scores are on each retriever's own scale (BM25,
 * cosine, reciprocal-rank fusion, cross-encoder logits), so a profile is read against itself only.
 * Non-finite scores are left out.
 */
export function scoreProfiles(results: Results, retrievers: readonly string[] = Object.keys(results), k = Infinity): ScoreProfile[] {
  return lists(results, retrievers, k).map(({ retriever, hits }) => {
    const points = hits.flatMap((hit, i) => (Number.isFinite(hit.score) ? [{ rank: i + 1, score: hit.score, passageId: hit.passage.passage_id }] : []));
    const scores = points.map((p) => p.score);
    const min = scores.length ? Math.min(...scores) : 0;
    const max = scores.length ? Math.max(...scores) : 0;
    let steepest: ScoreProfile["steepest"] = null;
    for (let i = 1; i < points.length; i++) {
      const prev = points[i - 1] as ScorePoint;
      const cur = points[i] as ScorePoint;
      if (cur.rank !== prev.rank + 1) continue;
      const drop = prev.score - cur.score;
      if (drop > 0 && (!steepest || drop > steepest.drop)) steepest = { afterRank: prev.rank, drop, share: max > min ? drop / (max - min) : 0 };
    }
    return { retriever, points, min, max, steepest };
  });
}

/**
 * One sentence on the score profiles: where a list has one decisive step and where its scores are
 * near-ties. Shares of each list's own range, since the scales differ.
 */
export function scoreTakeaway(profiles: readonly ScoreProfile[]): string {
  const stepped = profiles.filter((p) => p.steepest !== null && p.points.length >= 3);
  if (stepped.length === 0) return "Too few scored passages to describe how the scores fall.";
  const pct = (share: number) => `${Math.round(share * 100)}%`;
  const sharpest = stepped.reduce((a, b) => ((b.steepest?.share ?? 0) > (a.steepest?.share ?? 0) ? b : a));
  const flattest = stepped.reduce((a, b) => ((b.steepest?.share ?? 1) < (a.steepest?.share ?? 1) ? b : a));
  const s = sharpest.steepest as NonNullable<ScoreProfile["steepest"]>;
  const head = `${sharpest.retriever} has the sharpest step: ${pct(s.share)} of its score range falls between ranks ${s.afterRank} and ${s.afterRank + 1}.`;
  if (flattest === sharpest) return head;
  const f = flattest.steepest as NonNullable<ScoreProfile["steepest"]>;
  return `${head} ${flattest.retriever} falls most evenly (largest step ${pct(f.share)} of its range).`;
}
