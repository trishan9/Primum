import type { Health } from "./types.ts";

/** Something the running server lacks that would make a question fail or lose its guarantee. */
export type Gap = "api_key" | "index" | "encoder" | "bundle";

/**
 * Whether a retriever's dense stage uses the fine-tuned encoder (`dense-ft`, `hybrid-ft` and
 * their `+concepts` / `+rerank` variants). The server never falls back to another encoder, so
 * without the trained model such a retriever fails with `no_index`.
 */
export function needsFinetunedEncoder(retriever: string): boolean {
  return /^(dense|hybrid)-ft(\+|$)/.test(retriever);
}

/**
 * What `/api/health` says is missing for a question asked with `retriever` (the reader's
 * choice, or the server's default; null while neither is known).
 */
export function readinessGaps(health: Health, retriever: string | null): Gap[] {
  const gaps: Gap[] = [];
  if (health.api_key === false) gaps.push("api_key");
  const indexes = health.indexes;
  const noIndex =
    indexes === false ||
    (Array.isArray(indexes) && indexes.length === 0) ||
    (typeof indexes === "object" &&
      indexes !== null &&
      !Array.isArray(indexes) &&
      Object.values(indexes).every((v) => !v || (typeof v === "object" && Object.keys(v as object).length === 0)));
  if (noIndex) gaps.push("index");
  if (retriever !== null && needsFinetunedEncoder(retriever) && health.finetuned_encoder === false) gaps.push("encoder");
  if (health.deployment_bundle === false) gaps.push("bundle");
  return gaps;
}
