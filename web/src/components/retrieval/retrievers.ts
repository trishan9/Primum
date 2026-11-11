import type { Hit, Passage } from "@/lib/types";

/** The comparison a first visit starts with: lexical, two dense encoders and their fusion. */
export const DEFAULT_SELECTION = ["bm25", "dense-base", "dense-ft", "hybrid-ft"];
export const K_OPTIONS = [5, 10, 20];
export const CORPORA = [
  { id: "kb", label: "Knowledge base" },
  { id: "pubmedqa", label: "PubMedQA" },
  { id: "nfcorpus", label: "NFCorpus" },
] as const;

export const EXAMPLES = [
  "statin muscle pain",
  "does metformin cause vitamin B12 deficiency",
  "il-6 inhibitors in covid-19",
];

/** What each retriever name means, from the system specification. Descriptions, not results. */
export const RETRIEVER_NOTES: Record<string, string> = {
  bm25: "Okapi BM25 over words. Matches the terms you typed, weighting rare ones more.",
  "dense-base": "A general-purpose sentence encoder. Matches meaning, with no biomedical training.",
  "dense-biomed": "An off-the-shelf encoder trained on PubMed text. The domain baseline.",
  "dense-ft":
    "This project's encoder: the base encoder fine-tuned contrastively on biomedical question and abstract pairs.",
  "hybrid-base": "BM25 and the base encoder, merged by reciprocal rank fusion.",
  "hybrid-ft": "BM25 and the fine-tuned encoder, merged by reciprocal rank fusion. The default for live answers.",
  "hybrid-ft+rerank": "The fused shortlist, reread and reordered by a cross-encoder.",
  "bm25+concepts": "BM25 over the query plus the synonyms of the medical concepts recognised in it.",
};

export type Family = "Lexical" | "Dense" | "Fused" | "Reranked" | "Expanded";

/** The kind of search a retriever does, read from its name. */
export function familyOf(name: string): Family {
  if (name.includes("rerank")) return "Reranked";
  if (name.includes("+concepts")) return "Expanded";
  if (name.startsWith("hybrid")) return "Fused";
  if (name.startsWith("dense")) return "Dense";
  return "Lexical";
}

/** What a retriever's score is, so a BM25 22.2 is not read against a cosine 0.66. */
export function scoreKind(name: string): string {
  const family = familyOf(name);
  if (family === "Reranked") return "rerank";
  if (family === "Fused") return "RRF";
  if (family === "Dense") return "cosine";
  return "BM25";
}

export const needsConcepts = (name: string) => name.includes("+concepts");

export function rankOf(hits: Hit[] | undefined, id: string): number | null {
  const index = hits?.findIndex((h) => h.passage.passage_id === id) ?? -1;
  return index === -1 ? null : index + 1;
}

export interface Movement {
  /** Short visible form: "new", "=", "+3", "−2". */
  text: string;
  /** The same in words for a screen reader. */
  spoken: string;
  direction: "new" | "same" | "up" | "down";
}

/** How a passage moved relative to the ranking on its left. */
export function movement(previous: number | null, current: number): Movement {
  if (previous === null) return { text: "new", spoken: "not in the ranking to the left", direction: "new" };
  const delta = previous - current;
  if (delta === 0) return { text: "=", spoken: "same rank as on the left", direction: "same" };
  return delta > 0
    ? { text: `+${delta}`, spoken: `up ${delta} from the left`, direction: "up" }
    : { text: `−${-delta}`, spoken: `down ${-delta} from the left`, direction: "down" };
}

export interface Agreement {
  /** Passages that appear in at least one ranking. */
  distinct: number;
  /** Passages that appear in every ranking. */
  shared: number;
  /** How many rankings put the most common first passage first. */
  sameTop: number;
  rankings: number;
}

/** Counts of where the rankings agree, computed from the response; nothing here is typed in. */
export function agreement(retrievers: string[], results: Record<string, Hit[]>, k: number): Agreement {
  const lists = retrievers.map((name) => (results[name] ?? []).slice(0, k).map((h) => h.passage.passage_id));
  const all = new Set(lists.flat());
  const shared = [...all].filter((id) => lists.every((list) => list.includes(id))).length;
  const tops = new Map<string, number>();
  for (const list of lists) if (list[0]) tops.set(list[0], (tops.get(list[0]) ?? 0) + 1);
  const sameTop = Math.max(0, ...tops.values());
  return { distinct: all.size, shared, sameTop, rankings: lists.length };
}

/** A line to name a passage by: its title, or the start of its text when it has none (most PubMed abstracts here). */
export function passageHeading(passage: Passage): string {
  if (passage.title) return passage.title;
  const text = passage.text.trim().replace(/\s+/g, " ");
  if (!text) return docLabel(passage.source, passage.doc_id);
  return text.length > 96 ? `${text.slice(0, 96).trimEnd()}…` : text;
}

/** "PMID 123" for a PubMed article, the bare id otherwise. */
export function docLabel(source: string, docId: string): string {
  return source === "pubmed" && /^\d+$/.test(docId) ? `PMID ${docId}` : docId;
}
