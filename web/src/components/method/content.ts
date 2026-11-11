/**
 * The words of the Method page that are data rather than prose: the sections it lists, the six
 * pipeline stages, and one analogy per idea with the point where it stops holding. Nothing here
 * is a result; every figure on the page comes from the API at runtime.
 */

export const SECTIONS = [
  { id: "thesis", label: "Thesis" },
  { id: "pipeline", label: "Pipeline" },
  { id: "retrieval", label: "Retrieval" },
  { id: "generation", label: "Generation" },
  { id: "verification", label: "Verification" },
  { id: "calibration", label: "Calibration" },
  { id: "threshold", label: "Certified threshold" },
  { id: "triage", label: "Safety triage" },
  { id: "limits", label: "Limits" },
  { id: "data", label: "Data" },
] as const;

export type SectionId = (typeof SECTIONS)[number]["id"];

export type StageGlyph = "shield" | "concepts" | "search" | "pen" | "check" | "gauge";

export const STAGES: readonly { name: string; glyph: StageGlyph; does: string; gives: string }[] = [
  { name: "Safety triage", glyph: "shield", does: "A fixed rule list looks for red-flag presentations.", gives: "routine, urgent, or emergency" },
  { name: "Concepts", glyph: "concepts", does: "Terms are matched to a dictionary built from MedlinePlus and MeSH.", gives: "linked concepts and synonyms" },
  { name: "Retrieval", glyph: "search", does: "Lexical and dense search run separately, are fused, then reranked.", gives: "ranked passages with their scores" },
  { name: "Generation", glyph: "pen", does: "The model writes an answer bound to those passages.", gives: "text with numbered citations" },
  { name: "Verification", glyph: "check", does: "Each claim is tested against the sentences it cites.", gives: "supported, unsupported, or contradicted" },
  { name: "Confidence", glyph: "gauge", does: "The signals are combined, calibrated, and compared with the threshold.", gives: "release or withhold" },
];

export const ANALOGIES = {
  retrieval: {
    text: "Two judges score the same contest, one out of ten and one out of a hundred. You cannot add their marks. You can ask each for a ranking and reward the entries both placed near the top.",
    limit: "the judges here are not independent. Both read the same passage text, so when they agree it is weaker evidence than two independent opinions would be.",
  },
  generation: {
    text: "An open-book exam where the only book allowed is the handful of pages on the desk, and every sentence of the answer must carry the page number it came from.",
    limit: "a model can ignore the instruction. It may state something from memory and attach a citation anyway, which is exactly why the next stage does not take the citations on trust.",
  },
  verification: {
    text: "A fact-checker who does not ask whether a sentence sounds right. They open the cited source, find the relevant line, and check that it says what the writer claims it says.",
    limit: "the checker is itself a model with its own error rate, and it checks whether a claim follows from the passage, not whether the passage is right. A claim faithfully supported by a flawed study is still marked supported.",
  },
  calibration: {
    text: "A weather forecaster is calibrated if it rains on about seven of every ten days for which they said 70%. Being calibrated is not being right every day. It is knowing how often you are right.",
    limit: "a forecaster is calibrated against the weather of the place they forecast for. This model is calibrated on benchmark research questions, and a question of a different kind may not follow the same curve.",
  },
  threshold: {
    text: "An exam board wants at most one in ten of the students who pass to be unprepared. It tries pass marks on a practice exam. A mark that gets exactly one in ten on the practice paper is not safe, since the real cohort will differ, so the board sets the mark a little higher: high enough that it would be surprised, at a stated level, to be wrong.",
    limit: "an exam board can keep testing students until it is sure. Primum has a fixed set of calibration questions, and the guarantee is about the procedure across such sets. It does not promise anything about one particular answer.",
  },
  triage: {
    text: "The laminated card at a clinic reception desk that says which complaints go straight to a nurse. It is not a diagnosis, and nobody pretends it is.",
    limit: "a nurse looks at the patient. This list reads one sentence of text, so it misses emergencies described in unusual words and flags some questions that are only about emergencies.",
  },
} as const;

export type AnalogyName = keyof typeof ANALOGIES;

export const SOURCES = [
  {
    name: "PubMedQA",
    href: "https://huggingface.co/datasets/qiaojin/PubMedQA",
    text: "the expert-labelled subset of yes, no or maybe research questions, each with its PubMed abstract. Used for the confidence and abstention experiments. The abstract’s conclusion is never indexed, because it would leak the label.",
  },
  {
    name: "NFCorpus",
    href: "https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/",
    text: "in its BEIR form: a medical retrieval benchmark with graded relevance judgements. Used to compare retrievers.",
  },
  {
    name: "MedlinePlus health topics",
    href: "https://medlineplus.gov/xml.html",
    text: "from the US National Library of Medicine: the knowledge base behind live answers, and the source of the concept dictionary.",
  },
] as const;
