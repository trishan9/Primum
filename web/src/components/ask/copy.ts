import type { Source, StageName, VerdictLabel } from "@/lib/types";

export const STAGE_COPY: Record<StageName, { label: string; idle: string }> = {
  triage: { label: "Safety triage", idle: "Rule check for emergencies" },
  concepts: { label: "Concepts", idle: "Link terms to MeSH" },
  retrieve: { label: "Retrieval", idle: "Lexical and dense search, fused" },
  generate: { label: "Generation", idle: "Draft from the passages only" },
  verify: { label: "Verification", idle: "Check each claim against its source" },
  confidence: { label: "Confidence", idle: "Calibrate, compare with the threshold" },
};

export const VERDICT_COPY: Record<VerdictLabel, { label: string; sentence: string }> = {
  supported: { label: "Supported", sentence: "Supporting sentence" },
  unsupported: { label: "Unsupported", sentence: "Closest sentence found" },
  contradicted: { label: "Contradicted", sentence: "Contradicting sentence" },
};

export const SOURCE_COPY: Record<Source, { label: string; open: string }> = {
  pubmed: { label: "PubMed", open: "Open on PubMed" },
  medlineplus: { label: "MedlinePlus", open: "Open on MedlinePlus" },
  nfcorpus: { label: "NFCorpus", open: "Open source" },
  pdf: { label: "Local PDF", open: "Open document" },
};

/** Plain names for the signals the confidence model reads (SPEC 3.5). */
export const FEATURE_COPY: Record<string, string> = {
  ret_top: "Top passage score",
  ret_mean: "Mean passage score",
  ret_margin: "Margin between the first and second passage",
  ret_agree: "Agreement between lexical and dense search",
  nli_mean_entail: "Mean entailment across claims",
  nli_min_entail: "Entailment of the weakest claim",
  nli_max_contra: "Strongest contradiction found",
  frac_supported: "Share of claims supported",
  n_claims: "Number of claims",
  verbal_conf: "Confidence the model stated",
  sc_agreement: "Agreement across repeated samples",
  sc_entropy: "Entropy across repeated samples",
  verbal_margin: "Margin between the two likeliest decisions",
};

export const EXAMPLE_QUESTIONS: readonly { question: string; note?: string }[] = [
  { question: "What are the warning signs of dehydration in older adults?" },
  { question: "Does regular low-dose aspirin lower the risk of colorectal cancer?" },
  { question: "Is it safe to take ibuprofen with high blood pressure?" },
  {
    question: "I have crushing chest pain spreading to my left arm and I am sweating. What should I take?",
    note: "shows emergency triage",
  },
];
