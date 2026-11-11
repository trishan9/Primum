import type { ExperimentName } from "@/lib/api";

export type QuestionId = "rq1" | "rq2" | "rq3";

export interface Question {
  id: QuestionId;
  label: string;
  /** A two or three word name for the question, used as its tab and heading. */
  topic: string;
  /** One word for the tab on a phone. */
  short: string;
  file: ExperimentName;
  question: string;
  command: string;
  /** What the section will hold once the experiment has run. Describes the analysis, never its outcome. */
  willShow: string;
  /** True when the results do not involve the generator, so they are read from the main run whichever is picked. */
  generatorIndependent?: boolean;
}

export const QUESTIONS: readonly Question[] = [
  {
    id: "rq1",
    label: "RQ1",
    topic: "Retrieval",
    short: "Retrieval",
    file: "rq1_retrieval",
    question:
      "Does a domain-adapted, contrastively fine-tuned encoder, fused with lexical BM25, retrieve biomedical evidence better than either alone?",
    command: "primum exp rq1",
    generatorIndependent: true,
    willShow:
      "nDCG@10, Recall@100 and MRR@10 on the NFCorpus test queries, and Recall@1, 5 and 20 with MRR@10 for finding the gold abstract of each PubMedQA question, for every retriever, with 95% bootstrap intervals and paired permutation tests against BM25 and the base dense encoder.",
  },
  {
    id: "rq2",
    label: "RQ2",
    topic: "Confidence and calibration",
    short: "Confidence",
    file: "rq2_confidence",
    question:
      "Which signals predict that a generated answer is correct, and is a learned combination better calibrated than any single signal?",
    command: "primum exp rq2",
    willShow:
      "AUROC, AURC, ECE and Brier score for each single signal and for the logistic and MLP confidence models under each calibration method, over repeated splits; a feature ablation; a reliability diagram; and risk-coverage curves.",
  },
  {
    id: "rq3",
    label: "RQ3",
    topic: "Certified abstention",
    short: "Abstention",
    file: "rq3_conformal",
    question:
      "Does the certified threshold keep selective risk at or below α on unseen questions, and how much coverage does that cost?",
    command: "primum exp rq3",
    willShow:
      "For each certified α: how often test risk exceeded α across repeated splits, with a Clopper-Pearson interval, set against the tolerated rate δ; mean coverage; and the same for an uncorrected plug-in threshold and for single-signal confidences.",
  },
];

export const isQuestionId = (value: string): value is QuestionId => QUESTIONS.some((q) => q.id === value);
