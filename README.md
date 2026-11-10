# Primum

*Primum non nocere*: first, do no harm.

Primum is a selective clinical question-answering system. It answers a biomedical question only when its calibrated confidence clears a threshold that was certified, with a distribution-free test on held-out calibration data, to keep the expected error rate among answered questions (selective risk) at or below a chosen level α with probability at least 1 − δ over the calibration draw. Otherwise it abstains and says why. The threshold is chosen with Learn-then-Test (fixed-sequence binomial testing on a held-out calibration set), so the guarantee needs no parametric model of the confidence distribution. It assumes new questions are drawn i.i.d. from the same distribution as the calibration questions and that selective risk does not fall as the threshold is lowered (monotone risk).

How much the certified rule lets the system answer differed markedly between the two generators tried. With the same pipeline and procedure, at α = 0.20 the mean coverage over 200 random splits (a split that certifies nothing counts as 0) was 15.8% with Claude Haiku 5.5 and 54.4% with Claude Sonnet 5.5, which is both more accurate and ranks its answers somewhat better by confidence; the violation rate stayed below δ for both (two separate runs, compared descriptively; see [below](#does-a-stronger-generator-change-the-picture)).

> Analogy: a junior doctor who is allowed to answer alone only on the kinds of cases where an audit has shown they are wrong less than one time in ten, and who escalates everything else.

Every result number in this file is read from a `results/**/*.json` file written by code (the exceptions: the planned encoder batch size is the CLI default in `src/primum/cli.py`, and the total ledger spend is the output of `uv run primum spend`). The full generated tables are in [docs/RESULTS.md](docs/RESULTS.md); the build contract is [docs/SPEC.md](docs/SPEC.md).

## Research questions

| | Question | What is compared |
|---|---|---|
| RQ1 | Does a contrastively fine-tuned encoder, fused with lexical BM25, retrieve biomedical evidence better than either alone? | BM25 written from scratch; dense base encoder; dense biomedical encoder; own InfoNCE fine-tuning with BM25-mined hard negatives; Reciprocal Rank Fusion hybrid; cross-encoder rerank; MedlinePlus concept query expansion |
| RQ2 | Which signals predict that an answer is correct, and is a learned combination better calibrated than any single signal? | Retrieval, NLI-verification, stated (verbal) confidence and self-consistency signals; logistic and MLP combinations; raw, temperature, Platt and isotonic calibration; AUROC, AURC, E-AURC, ECE, Brier |
| RQ3 | Does the certified threshold keep selective risk ≤ α on unseen questions, and what does it cost in coverage? | Certified (binomial and Hoeffding-Bentkus) vs naive plug-in threshold; violation rate against δ; coverage; LAC and APS conformal prediction sets |

## Headline findings

Setup: 1000 PubMedQA expert-labelled questions, generator `claude-haiku-5-5` at low effort, retriever `hybrid-ft` (top 5); the second generator, `claude-sonnet-5-5`, has [its own subsection](#does-a-stronger-generator-change-the-picture). Brackets are 95% bootstrap CIs over questions or queries, Clopper-Pearson CIs, or, for RQ2/RQ3 metrics, the central 95% range of per-split values over 200 repeated random splits (split variability, not a confidence interval), as stated per table. Permutation tests use 10000 permutations, so the smallest attainable p is 1/(10000+1); such values are written as "p < 0.001 (floor)".

### Evidence helps; retrieval leaves a small but significant gap to the gold abstract

Source: `results/pubmedqa_runs.json`. Figure: [pubmedqa_accuracy.svg](results/figures/pubmedqa_accuracy.svg).

| Condition | Accuracy | Macro-F1 | Gold abstract in evidence |
|---|---|---|---|
| closed book | 0.528 [0.499, 0.560] | 0.408 [0.377, 0.439] | – |
| RAG (`hybrid-ft`, top 5) | 0.697 [0.668, 0.725] | 0.569 [0.540, 0.601] | 94.6% |
| oracle (gold abstract only) | 0.748 [0.721, 0.773] | 0.593 [0.565, 0.622] | 100.0% |

Retrieval adds 16.9 accuracy points over closed book (0.169 [0.135, 0.202], paired permutation p < 0.001 (floor)). Handing the model the gold abstract adds 5.1 points more (0.051 [0.029, 0.073], p < 0.001 (floor)): a real but small gap. The oracle is not a true ceiling either, because the indexed passage omits the abstract's conclusion. Most of the remaining error is in the yes/no/maybe judgment, not in finding the evidence.

### RQ1: retrieval

Source: `results/rq1_retrieval.json`. Figures: [rq1_nfcorpus.svg](results/figures/rq1_nfcorpus.svg), [rq1_pubmedqa.svg](results/figures/rq1_pubmedqa.svg).

| System | NFCorpus nDCG@10 (323 test queries) | PubMedQA gold-abstract Recall@1 (1000 queries) |
|---|---|---|
| `bm25` (from scratch) | 0.319 [0.287, 0.352] | 0.772 [0.746, 0.798] |
| `dense-base` (all-MiniLM-L6-v2) | 0.317 [0.282, 0.353] | 0.749 [0.722, 0.775] |
| `dense-biomed` (pubmedbert-base-embeddings) | 0.266 [0.233, 0.299] | 0.827 [0.804, 0.850] |
| `dense-ft` (own fine-tuning) | 0.305 [0.271, 0.337] | 0.783 [0.758, 0.808] |
| `hybrid-base` (RRF) | 0.354 [0.318, 0.388] | 0.804 [0.779, 0.830] |
| `hybrid-ft` (RRF) | 0.357 [0.324, 0.393] | 0.817 [0.793, 0.841] |
| `hybrid-ft+rerank` | 0.361 [0.326, 0.396] | 0.859 [0.838, 0.880] |
| `bm25+concepts` | 0.305 [0.269, 0.340] | 0.671 [0.641, 0.700] |

What the paired tests (Holm-adjusted permutation p) say:

- **Fine-tuning helped on PubMedQA, not on NFCorpus.** Against `dense-base`, `dense-ft` gains Recall@1 0.034 [0.011, 0.057] (p = 0.0116) on PubMedQA, but on NFCorpus its nDCG@10 difference is −0.012 [−0.029, 0.005] (p = 0.5315), i.e. no detectable gain and a slightly lower point estimate. Training pairs are drawn from PubMedQA-artificial (PQA-A) questions, whose PMIDs are disjoint from PQA-L, plus NFCorpus training queries, so the PubMedQA gain is in-domain format adaptation rather than transfer.
- **Fine-tuning did not reach the biomedical baseline on PubMedQA.** `dense-biomed` has the higher Recall@1 (0.827 vs 0.783, by point estimate; this pair was not tested), though it is the worst dense model on NFCorpus (nDCG@10 0.266).
- **Fusion is the reliable win.** Every hybrid beats BM25 on both benchmarks (Holm p ≤ 0.0015). `hybrid-ft+rerank` has the highest point estimate on both; against BM25 it gains nDCG@10 0.041 [0.028, 0.057] on NFCorpus and Recall@1 0.087 [0.064, 0.110] on PubMedQA (both Holm-adjusted p = 0.0007). The rerank step itself was not tested against `hybrid-ft`, and on NFCorpus the gap is small (0.361 vs 0.357). Neither `hybrid-ft` vs `dense-ft` nor `hybrid-ft` vs `hybrid-base` was tested, so the role of fine-tuning inside fusion is untested.
- **Concept expansion hurt on PubMedQA.** There, `bm25+concepts` loses Recall@1 −0.101 [−0.124, −0.080] against BM25 (Holm-adjusted p = 0.0007); on NFCorpus its nDCG@10 change is −0.014 [−0.027, −0.0004] (Holm p = 0.1038, not significant after adjustment). Appending synonyms dilutes the precise wording that matches an abstract.

Encoder training (`results/encoder_training.json`, [encoder_loss.svg](results/figures/encoder_loss.svg)): 56389 pairs from 42064 queries, 1763 steps on `mps` in 2060.45 s, batch size 32, BM25 hard negatives on. On a held-out sample of the training distribution (500 queries, pool of 651 passages), MRR@10 went from 0.971 to 0.982. That is a sanity check that training ran, not a benchmark.

### RQ2: which signals predict correctness

Source: `results/rq2_confidence.json` (1000 RAG answers, 697 correct; 200 stratified 40/30/30 splits; Platt calibration). Figures: [rq2_estimators.svg](results/figures/rq2_estimators.svg), [rq2_reliability.svg](results/figures/rq2_reliability.svg), [rq2_risk_coverage.svg](results/figures/rq2_risk_coverage.svg). Brackets in the table are split ranges (central 95% of per-split values).

| Estimator | AUROC | AURC | ECE | Brier |
|---|---|---|---|---|
| stated verbal confidence | 0.735 [0.680, 0.781] | 0.167 [0.137, 0.198] | 0.050 [0.028, 0.084] | 0.181 [0.169, 0.196] |
| self-consistency agreement | 0.637 [0.593, 0.683] | 0.240 [0.218, 0.260] | 0.031 [0.009, 0.068] | 0.193 [0.181, 0.205] |
| NLI max contradiction | 0.578 [0.525, 0.632] | 0.244 [0.211, 0.277] | 0.029 [0.001, 0.080] | 0.209 [0.205, 0.215] |
| NLI mean entailment | 0.496 [0.417, 0.562] | 0.296 [0.251, 0.344] | 0.014 [0.000, 0.073] | 0.212 [0.211, 0.215] |
| fraction of claims supported | 0.491 [0.419, 0.553] | 0.305 [0.271, 0.346] | 0.013 [0.000, 0.062] | 0.212 [0.211, 0.214] |
| logistic, deploy features (deployed) | 0.733 [0.681, 0.780] | 0.169 [0.137, 0.206] | 0.049 [0.025, 0.082] | 0.182 [0.170, 0.197] |
| logistic, all features | 0.741 [0.688, 0.790] | 0.168 [0.136, 0.205] | 0.053 [0.027, 0.082] | 0.179 [0.166, 0.195] |

- **Stated verbal confidence is the best single signal** (highest mean AUROC).
- **NLI entailment features are at chance for correctness** (mean entailment AUROC 0.496 [0.417, 0.562], fraction of claims supported 0.491 [0.419, 0.553]); max contradiction is weakly informative (0.578 [0.525, 0.632]). Interpretation: verification measures whether the answer's claims are faithful to the retrieved evidence, not whether the final yes/no/maybe judgment is right. A claim can be perfectly supported by the abstract while the conclusion drawn from it is wrong.
- **The learned combination shows no improvement over stated confidence.** Against stated confidence, the all-features logistic model's Brier difference is −0.002 (bootstrap over questions [−0.007, 0.003], p = 0.4156) and its AUROC difference is 0.006 (split range [−0.029, 0.029]; higher in 70% of splits). The deployed `logistic:deploy` model is not better than stated confidence either: Brier difference 0.001 (bootstrap [−0.002, 0.005], p = 0.4070), AUROC difference −0.002 (split range [−0.034, 0.018]; higher in 46.5% of splits). It uses the deploy feature set because chat answers have no self-consistency samples (`FEATURES_DEPLOY` in docs/SPEC.md), not because it won here. Dropping the verbal group is the only ablation that clearly hurts (AUROC 0.670 [0.623, 0.718]).
- Low ECE for the weak signals is not a virtue: a calibrated but uninformative score predicts the base rate for everyone.

### RQ3: does the certified threshold hold?

Source: `results/rq3_conformal.json` (200 splits, n_cal = 300, δ = 0.1, at least 30 calibration points selected; confidence `logistic:deploy` + Platt, fitted on the training split only). Figure: [rq3_selective.svg](results/figures/rq3_selective.svg).

Violation rate is the share of repeats whose test selective risk exceeds α (Clopper-Pearson interval). A repeat that certifies no threshold answers nothing and counts as no violation. The repeats re-split one dataset, so the interval reflects Monte-Carlo error over splits, not sampling error of the dataset. Test risk is itself estimated on about 300 questions, so test noise near α inflates the measured violation rate, for both methods and especially for plug-in. Coverage is the mean share of test questions answered; "Certified: splits certifying" is the share of splits in which any threshold was certified (`certified_rate`).

| α | Certified: splits certifying | Certified: violation | Certified: coverage | Plug-in: violation | Plug-in: coverage |
|---|---|---|---|---|---|
| 0.05 | 0.0% | 0.0% [0.0%, 1.8%] | 0.000 | 60.0% [52.9%, 66.8%] | 0.069 |
| 0.10 | 2.5% | 1.5% [0.3%, 4.3%] | 0.003 | 61.0% [53.9%, 67.8%] | 0.207 |
| 0.15 | 19.0% | 4.5% [2.1%, 8.4%] | 0.035 | 53.5% [46.3%, 60.6%] | 0.488 |
| 0.20 | 41.0% | 3.5% [1.4%, 7.1%] | 0.158 | 51.5% [44.3%, 58.6%] | 0.681 |

- **Consistent with the guarantee on this dataset.** The certified violation rate is below δ = 10% at every α, including the upper end of each interval, but at α ≤ 0.10 this is mostly because the procedure certifies nothing (2.5% of splits at α = 0.10, none at α = 0.05). The guarantee is marginal over calibration draws, so it does not bound the violation rate among the splits that certify a threshold, which is high at α = 0.10: 3 of the 5 certifying splits (violation rate 0.015 and certified rate 0.025 of 200 repeats) exceeded α on test, and their mean test selective risk is 0.125. The naive plug-in threshold (smallest threshold whose calibration risk is ≤ α, no correction) violates in roughly half of the splits or more.
- **The price is coverage.** With 300 calibration rows and a model that is right 69.7% of the time overall, the certified method answers almost nothing at α ≤ 0.10 and on average 15.8% of questions at α = 0.20 (a threshold is certified in 41% of splits; coverage ranges from 0.000 to 0.647 across splits).
- **Stated confidence alone certifies more coverage than the deployed model.** With the same certified procedure, stated verbal confidence answers on average 32.0% of questions at α = 0.20 (certified in 72.5% of splits; violation 1.5% [0.3%, 4.3%]) against 15.8% for the deployed logistic model. This is reported, not used to swap the deployed model: choosing the model on these test splits would spend the same data twice.
- Conformal prediction sets over the stated yes/no/maybe probabilities reach their targets to within split noise (at α = 0.10: LAC coverage 0.909, mean set size 1.916; APS coverage 0.899 against a 0.90 target, mean set size 2.060), but a set of about two labels out of three is a weak answer.

Deployed thresholds (trained on 400 answers, certified on 300): α = 0.05 none certified; α = 0.10 0.875; α = 0.15 0.849; α = 0.20 0.736. The deployed split is one draw. At α = 0.10 a threshold was certified in only 2.5% of the 200 resampled splits, so the deployed α = 0.10 threshold is an unusually favourable draw; the certificate is a statement about the procedure, not about this realised threshold.

### Does a stronger generator change the picture?

Sources: `results/sonnet/` (`pubmedqa_runs.json`, `rq2_confidence.json`, `rq3_conformal.json`) and `results/sonnet_sc300/` (`pubmedqa_runs.json`, `rq2_confidence.json`). Figure: [generators_rq3.svg](results/figures/generators_rq3.svg). Full tables: the "Second generator" section of [docs/RESULTS.md](docs/RESULTS.md).

**Design.** The same pipeline (retriever `hybrid-ft`, top 5, low effort) was re-run with `claude-sonnet-5-5` as generator, `rag` condition only. To fit the budget it used 1 sample per question on all 1000 questions, plus 5 samples per question on the first 300 questions in dataset order (a self-consistency subset, not a random sample). This design was fixed before any Sonnet result was seen. The sample count does not touch the deployed model: every `FEATURES_DEPLOY` feature (retrieval, NLI, stated confidence) is computed from sample 0 alone (`primum.experiments.run_pubmedqa._answer`), and extra samples feed only the self-consistency features in `FEATURES_FULL`. RQ3 uses `logistic:deploy` for both generators, so its inputs are defined identically. Sample 0 of the subset run is the same cached response as in the 1-sample run (the cache key is model + prompt + sample id); the repeat samples used Anthropic prompt caching, which changes billing only. There is no closed-book or oracle run for Sonnet. Every comparison with Haiku below is descriptive: each interval describes one run, and the two RQ3 analyses use separate sets of repeated splits, so they are not a paired test.

- **Accuracy rises.** `rag` accuracy 0.797 [0.773, 0.821] for Sonnet (n = 996; 4 failed generations excluded: all four returned stated probabilities of all zeros and were rejected as invalid output; RQ2/RQ3 and the guarantee cover valid generations only) against 0.697 [0.668, 0.725] for Haiku (n = 1000, 0 failed); macro-F1 0.677 [0.643, 0.708] against 0.569 [0.540, 0.601] (bootstrap CIs over questions).
- **Stated confidence is again the best single signal; no learned combination improved on it (differences not significant).** RQ2 on 996 Sonnet answers: stated verbal confidence AUROC 0.775 [0.718, 0.830], `logistic:deploy` 0.764 [0.700, 0.820] (split ranges). Against stated confidence, `logistic:deploy` has Brier difference 0.001 (bootstrap [−0.001, 0.003], p = 0.1821) and AUROC difference −0.011 (split range [−0.042, 0.012]; higher in 21.0% of splits). With one sample, the self-consistency features were not measured, so RQ2 drops them (`meta.unavailable_features`: `sc_agreement`, `sc_entropy`) and skips the `full` feature set.
- **NLI entailment features are again near chance.** Mean entailment AUROC 0.484 [0.426, 0.538], fraction of claims supported 0.485 [0.425, 0.531]. Max contradiction, weakly informative for Haiku, is at most weakly informative here: 0.559 [0.440, 0.635], a split range that includes 0.5.
- **Consistent with the guarantee, and coverage rises.** RQ3 on Sonnet (n_cal = 299, δ = 0.1, 200 splits, `logistic:deploy` + Platt). Certified violation rate 1.5% [0.3%, 4.3%], 2.0% [0.5%, 5.0%], 0.5% [0.0%, 2.8%] and 0.0% [0.0%, 1.8%] at α = 0.05, 0.10, 0.15, 0.20, all below δ = 10%. As for Haiku, at small α this is mostly because little is certified: the guarantee is marginal over calibration draws, so it does not bound the violation rate among the splits that do certify, and there violations are common (α = 0.05: 3 of 4 certifying splits exceeded α on test, mean test selective risk 0.070; α = 0.10: 4 of 14, mean 0.084; counts from `violation_rate` × 200 and `certified_rate` × 200 under `selective.certified` in `results/sonnet/rq3_conformal.json`). Mean coverage is 0.003, 0.016, 0.179 and 0.544 (Haiku: 0.000, 0.003, 0.035, 0.158). A threshold is certified in 2.0%, 7.0%, 30.0% and 61.5% of splits (Haiku: 0.0%, 2.5%, 19.0%, 41.0%). Stated confidence alone, under the same procedure, answers 0.836 of questions at α = 0.20 (certified in 97.0% of splits; violation 0.0%; reported, not used to choose the deployed model, for the same reason as for Haiku).
- **The naive plug-in threshold still violates at α = 0.05 to 0.15.** Violation rate 63.0% [55.9%, 69.7%], 50.5% [43.4%, 57.6%] and 42.5% [35.6%, 49.7%] at α = 0.05, 0.10, 0.15 (coverage 0.190, 0.659, 0.848). At α = 0.20 it answers almost everything (coverage 0.985) and violates in 13.0% [8.7%, 18.5%] of splits: the point estimate is above δ, but the interval includes it.
- **Self-consistency on the first 300 questions.** 298 answers (2 failed: in a 5-sample run a question fails if any of its samples is invalid, so one of the two, qid 17444776, was answered in the 1-sample run), accuracy 0.735 [0.685, 0.785]. Sonnet is less accurate on this first-300 subset than on all 996 answers, which include it; it is a dataset-order slice, not a random sample, so its numbers need not carry over. Self-consistency agreement has mean AUROC 0.605 [0.535, 0.685] against 0.732 [0.628, 0.828] for stated confidence (means over the same splits set side by side, not a paired test). No learned model beats stated confidence here: the all-features logistic model, which includes self-consistency, has AUROC 0.693 [0.589, 0.793] and Brier difference 0.007 (bootstrap [−0.001, 0.013], p = 0.0791), and `logistic:deploy` is worse in Brier (0.007 [0.001, 0.014], p = 0.0330). The first-sample decision matches the 1-sample run on all 298 questions answered in both.
- **Per-generator deployment bundle.** RQ3 saves the bundle under the generator that produced the records: `artifacts/deployment/claude-sonnet-5-5/bundle.json` (trained on 399 answers, certified on 299, 298 held out; α = 0.05 and 0.10 none certified, α = 0.15 0.732, α = 0.20 0.581) beside the Haiku one. The live app defaults to Haiku; `PRIMUM_LLM_MODEL=claude-sonnet-5-5` switches both the generator and the bundle, because a threshold certified on one model's answers says nothing about another's. The certificate covers only the generator and retriever it was certified with, and in the app it is reported as applying only to that bundle's held-out PubMedQA questions (298 for Sonnet) in benchmark mode; for chat questions the threshold is an operating point, and a generator without its own bundle is answered unscored. With the Sonnet bundle nothing is certified at α = 0.05 or 0.10, so the app abstains on every question at those levels.

## Architecture

```
 PubMedQA (PQA-L + PQA-U) ─┐                         ┌─ BM25 (scipy sparse, from scratch)
 NFCorpus (BEIR)          ─┼─ primum data build ──▶  ├─ dense: base / biomed / fine-tuned (InfoNCE)
 MedlinePlus topics       ─┘                         └─ concept dictionary (MedlinePlus)
                                    │
 question ─▶ triage (rules) ─▶ retrieve: BM25 + dense (fine-tuned) ─▶ RRF ─▶ top 5
            (cross-encoder rerank, hybrid-ft+rerank, is evaluated in RQ1 but is not the default retriever)
                                    │
            LLM: Claude Haiku 5.5 (default) or Sonnet 5.5, low effort  (sqlite response cache, hard budget ledger)
                                    │
            claim extraction ─▶ NLI verification per claim (DeBERTa-v3-base NLI, MNLI/FEVER/ANLI)
                                    │
            features ─▶ confidence model (logistic, Platt) ─▶ calibrated P(correct)
                                    │
            certified threshold for chosen α (Learn-then-Test) ─▶ ANSWER  or  ABSTAIN
                                    │
            FastAPI, Server-Sent Events  ─▶  Next.js web: Ask / Retrieval lab / Experiments / Method
```

Package layout (`src/primum/`):

| Module | Role |
|---|---|
| `data/` | download, build corpora, contrastive pairs, chunking, MedlinePlus concepts |
| `retrieval/` | BM25, dense index, RRF fusion, reranker, concept expansion, contrastive training loop, retrieval metrics |
| `llm/` | Anthropic client, prompts, sqlite cache and budget ledger, price table |
| `generation/` | benchmark answers, streamed chat answers, claim extraction, evidence formatting |
| `verification/` | NLI model wrapper and claim verification |
| `uncertainty/` | features, PyTorch confidence models, calibration (PAV from scratch), conformal and Learn-then-Test, selective metrics, statistics, deployment bundle |
| `safety/` | rule-based red-flag triage (a safety rail, not a research contribution) |
| `pipeline/` | orchestrator that emits staged events; trace store |
| `api/` | FastAPI app (`/api/ask` streams SSE) |
| `experiments/` | one script per results file, plus `report.py` that writes figures and `docs/RESULTS.md` |

The web app is in `web/`; see [web/README.md](web/README.md).

## Data and leakage guards

From the data manifests in `results/rq1_retrieval.json`:

- **PubMedQA PQA-L**: 1000 expert-labelled questions (yes 552, no 338, maybe 110).
- **PubMedQA PQA-U** abstracts as distractors: 61249, giving a retrieval corpus of 62249 passages.
- **NFCorpus** (BEIR): 3633 documents; 323 test queries used for evaluation.
- **MedlinePlus** health topics: 3183 passages and 1017 concepts in the chat knowledge base (65432 passages in total with the PubMedQA passages).

Guards:

- A PubMedQA passage is the abstract body without its conclusion. `long_answer` is never indexed, because it states the answer.
- Contrastive training pairs exclude every PQA-L PMID and every PQA-L question; NFCorpus test queries are never trained on. `src/primum/data/pairs.py` checks for leaked PMIDs after building.
- In RQ3 and the deployment bundle, the confidence model and its calibrator (cross-fitted) see only the training split; in RQ2 the calibrator is fitted on the calibration split. Test rows never touch either. The threshold is certified only on the untouched calibration split; evaluation uses the test split.
- A local copy of the Gale Encyclopedia of Medicine (PDF) exists only on the author's machine. It is copyrighted, git-ignored (`*.pdf`), and was not included in the knowledge base used here (`include_pdf: false` in the manifest).

## Reproducibility

```bash
uv sync
uv run primum data build all
uv run primum data pairs
uv run primum train encoder --batch-size 32 --hard-negatives
uv run primum index build nfcorpus -r bm25 -r dense-base -r dense-biomed -r dense-ft
uv run primum index build pubmedqa -r bm25 -r dense-base -r dense-biomed -r dense-ft
uv run primum index build kb -r bm25 -r dense-ft
uv run primum exp pubmedqa     # calls the Anthropic API; needs ANTHROPIC_API_KEY in .env
uv run primum exp rq1
uv run primum exp rq2
uv run primum exp rq3          # also saves the deployment bundle
uv run primum exp report       # figures and docs/RESULTS.md from results/*.json
uv run primum serve            # API on :8000
pnpm -C web install && pnpm -C web dev   # web on :3000
```

Second generator (Claude Sonnet 5.5). The benchmark runs set the generator and a raised hard budget through the environment; RQ2 and RQ3 read the records from the run directory, and RQ3 saves the bundle under the generator recorded there (`artifacts/deployment/claude-sonnet-5-5/`). The `primum exp` CLI has no flags for the generator, conditions, samples or results directory, so these runs call `run()` directly. `conditions` and `self_consistency_samples` are recorded in each file's `meta`; `limit=300` shows only as `meta.n`; `meta.config.results_dir` shows the default because the directory is an argument. RQ2/RQ3 read the generator from the records (`meta.models.llm`); they were run with the same environment, so `meta.config.llm_model` agrees. The benchmark runs' `created` is a fixed label passed to `run()`; RQ2/RQ3 were given the wall-clock time:

```bash
# all 1000 questions, rag only, 1 sample -> results/sonnet/
PRIMUM_LLM_MODEL=claude-sonnet-5-5 PRIMUM_BUDGET_USD=28 uv run python -c "from pathlib import Path; from primum.experiments import run_pubmedqa; run_pubmedqa.run(created='2026-11-10T17:00:00+00:00', conditions=['rag'], samples=1, results_dir=Path('results/sonnet'))"
# first 300 questions in dataset order, rag only, 5 samples -> results/sonnet_sc300/
PRIMUM_LLM_MODEL=claude-sonnet-5-5 PRIMUM_BUDGET_USD=28 uv run python -c "from pathlib import Path; from primum.experiments import run_pubmedqa; run_pubmedqa.run(created='2026-11-10T17:00:00+00:00', conditions=['rag'], samples=5, limit=300, results_dir=Path('results/sonnet_sc300'))"
# RQ2 on both, RQ3 on the full run (no API calls)
PRIMUM_LLM_MODEL=claude-sonnet-5-5 uv run python -c "from pathlib import Path; from primum.experiments import rq2_confidence; rq2_confidence.run(created='2026-11-10T17:41:20+00:00', results_dir=Path('results/sonnet'))"
PRIMUM_LLM_MODEL=claude-sonnet-5-5 uv run python -c "from pathlib import Path; from primum.experiments import rq2_confidence; rq2_confidence.run(created='2026-11-10T17:41:20+00:00', results_dir=Path('results/sonnet_sc300'))"
PRIMUM_LLM_MODEL=claude-sonnet-5-5 uv run python -c "from pathlib import Path; from primum.experiments import rq3_conformal; rq3_conformal.run(created='2026-11-10T17:41:20+00:00', results_dir=Path('results/sonnet'))"
uv run primum exp report
PRIMUM_LLM_MODEL=claude-sonnet-5-5 uv run primum serve   # live app with Sonnet and its bundle
```

- **Cost.** Haiku PubMedQA run spend $3.2885 (`meta.run_spend_usd` in `results/pubmedqa_runs.json`), 140 calls served from the cache (`meta.cached_calls`), against a $15.00 hard budget (`meta.config.budget_usd`). Sonnet runs: $10.7186 for the 1-sample run (`results/sonnet/pubmedqa_runs.json`) and $6.9184 for the 5-sample subset (`results/sonnet_sc300/pubmedqa_runs.json`), against `PRIMUM_BUDGET_USD=28`, a hard cap on cumulative ledger spend over all runs and tags (not per run). `uv run primum spend` reports total $21.3493 over all tags (including chat use) and prints the budget of the current shell ($15.00 by default). Re-running is free while the sqlite cache is present.
- **Seeds.** One seed, 13, is used throughout (`meta.seeds` in each results file); each split repeat derives its own generator from it. Self-consistency uses 5 fixed sample ids.
- **Deviation from plan.** The encoder was planned with batch size 64 (the CLI default). With hard negatives, 64 ran out of MPS memory on a 16 GB machine, so it was trained with batch size 32 (`hyperparameters.batch_size` in `results/encoder_training.json`). Fewer in-batch negatives make the contrastive task easier, which may be part of why fine-tuning gave small gains.
- Tests: `uv run pytest -q` (no network, no models, no API key), including simulation tests that the certified threshold's violation frequency stays within δ on synthetic data with known risk.

## Limitations

- **Two generators, unevenly covered.** The full design (closed book, RAG, oracle; 5 samples per question) is for Claude Haiku 5.5 only. Claude Sonnet 5.5 was run in the `rag` condition only, so there is no closed-book or oracle comparison for it, and with 1 sample per question its self-consistency features exist only for the first 300 questions in dataset order, which are not a random sample (Sonnet is less accurate on them than on all 1000). The two generators' RQ3 analyses use separate repeated splits and are compared descriptively, not with a paired test. Both used low effort; more reasoning effort was not tried.
- **Small calibration set.** For Haiku, n = 1000 with 300 calibration rows limits how much coverage can be certified; at α = 0.05 the binomial test never passed in 200 splits (it needs enough selected points with almost no errors).
- **Scope of the guarantee.** It holds for new questions drawn i.i.d. from the same distribution as the PubMedQA calibration questions, answered by the same pipeline, under the monotone-risk condition. It does not hold for the free-form consumer chat mode, where the threshold is an operating point, not a guarantee.
- **Generic verifier.** The NLI model is a general-domain DeBERTa-v3-base NLI model (MNLI/FEVER/ANLI), `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, not a clinical one.
- **No clinician evaluation.** Correctness is agreement with PubMedQA labels; no clinician has judged the answers or the abstentions.
- **Label source.** PubMedQA labels are derived from abstract conclusions, so "correct" means agreeing with one study's abstract, not with clinical consensus.

## Explain it to a professor

**Why abstain at all, rather than just be more accurate?**
Accuracy is an average over every question; a clinician cares about the error rate on the answers actually given. Abstention lets the system trade how often it speaks for how often it is wrong when it does. *Analogy: a referee who only blows the whistle when sure, and is judged on the whistles blown.*

**What does Learn-then-Test guarantee?**
For a fixed α and δ, it walks from the strictest candidate threshold (quantiles of the calibration confidences, starting where at least 30 points are selected) downwards, testing H: risk > α at each with an exact binomial p-value at full level δ, stops at the first threshold it cannot reject, and certifies the last one rejected. Then, over the random draw of the calibration set, P(selective risk ≤ α) ≥ 1 − δ, with no assumption on the score distribution beyond i.i.d. calibration and test questions and the monotone-risk condition (see `primum.uncertainty.conformal`). *Analogy: a smoke alarm whose sensitivity is set on test fires it was never tuned on, with a safety margin so that it fails its promise at most one time in ten.*

**What is the difference between α and δ?**
α is the error rate you tolerate among answered questions; δ is the chance that the calibration set was unlucky and the promise about α does not hold. *Analogy: α is "at most one bad batch in ten"; δ is the chance the quality inspector was fooled by the sample he checked.*

**Why does the plug-in threshold fail?**
It picks the threshold whose calibration error is just under α, so it is tuned to the noise in 300 rows; on fresh data it overshoots about half the time. *Analogy: setting a high-jump bar exactly at your best practice jump and promising never to knock it.*

**Why might a more accurate generator get so much more coverage?**
The certificate needs, among the answers above the threshold, so few errors that "risk > α" can be rejected on 300 calibration rows. With Sonnet right 79.7% of the time overall, and its confidence ranking right from wrong somewhat better (AUROC 0.764 vs 0.733 for the deployed model), even a loose cut of confident answers can pass at α = 0.20; with Haiku right 69.7% of the time, only a thin, very confident slice can, and often none does. The guarantee is the same; what it costs depends on how many clean answers there are to keep. *Analogy: an inspector who passes a batch only if a sample shows almost no defects; a factory with fewer defects overall gets far more of its batches passed under the same rule.* With two generators this is an explanation consistent with the data, not a tested effect of accuracy.

**What is Reciprocal Rank Fusion?**
Each retriever ranks the documents; a document's fused score is the sum of 1 / (60 + its rank) over retrievers. Only ranks are combined, so BM25 and cosine scores never need to share a scale. *Analogy: two judges rank the same contestants; you add up placings, not their incompatible score sheets.*

**What does the InfoNCE fine-tuning do?**
Symmetric InfoNCE (temperature 0.05): each question must pick its own passage out of every passage and BM25-mined hard negative in the batch, and each passage must pick its question; known positives of the same query are masked out of the denominator. *Analogy: picking the right suspect out of a line-up where one decoy was chosen to look like him.*

**What does calibration mean, and how is it measured?**
A score is calibrated when, among answers given 0.7, about 70% are right. ECE is the gap between stated and observed accuracy in each of 10 confidence bins, averaged with weights proportional to bin size; Brier is the mean squared error of the probability. Temperature, Platt and isotonic scaling are post-hoc maps from raw score to probability. *Analogy: a weather forecaster whose "70% chance of rain" is followed by rain on 70% of those days.*

**What are AUROC and AURC?**
AUROC is the probability that a random correct answer gets a higher score than a random wrong one (0.5 is chance). AURC is the area under the risk-coverage curve: answer the most confident questions first, track the error rate as you answer more, and average it; lower is better. *Analogy: AUROC asks whether the queue is ordered well; AURC asks how many mistakes you make on average as you work through it from the front.*

**Why are the entailment features near chance?**
NLI checks whether each sentence of the answer is supported by the evidence. The model can quote the abstract faithfully and still choose the wrong yes/no/maybe. Faithfulness and correctness are different properties. *Analogy: a student who copies the textbook accurately but answers the exam question wrongly.*

## Not medical advice

Primum is a research prototype built to study selective prediction. It is not a medical device, has not been evaluated by clinicians, and must not be used to make decisions about anyone's health. If you have a medical concern, contact a qualified clinician; in an emergency, call your local emergency number.
