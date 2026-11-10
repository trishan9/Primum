# Primum — system specification

*Primum non nocere* — "first, do no harm". A clinical question-answering system whose
defining behaviour is that it **declines to answer when it cannot certify its own reliability**.

This file is the build contract. `src/primum/types.py` and `src/primum/config.py` are already
written and are authoritative: do not change their public names without updating every caller.

## 1. Thesis

> A retrieval-grounded LLM answering medical questions can be made *selectively reliable*.
> If we (i) measure how well each claim in an answer is supported by retrieved evidence,
> (ii) learn a calibrated estimate of the probability that the answer is correct, and
> (iii) choose the answer/abstain threshold with a distribution-free statistical test,
> then the error rate among the answers the system does give is at most α with probability
> at least 1 − δ, at higher coverage than single-signal confidence baselines.

Research questions, each answered by one experiment script and one results file:

| | Question | Experiment | Output |
|---|---|---|---|
| RQ1 | Does a domain-adapted (contrastively fine-tuned) encoder, fused with lexical BM25, retrieve biomedical evidence better than either alone? | `experiments/rq1_retrieval.py` | `results/rq1_retrieval.json` |
| RQ2 | Which signals predict that a generated answer is correct, and is a learned combination better calibrated than any single signal? | `experiments/rq2_confidence.py` | `results/rq2_confidence.json` |
| RQ3 | Does the certified threshold actually keep selective risk ≤ α on unseen questions, and how much coverage does it cost? | `experiments/rq3_conformal.py` | `results/rq3_conformal.json` |

Integrity rule (non-negotiable): **no number appears in the README, paper, or UI unless it is
read from a file in `results/` produced by running code.** No placeholder, illustrative, or
"expected" metrics anywhere. If an experiment has not run, the UI and docs say so.

## 2. Data (all openly licensed, downloaded by code)

| Corpus id | Source | Used for |
|---|---|---|
| `pubmedqa` | HF dataset `qiaojin/PubMedQA`: all 1,000 `pqa_labeled` abstracts + `pqa_unlabeled` abstracts as distractors | QA experiments (RQ2, RQ3), gold-abstract retrieval (RQ1) |
| `nfcorpus` | BEIR NFCorpus (corpus, queries, train/dev/test qrels) | Retrieval benchmark (RQ1) |
| `kb` | MedlinePlus health-topics XML (NLM, public domain) + the `pubmedqa` passages (+ optional local PDFs in `data/raw/pdf/`, git-ignored) | Live question answering |

- Contrastive training pairs: `pqa_artificial` (question → its abstract), excluding every PMID in `pqa_labeled`; plus NFCorpus *train* qrels (query → relevant doc). Never train on NFCorpus test queries or PQA-L questions.
- A PubMedQA passage = the abstract *without its conclusion* (the `context.contexts` list joined), title = the question's article title if available else "". `long_answer` (the conclusion) is never indexed: it would leak the label.
- Processed layout: `data/processed/<corpus>/passages.jsonl` (one `Passage` JSON per line) plus dataset-specific files (`pubmedqa/labeled.jsonl` of `PubMedQAExample`, `nfcorpus/queries.<split>.jsonl`, `nfcorpus/qrels.<split>.tsv`, `pairs/train.jsonl` of `{"query","positive"}`).
- MedlinePlus topics also yield a concept dictionary `data/processed/kb/concepts.jsonl`: `{preferred, synonyms[], mesh_ids[], url}` from topic titles, `also-called`, `see-reference` and `mesh-heading` elements.

Module `primum.data` public API (file: `data/__init__.py` re-exports):

```python
def build_corpus(name: Literal["pubmedqa","nfcorpus","kb"], *, force: bool = False, max_distractors: int | None = None) -> Path
def load_passages(corpus: str) -> list[Passage]
def load_pubmedqa_labeled() -> list[PubMedQAExample]          # 1,000 items, stable order
def load_nfcorpus(split: Literal["train","dev","test"]) -> tuple[dict[str,str], dict[str,dict[str,int]]]  # queries, qrels[qid][doc_id]=grade
def build_contrastive_pairs(n_pubmedqa: int = 40000, *, force: bool = False) -> Path
def load_contrastive_pairs() -> list[tuple[str,str]]
def load_concepts() -> list[dict]
def chunk_text(text: str, *, max_words: int = 220, overlap: int = 40) -> list[str]   # sentence-aware
```

For `nfcorpus`, `Passage.passage_id == doc_id ==` the BEIR doc id (documents are short; no chunking).

## 3. Engines

### 3.1 Retrieval engine — `primum.retrieval`

```python
class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int = 10) -> list[Hit]: ...
    def search_batch(self, queries: list[str], k: int = 10) -> list[list[Hit]]: ...

def build_retriever(name: str, corpus: str) -> Retriever      # cached per (name, corpus)
RETRIEVERS = ["bm25", "dense-base", "dense-biomed", "dense-ft", "hybrid-base", "hybrid-ft", "hybrid-ft+rerank"]
```

- `bm25.py` — Okapi BM25 **implemented from scratch** on a scipy sparse term–document matrix (k1=1.2, b=0.75, Lucene-style idf `ln(1 + (N − df + 0.5)/(df + 0.5))`), simple biomedical-aware tokenizer (lowercase, keep alphanumerics and intra-word hyphens/digits such as `il-6`, `covid-19`; small stopword list). Index is saved/loaded under `data/indexes/<corpus>/bm25/`.
- `dense.py` — `DenseIndex(encoder, corpus)`: encodes passages with a sentence-transformers model on MPS/CPU, L2-normalises, stores a float32 `.npy`; search is exact inner product in numpy (corpora are ≤ ~100k passages, so exact search is fast and has no ANN recall loss). Cached under `data/indexes/<corpus>/dense-<slug>/`.
- `fusion.py` — Reciprocal Rank Fusion: `score(d) = Σ_r 1 / (rrf_k + rank_r(d))`.
- `rerank.py` — cross-encoder reranker over the fused top `candidate_k`.
- `train_contrastive.py` — **our own PyTorch training loop** (no `SentenceTransformer.fit`): bi-encoder initialised from `settings.base_encoder`, mean pooling, in-batch-negative InfoNCE with temperature τ (default 0.05), symmetric loss, AdamW + linear warmup, optional one round of BM25-mined hard negatives. Writes the model (sentence-transformers format so `DenseIndex` can load it) to `settings.finetuned_encoder_dir` and a training log `results/encoder_training.json` (loss curve, steps, hyperparameters, wall time, device).
- `metrics.py` — nDCG@k (graded, with the linear gain `rel / log2(i+1)` that pytrec_eval/BEIR use, so numbers are comparable to published BEIR results), Recall@k, MRR@k, MAP; all return per-query arrays so the stats module can bootstrap.
- `concepts.py` — dictionary-based longest-match concept recogniser over `load_concepts()`: `link_concepts(text) -> list[Concept]`, `expand_query(query) -> str` (appends synonyms of recognised concepts).

`Hit.components` must carry every stage score that contributed (`bm25`, `dense`, `rrf`, `rerank`) and the per-retriever ranks as `bm25_rank`, `dense_rank`.

### 3.2 Language-model layer — `primum.llm`

```python
class LLMResult(BaseModel): text: str; parsed: Any | None; usage: Usage; model: str
class BudgetExceeded(RuntimeError): ...
class LLMUnavailable(RuntimeError): ...       # no API key configured

class LLMClient:
    def __init__(self, settings: Settings | None = None): ...
    def complete(self, *, system: str, user: str, schema: type[BaseModel] | None = None,
                 model: str | None = None, max_tokens: int = 2000, effort: str | None = None,
                 sample_id: int = 0, tag: str = "") -> LLMResult
    async def acomplete(...same...) -> LLMResult
    async def astream(self, *, system: str, user: str, model: str | None = None,
                      max_tokens: int = 4000, tag: str = "") -> AsyncIterator[str]   # text deltas
    def spend(self) -> dict   # {"total_usd":..., "by_tag": {...}, "calls":..., "cached_calls":...}
def get_llm() -> LLMClient
```

- Official `anthropic` SDK only (1.x). Read the `claude-api` skill before writing this module; follow it for structured outputs (`output_config.format` / `messages.parse`), streaming, error-handling chains, and model ids. Constraints that matter here: `claude-haiku-5-5` rejects non-default `temperature`/`top_p`/`top_k` and `budget_tokens`; thinking is adaptive and on by default — control cost with `output_config={"effort": ...}`. Do not use assistant prefill or forced `tool_choice`.
- Construct the client with `api_key=settings.anthropic_api_key, base_url=settings.anthropic_base_url` (the shell may export a foreign `ANTHROPIC_BASE_URL`).
- **Response cache** (sqlite at `settings.cache_db`): key = sha256 of (model, system, user, schema JSON, effort, max_tokens, sample_id). A cache hit returns `usage.cached=True, cost_usd=0`. `sample_id` is how callers draw several independent samples of the same prompt (self-consistency) while keeping every sample reproducible.
- **Budget ledger** (same sqlite file): every real call records tokens and cost (price table in code for the models in the skill; Haiku 5.5 $0.10/$0.50 per MTok, Sonnet 5.5 $2/$10, Opus 5.5 $4/$20, cache reads 10% of input). Before each real call, if cumulative spend ≥ `settings.budget_usd` raise `BudgetExceeded`. This is a hard stop, not a warning.
- Check `stop_reason` (`refusal`, `max_tokens`) before reading content; surface a typed error.
- Prompts live in `llm/prompts.py` as module constants; nothing else in the codebase contains prompt text.

### 3.3 Generation — `primum.generation`

```python
def answer_benchmark(question: str, hits: list[Hit], *, sample_id: int = 0, llm=None) -> tuple[BenchmarkAnswer, Usage]
async def stream_chat_answer(question: str, hits: list[Hit], *, llm=None) -> AsyncIterator[str]
def extract_claims(question: str, answer_text: str, hits: list[Hit], *, llm=None) -> tuple[ChatAnswer, Usage]
def format_evidence(hits: list[Hit]) -> str     # "[1] (title) text ..." blocks, shared by all prompts
```

- Benchmark mode: one structured call. With `hits=[]` it is the closed-book condition (the prompt says no evidence is provided). Claims must cite passages by 1-based index.
- Chat mode: stream a plain-language, evidence-bound answer with `[n]` markers; instruct the model to say what the evidence does not cover rather than fill gaps from memory. Then one structured call (`extract_claims`) decomposes the finished answer into atomic claims with citations and a verbal confidence.

### 3.4 Verification engine — `primum.verification`

```python
class NLIModel:
    def predict(self, premises: list[str], hypotheses: list[str]) -> np.ndarray   # [n,3] = entail, neutral, contra
def get_nli() -> NLIModel
def verify_claims(claims: list[Claim], hits: list[Hit], *, nli=None) -> list[ClaimVerdict]
def split_sentences(text: str) -> list[str]
```

A claim is checked against every sentence (and every 2-sentence window) of its cited passages; if it cites nothing, against all hits. `p_entail` = max over premises; `p_contra` = the contradiction probability of the premise with the highest contradiction. Label: `supported` if `p_entail ≥ 0.5`, else `contradicted` if `p_contra ≥ 0.5`, else `unsupported`. The NLI model is a pretrained local cross-encoder (`settings.nli_model`); read label order from the model config, never hardcode it.

### 3.5 Uncertainty engine — `primum.uncertainty` (pure numpy/torch; no LLM, no I/O besides save/load)

`features.py`
```python
FEATURES_DEPLOY = ["ret_top", "ret_mean", "ret_margin", "ret_agree", "nli_mean_entail", "nli_min_entail",
                   "nli_max_contra", "frac_supported", "n_claims", "verbal_conf"]
FEATURES_FULL = FEATURES_DEPLOY + ["sc_agreement", "sc_entropy", "verbal_margin"]
def extract_features(hits, verdicts, *, verbal_conf, sample_decisions: list[Decision] | None = None,
                     probabilities: dict | None = None) -> dict[str, float]
```
`ret_*` use a scale-free score (the reranker score if present else the dense cosine; `ret_agree` = fraction of the top-k that both BM25 and dense placed in their own top-k). `sc_agreement` = share of self-consistency samples equal to the modal decision; `sc_entropy` = normalised entropy of the sample distribution; `verbal_margin` = top stated probability minus the runner-up. Missing inputs give 0.0 and callers choose the matching feature list. With no hits (closed-book), retrieval and NLI features are 0.

`models.py` — `LogisticConfidence` and `MLPConfidence` (one hidden layer, dropout, weight decay, early stopping on a validation split), **both written in PyTorch** with feature standardisation fitted on the training split; `fit(X, y)`, `predict_proba(X) -> np.ndarray`, `state_dict`-style `to_json()/from_json()`.

`calibration.py` — `TemperatureScaling`, `PlattScaling`, `IsotonicCalibration` (pool-adjacent-violators **from scratch**), plus `ece(p, y, n_bins=10, strategy="equal_width"|"equal_mass")`, `brier(p, y)`, `reliability_bins(p, y, n_bins)`.

`conformal.py`
```python
def conformal_quantile(scores: np.ndarray, alpha: float) -> float           # ceil((n+1)(1-alpha))/n empirical quantile; +inf if that level > 1
def lac_sets(probs_cal, y_cal, probs_test, alpha) -> np.ndarray[bool]        # [n_test, n_classes]
def aps_sets(probs_cal, y_cal, probs_test, alpha, *, randomized=True, rng=None) -> np.ndarray[bool]
def binomial_pvalue(k: int, n: int, alpha: float) -> float                   # P(Bin(n, alpha) <= k): p-value for H0: risk > alpha
def hoeffding_bentkus_pvalue(risk_hat: float, n: int, alpha: float) -> float
class SelectiveGuarantee(BaseModel): alpha; delta; threshold: float | None; n_cal; n_selected; cal_coverage; cal_risk; pvalue
def certify_threshold(conf_cal, correct_cal, *, alpha, delta, n_grid: int = 100, min_selected: int = 20,
                      method: Literal["binomial","hb"] = "binomial") -> SelectiveGuarantee
```
`certify_threshold` is Learn-then-Test with **fixed-sequence testing**: candidate thresholds are the descending quantiles of the calibration confidences, restricted to those selecting ≥ `min_selected` points; walk from the most conservative candidate towards lower thresholds, testing `H_j: R(λ_j) > α` with the exact binomial tail p-value, and stop at the first failure; return the last λ that passed (or `threshold=None` if the first test fails). This controls the family-wise error, so `P(R(λ̂) ≤ α) ≥ 1 − δ` over the draw of the calibration set, with no distributional assumption beyond exchangeability. The grid and `min_selected` must be fixed before looking at labels (they depend only on `conf_cal`, not `correct_cal`).

`metrics.py` — `risk_coverage_curve`, `aurc`, `e_aurc`, `auroc` (rank-based, from scratch), `coverage_at_risk`, `selective_risk(conf, correct, threshold)`.
`stats.py` — `bootstrap_ci(values, stat=np.mean, n_boot=2000, level=0.95, rng)`, `paired_bootstrap_pvalue(a, b, n_boot, rng)`, `paired_permutation_pvalue(a, b, n_perm, rng)`.
`deploy.py` — `DeploymentBundle` (feature list, confidence model JSON, calibrator, certified thresholds for α ∈ {0.05, 0.10, 0.15, 0.20}, provenance: n_cal, model id, retriever, date string passed in by the caller); `save(dir)`, `load(dir)`, `score(features) -> ConfidenceReport`.

Unit tests for this package must include **simulation tests of the guarantees** on synthetic data: marginal coverage of LAC/APS sets ≥ 1 − α on average; `certify_threshold` violation frequency `P(true risk > α) ≤ δ` (within Monte-Carlo error) over ≥ 2,000 synthetic calibration draws in a setting where the true risk at each threshold is known in closed form; PAV output is monotone and matches scikit-learn's `IsotonicRegression`; `auroc` matches scikit-learn.

### 3.6 Safety triage — `primum.safety.triage`

`triage(question: str) -> TriageResult`. A transparent rule lexicon of red-flag presentations (chest pain with radiation/sweating, stroke signs, anaphylaxis, suicidal intent, overdose, severe bleeding, infant fever, etc.). `emergency` short-circuits the pipeline: the system shows an emergency-services message and still answers nothing clinical. Rules only, clearly documented as such — this is a safety rail, not a research contribution.

### 3.7 Orchestrator — `primum.pipeline.orchestrator`

```python
class Event(BaseModel): type: str; data: dict
async def run_pipeline(question: str, *, mode: Literal["chat","benchmark"] = "chat", retriever: str | None = None,
                       alpha: float | None = None, gold_label: Decision | None = None) -> AsyncIterator[Event]
```
Stages and the events they emit, in order (every stage also emits `stage` `{name, status: "start"|"done", ms}`):

| Stage | Event type → data |
|---|---|
| triage | `triage` → `TriageResult` |
| concepts | `concepts` → `{concepts: Concept[]}` |
| retrieve | `retrieval` → `{retriever, hits: Hit[]}` |
| generate | `token` → `{text}` (many, chat mode) then `answer` → `{text, decision?}` |
| verify | `verdicts` → `{verdicts: ClaimVerdict[]}` |
| confidence | `confidence` → `ConfidenceReport` |
| done | `done` → `PipelineResult` (also persisted to the trace store) |
| any failure | `error` → `{stage, message, kind}` (`kind` ∈ `no_api_key`, `budget`, `no_index`, `internal`) |

Chat mode uses `FEATURES_DEPLOY`; benchmark mode additionally draws `self_consistency_samples` decisions and uses `FEATURES_FULL` if the deployment bundle has a full model. Deployment bundles are per generator: `settings.deployment_dir` is `artifacts/deployment/<settings.llm_model>/`, because thresholds certified on one model's answers say nothing about another's. Setting `PRIMUM_LLM_MODEL` therefore switches the generator and the bundle together (default `claude-haiku-5-5`; `claude-sonnet-5-5` has its own bundle). If no deployment bundle exists yet for the configured generator, `confidence` is emitted with `threshold=None, answered=True` and a flag `uncalibrated: true` in the event data so the UI can say the guarantee is not active. Blocking model inference (encoders, NLI) runs in a thread executor so the event loop stays responsive. When the decision is to abstain, the answer text is still returned (the UI shows it collapsed behind an explicit "show withheld draft" control) — the trace must record that the system abstained.

### 3.8 API — `primum.api.app:app` (FastAPI, port 8000)

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness + which artefacts exist (`indexes`, `finetuned_encoder`, `deployment_bundle`, `api_key`) |
| `GET /api/meta` | models, retriever list, α options, δ, corpus sizes, LLM spend so far, budget |
| `POST /api/ask` | body `{question, mode?, retriever?, alpha?, benchmark_qid?}` → `text/event-stream`; each SSE message has `event: <type>` and `data: <json>` exactly as in §3.7 |
| `POST /api/retrieve` | body `{query, k?, retrievers: string[], corpus?}` → `{results: {retriever: Hit[]}, concepts: Concept[], expanded_query}` |
| `GET /api/benchmark/questions?limit=&offset=&split=` | PubMedQA items (no gold label unless `reveal=true`) |
| `GET /api/experiments` | `{available: string[], files: {name: {updated}}}` |
| `GET /api/experiments/{name}` | the raw results JSON (`rq1_retrieval`, `rq2_confidence`, `rq3_conformal`, `encoder_training`, `pubmedqa_runs`) or 404 |
| `GET /api/calibration` | per-item `{p_correct, correct}` arrays from the held-out data plus certified thresholds, for the interactive explainer; 404 until RQ3 has run |
| `GET /api/traces?limit=` , `GET /api/traces/{id}` | stored `PipelineResult`s |

`/api/experiments*` and `/api/calibration` take an optional `?run=` (`main` by default, `sonnet`, `sonnet_sc300`; see the results layout in §3.9); an unknown run is a 404. CORS for `http://localhost:3000`. Models load lazily on first use and a `GET /api/warmup` triggers loading. Errors are JSON `{error: {kind, message}}` with correct status codes. Input validation: question length 3–2,000 chars.

### 3.9 Experiments — `primum.experiments` (CLI: `primum exp <name>`)

`run_pubmedqa.py` — for each of the 1,000 PQA-L questions and each condition ∈ {`closed_book`, `rag` (default retriever, top-k), `oracle` (gold abstract only)}: generate the benchmark answer (+ self-consistency samples for the `rag` condition), verify claims, extract features, record correctness. Concurrency via a thread pool, resumable (the LLM cache makes re-runs free), prints spend as it goes and stops cleanly on `BudgetExceeded`. Writes `results/pubmedqa_runs.json`: per-item records `{qid, condition, decision, gold, correct, probabilities, sample_decisions, features, n_supported, n_claims}` + summary accuracy / macro-F1 with bootstrap CIs.

`rq1_retrieval.py` — NFCorpus test (nDCG@10, Recall@100, MRR@10) and PubMedQA gold-abstract retrieval (Recall@1/5/20, MRR@10) for every name in `RETRIEVERS` plus `bm25+concepts` (query expansion ablation); 95% bootstrap CIs; paired permutation tests of each system against `bm25` and against `dense-base`.

`rq2_confidence.py` — on the `rag` records: for each single signal and for logistic / MLP on `FEATURES_DEPLOY` and `FEATURES_FULL`, with raw / temperature / Platt / isotonic calibration: AUROC, AURC, E-AURC, ECE, Brier, via repeated stratified splits (train 40% / calibration 30% / test 30%, 200 repeats, seeds fixed), mean and 95% interval; feature ablation (drop one feature group: retrieval, NLI, verbal, self-consistency); reliability-diagram bins and the risk–coverage curve for the main models (test-split predictions pooled over all repeats). A feature that is constant across all records is treated as not measured (e.g. `sc_agreement`, `sc_entropy` from a run with 1 sample per question): it is listed in `meta.unavailable_features`, its single-signal estimator is left out and so is any ablation group none of whose features was measured, and every non-`deploy` feature set that needs it is skipped and listed in `meta.skipped_feature_sets`. `deploy` never contains self-consistency features, so it always runs.

`rq3_conformal.py` — same repeated-split protocol: fit confidence model on train, `certify_threshold` on calibration for α ∈ {0.05, 0.10, 0.15, 0.20}, δ = 0.10; on test record selective risk and coverage. Report the empirical violation rate `P(test risk > α)` with a Clopper–Pearson interval, mean coverage, and the same for baselines (uncalibrated plug-in threshold = smallest λ whose *calibration* risk ≤ α with no correction; each single-signal confidence with the certified procedure). Also prediction-set coverage and mean set size for LAC and APS. Finally fit the deployment bundle on a fixed final split and save it to `settings.deployment_dir_for(<llm of the records>)`, i.e. `artifacts/deployment/<llm_model>/bundle.json` for the generator that produced the answers (not the one currently configured), and write the per-item `{p_correct, correct}` arrays used by `/api/calibration`: the calibration rows the deployed thresholds were certified on, plus a separate `heldout` block of rows never used to fit, calibrate or certify. The confidence model and its calibrator see only the training split; `certify_threshold` sees only the untouched calibration split.

`report.py` — reads `results/*.json` and writes `results/figures/*.png|svg` (matplotlib, consistent restrained style) and `docs/RESULTS.md` (tables generated from the JSON, never hand-typed).

Every results JSON has a top-level `meta`: `{created, config, n, seeds, models}`. `created` is passed in by the caller (the CLI uses the current time); `config` is a git-free snapshot of the analysing process's settings, so for RQ2/RQ3 the generator is `models.llm`, read from the records, not `config.llm_model`.

Results layout, one directory per generator run. Every experiment's `run()` takes `results_dir` (default `settings.results_dir`) and reads and writes its files there:

| Directory | Run (API `?run=`) | Contents |
|---|---|---|
| `results/` | `main` | default generator (`claude-haiku-5-5`): all three conditions, 5 samples per `rag` question; every results file, plus `figures/` |
| `results/sonnet/` | `sonnet` | `claude-sonnet-5-5`, `rag` only, 1 sample per question, all questions: `pubmedqa_runs.json`, `rq2_confidence.json`, `rq3_conformal.json` |
| `results/sonnet_sc300/` | `sonnet_sc300` | `claude-sonnet-5-5`, `rag` only, 5 samples per question, first 300 questions in dataset order (`limit=300`; not a random sample): `pubmedqa_runs.json`, `rq2_confidence.json` |

RQ1 and encoder training do not depend on the generator and exist only in `results/`. `report.py` adds a second-generator section from the two subdirectories; the API serves them through the whitelist `primum.api.results.RUNS`. Comparisons across directories are descriptive (separate runs and separate repeated splits).

### 3.10 CLI — `primum.cli` (typer)

`primum data build <corpus>|all`, `primum data pairs`, `primum index build <corpus> [--retriever ...]`, `primum train encoder [--epochs --batch-size --max-pairs]`, `primum exp pubmedqa|rq1|rq2|rq3|report|all`, `primum ask "<question>"` (prints the staged pipeline in the terminal), `primum serve` (uvicorn), `primum spend`.

## 4. Web app — `web/` (Next.js App Router, TypeScript, Tailwind)

Talks to the API at `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`). Four views:

1. **Ask** (`/`) — the instrument. Question box (+ example questions and a "benchmark question" picker). As SSE events arrive: a pipeline rail showing each stage and its latency; the streamed answer with `[n]` citation chips; each claim marked supported / unsupported / contradicted with the supporting sentence on hover/focus; an evidence column listing retrieved passages with source, link (PubMed / MedlinePlus), and per-stage scores; a confidence panel showing calibrated P(correct), the certified threshold for the chosen α, and the resulting decision. **Abstention is a first-class designed state**, not an error: it states plainly that the system is withholding an answer because it could not certify it at the chosen risk level, shows what evidence it found, and offers the withheld draft behind an explicit control. Emergency triage is another designed state. α is user-selectable (0.05 / 0.10 / 0.15 / 0.20).
2. **Retrieval lab** (`/retrieval`) — one query, rankings from several retrievers side by side, rank changes made visible, recognised concepts and the expanded query.
3. **Experiments** (`/experiments`) — RQ1–RQ3 rendered from `/api/experiments/*`: retrieval table with CIs, reliability diagram, risk–coverage curves, violation-rate chart against the δ line, coverage-vs-α. Honest empty states when a results file is missing.
4. **Method** (`/method`) — the thesis, the pipeline, and each idea explained in plain language with one analogy each, including an interactive explainer of the certified threshold driven by real data from `/api/calibration` (drag α, watch the threshold, coverage, and risk move).

A persistent, quiet notice: research prototype, not medical advice.

## 5. Quality bar

- Python 3.12, type hints throughout, `ruff` clean, `pytest` passing (fast tests need no network, no models, no API key).
- Modules are deep: small public surface (the signatures above), internals private. No LangChain.
- Deterministic where possible: seeds from `settings.seed`; every experiment is re-runnable and resumable.
- Fail loudly and specifically (typed errors), never silently fall back to a different method in an experiment.
- Comments explain *why* (the statistical reasoning, the leak being avoided), not *what*.
