# Primum web

The interface for Primum: four views over the API in `src/primum/api`.

| View | Path | What it is for |
| --- | --- | --- |
| Ask | `/` | Ask a question and watch the pipeline run: evidence, checked claims, and a release or withhold decision against the certified threshold. |
| Retrieval lab | `/retrieval` | One query, several retrievers, rankings side by side with rank changes drawn. |
| Experiments | `/experiments` | RQ1 to RQ3, rendered from the results files, with a picker over generator runs. |
| Method | `/method` | The thesis and each engine in plain language, with an interactive explainer of the certified threshold. |

## Run it

```bash
pnpm install
pnpm dev          # http://localhost:3000, expects the API on http://localhost:8000
```

Start the API from the project root with `primum serve`. To point the interface elsewhere, set
`NEXT_PUBLIC_API_URL` (see `.env.example`).

```bash
pnpm build        # production build
pnpm typecheck    # tsc --noEmit
pnpm lint         # eslint
pnpm test         # node --test: SSE parser, pipeline state, statistics, results readers, markdown
```

Node 22.18 or newer is needed for `pnpm test`, which runs the TypeScript tests directly.

## How it talks to the API

- `src/lib/types.ts` mirrors `src/primum/types.py` field for field.
- `src/lib/api.ts` is the typed client. `ask()` posts to `/api/ask` and yields pipeline events.
- `src/lib/sse.ts` parses `text/event-stream` from a `fetch` body (LF, CRLF or CR line endings,
  frames split across chunks, keep-alive comments).
- `src/lib/ask-state.ts` is the reducer that turns events into what the Ask view shows.
  `outcomeOf()` names the result: answered, withheld, emergency, failed.

Errors are specific. `no_api_key`, `budget` and `no_index` each say what stopped and which
command or setting fixes it; an unreachable API says where it was expected.

## Results files: what the Experiments view reads

`SPEC.md` fixes what each results file contains but not its exact layout, so `src/lib/results.ts`
looks for structure instead of assuming one layout. It recognises:

- **An estimate**: a number, `[estimate, lo, hi]`, or an object with one of `mean`, `value`,
  `estimate`, `rate` and an interval as `ci: [lo, hi]` or `lo`/`hi`, `ci_low`/`ci_high`,
  `lower`/`upper`.
- **A table**: an object mapping system names to objects of estimates
  (`{"bm25": {"ndcg@10": {...}}}`), or an array of rows with a `name`, `system`, `retriever`,
  `method` or `model` field. Found at any depth; `meta` and `per_query` are skipped. A table
  whose columns are risk levels (RQ3 `prediction_sets.{lac,aps}.{set_coverage,...}.{"0.05",...}`)
  is turned to one row per level. Columns named `p_value`, `*_p` or `*_p_holm` are shown as
  p-values (`<0.001`), never plotted.
- **A reliability diagram**: under a key containing `reliab`, per model, bins with
  `confidence` and `accuracy` (and `count`), as rows or as parallel arrays.
- **A risk-coverage curve**: under a key containing `risk_coverage`, per model, `coverage` and
  `risk`, as rows or as parallel arrays.
- **A record per method and risk level**: any object with a `violation_rate` (and optionally
  `violation_ci`, `coverage` or `mean_coverage`, `risk`), whose level is an `alpha` field or a
  path key such as `"0.10"`.
- **A loss curve**: `steps` and `loss` arrays, or `loss_curve: [{step, loss}]` as
  `encoder_training.json` writes it. Its `dev.{before,after}` block is drawn as a table under RQ1.
- **Provenance**: `meta.created`, `meta.n`, δ from `delta`, `meta.delta` or `meta.config.delta`,
  and `meta.models` shown as `name: value` pairs.

Read by name rather than by shape:

- **RQ2 `comparisons`**: `{vs_<reference>: {<model>: {brier_diff, brier_p, auroc_diff,
  auroc_better_share}}}`, differences being model minus reference. The "Reading" column comes from
  the Brier interval (entirely below zero, entirely above, or straddling it), so the page cannot
  claim an improvement the file does not show.
- **RQ3**: the charts draw the top-level methods under `selective`; `single_signal` and
  `min_selected_sensitivity` variants are in the table only. A summary states the largest
  violation rate of `certified` against δ and its coverage range, and `meta.guarantee` is quoted.
- **`pubmedqa_runs.json`**: `summary.pubmedqa.{closed_book,rag,oracle}` and `paired_tests` are
  drawn at the top of RQ2; `records` (every per-question answer) is not printed. `meta.complete:
  false` is flagged.

Anything not recognised is still shown as raw JSON under each section. A layout change can hide a
figure; it cannot put a wrong number on screen.

## Generator runs

The experiments were run with more than one generator. Each run's generator-dependent files
(`pubmedqa_runs`, `rq2_confidence`, `rq3_conformal`) sit in their own directory: the main run in
`results/`, the others in a subdirectory. The API names runs from a fixed whitelist
(`RUNS` in `src/primum/api/results.py`); a request names a run, never a path.

| Run | Directory |
| --- | --- |
| `main` (default) | `results/` |
| `sonnet` | `results/sonnet/` |
| `sonnet_sc300` | `results/sonnet_sc300/` |

- `GET /api/experiments?run=<run>`: `{run, available, files, runs}`. `runs` lists every run that has
  files, each as `{run, generator, n, self_consistency_samples, conditions, n_failed, available}`,
  read from that run's `meta` (`models.llm`, `n`, `self_consistency_samples`, `conditions`) and
  `summary.pubmedqa.<condition>.n_failed` (from the RQ2 or RQ3 `meta` when the run has no
  `pubmedqa_runs.json`).
- `GET /api/experiments/{name}?run=<run>`: that run's file, verbatim.
- `GET /api/calibration?run=<run>`: the `calibration` block of that run's `rq3_conformal.json`, with
  `run` and `llm` (the generator, `meta.models.llm`) added. Thresholds and δ come from the
  deployment bundle saved for that generator (`artifacts/deployment/<llm>/bundle.json`; the live
  one when it is the server's generator), else from the file.

Omitting `run` means `main`. An unknown run, including any path-like value, is a 404.

The Experiments page picks a run with a radio group (labels from `runLabel` in
`src/lib/results.ts`) and re-reads RQ2, RQ3 and the PubMedQA summary for it; RQ1 and encoder
training involve no generator and are always read from the main run. A provenance line gives the
run's generator, samples per question, n and failed generations, all from the index. Where a run
lacks something, the page says so: a missing results file reads "not computed for this run", a
run with only the `rag` condition says there is no closed-book or oracle comparison, and RQ2's
`meta.unavailable_features` / `skipped_feature_sets` are listed as not measured. Runs are shown one
at a time; the page makes no comparison between generators. The Method page's threshold explainer
has the same picker, over the runs that have RQ3 results, and names the generator it shows.

The Ask view names the server's configured generator (`/api/meta` `models.llm`) and the generator
of each answer; switching generator is a server setting (`PRIMUM_LLM_MODEL`), which also switches
the deployment bundle.

`/api/calibration` is read by `parseCalibration` in `src/lib/stats.ts`: either
`{p_correct: [...], correct: [...]}` or `{items: [{p_correct, correct}]}` for the calibration rows
(the rows the thresholds were certified on), an optional `heldout` block in the same layout (rows
never used to fit, calibrate or certify), `thresholds` keyed by α (or a list of
`{alpha, threshold}`), `delta`, and the certification settings `n_grid` and `min_selected`.

## The threshold explainer

`src/lib/stats.ts` implements the certificate in the browser: exact binomial tail p-values and
fixed-sequence testing over descending confidence quantiles, following
`primum.uncertainty.conformal.certify_threshold`. Moving the slider recomputes the certificate on
the real calibration rows, using the server's `n_grid` and `min_selected` (20 only if the server
sends none), so it lands on the same thresholds the server certified, which are drawn for
comparison. A split picker shows the same threshold applied to the held-out rows: the certificate
is always computed on the calibration split, and the held-out view is its out-of-sample check. The
Python module remains the reference implementation.

## Concept dictionary

Concept linking needs the `kb` corpus. When it is not built, the server says so instead of
failing: the `concepts` pipeline event carries `available: false` and `/api/retrieve` answers
`concepts_available: false`. The Ask view then draws the Concepts stage as not run ("no concept
dictionary is built"), and the Retrieval lab says the dictionary is missing and stops offering
`+concepts` retrievers, which could only fail.

## Development fixtures

`NEXT_PUBLIC_FIXTURES=1 pnpm dev` replays canned pipeline events on the Ask view so its states
(released, withheld, emergency, urgent, no bundle, benchmark, and each error) can be worked on
without a backend. The view is bannered "Fixture data. Not real output." while the flag is set,
every passage in the fixture is titled as a fixture, and the fixture module is not loaded
otherwise. There are no fixtures for Experiments or the explainer.

## Design

See `DESIGN.md` for the direction and the reason behind each major decision.
