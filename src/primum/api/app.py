"""HTTP API for the web interface (SPEC §3.8). Run with `primum serve` or `uvicorn primum.api.app:app`.

Design points:
- No model weights or indexes load at import or start-up (the import itself is not light:
  torch comes in with the uncertainty engine). They load on the first request that needs
  them (or on `GET /api/warmup`), inside worker threads, so the event loop keeps serving
  other requests while a model loads.
- Every error body is `{"error": {"kind", "message"}}`. Engine failures use the four pipeline
  kinds (`pipeline.errors`); request problems use `invalid` (422) and `not_found` (404).
- `/api/ask` streams Server-Sent Events: one message per pipeline event, `event: <type>` and
  `data: <json>`. Errors after the stream has started arrive as an `error` event, because the
  200 status line has already been sent by then.
- `/api/experiments*` and `/api/calibration` take `?run=` from a fixed whitelist of results
  subdirectories (`results.RUNS`), one per generator run; an unknown run is a 404.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import AfterValidator, BaseModel, Field, StringConstraints, field_validator
from sse_starlette import EventSourceResponse
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from primum.api.results import (
    DEFAULT_RUN,
    ResultMissing,
    UnknownRun,
    calibration_data,
    experiment_generator,
    experiment_path,
    experiments_index,
    run_dir,
)
from primum.config import Settings, get_settings, is_safe_model_name
from primum.data import CorpusNotBuilt
from primum.pipeline import Engines, Event, default_engines, error_kind, run_pipeline
from primum.pipeline.errors import display_paths, error_message
from primum.retrieval import RETRIEVERS
from primum.types import Claim, Hit, Passage, PubMedQAExample
from primum.uncertainty import ALPHAS, BUNDLE_FILENAME, DeploymentBundle

CORPORA = ("pubmedqa", "nfcorpus", "kb")
# The systems compared in RQ1, which are the ones the interface offers.
API_RETRIEVERS = [*RETRIEVERS, "bm25+concepts"]
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

_STATUS = {"no_api_key": 503, "budget": 503, "no_index": 503, "internal": 500}

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=2000)]


class ApiError(Exception):
    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind, self.message = status, kind, message


def _engine_error(exc: Exception, stage: str) -> ApiError:
    kind = error_kind(exc, stage)
    # `error_message` logs an internal failure and keeps server details out of the body.
    return ApiError(_STATUS[kind], kind, error_message(exc, stage))


# ---------------------------------------------------------------------------- request bodies


def _known_retriever(name: str) -> str:
    if name not in API_RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; expected one of {API_RETRIEVERS}")
    return name


RetrieverName = Annotated[str, AfterValidator(_known_retriever)]


class AskRequest(BaseModel):
    question: Text
    mode: Literal["chat", "benchmark"] | None = None  # default: benchmark iff benchmark_qid
    retriever: RetrieverName | None = None
    alpha: float | None = None
    benchmark_qid: str | None = None

    @field_validator("alpha")
    @classmethod
    def _certified_alpha(cls, alpha: float | None) -> float | None:
        # Only the levels the deployment bundle certifies; any other would have no threshold.
        if alpha is not None and not any(abs(alpha - a) < 1e-9 for a in ALPHAS):
            raise ValueError(f"alpha must be one of {list(ALPHAS)}")
        return alpha


class RetrieveRequest(BaseModel):
    query: Text
    k: int = Field(default=10, ge=1, le=100)
    retrievers: list[RetrieverName] = Field(min_length=1, max_length=len(API_RETRIEVERS))
    corpus: Literal["pubmedqa", "nfcorpus", "kb"] | None = None

    @field_validator("retrievers")
    @classmethod
    def _distinct(cls, names: list[str]) -> list[str]:
        if len(set(names)) != len(names):
            raise ValueError("retrievers must be distinct")
        return names


# ---------------------------------------------------------------------------- the app


def create_app(*, engines: Engines | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the API. `engines` and `settings` default to the real ones; tests pass their own."""
    settings = settings or get_settings()
    app = FastAPI(title="Primum", version="0.2.0")
    # Added first, so it sits inside the CORS layer (each added middleware wraps the previous).
    app.add_middleware(_JsonInternalErrors)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )

    def get_engines() -> Engines:
        # Resolved on first use, so importing the app does not open the trace store.
        return engines or default_engines()

    @lru_cache(maxsize=1)
    def benchmark_items() -> dict[str, PubMedQAExample]:
        from primum.data import load_pubmedqa_labeled

        return {example.qid: example for example in load_pubmedqa_labeled()}

    def load_benchmark() -> dict[str, PubMedQAExample]:
        try:
            return benchmark_items()
        except Exception as exc:
            raise _engine_error(exc, "benchmark") from exc

    # -------------------------------------------------------------- errors

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error_response(exc.status, exc.kind, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error["loc"] if part != "body")
            parts.append(f"{location}: {error['msg']}" if location else error["msg"])
        return _error_response(422, "invalid", "; ".join(parts))

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        kind = "not_found" if exc.status_code == 404 else "invalid"
        return _error_response(exc.status_code, kind, str(exc.detail))

    # -------------------------------------------------------------- status

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "api_key": bool(settings.anthropic_api_key),
            "deployment_bundle": (settings.deployment_dir / BUNDLE_FILENAME).is_file(),
            "finetuned_encoder": settings.finetuned_encoder_dir.is_dir(),
            "corpora": {name: _manifest(settings, name) is not None for name in CORPORA},
            "indexes": _built_indexes(settings),
        }

    @app.get("/api/meta")
    def meta() -> dict[str, Any]:
        return {
            "models": {
                "llm": settings.llm_model,
                "base_encoder": settings.base_encoder,
                "biomed_encoder": settings.biomed_encoder,
                # Relative to the project root: a client has no use for the server's layout.
                "finetuned_encoder": display_paths(str(settings.finetuned_encoder_dir), settings),
                "reranker": settings.reranker,
                "nli": settings.nli_model,
            },
            "retrievers": API_RETRIEVERS,
            "default_retriever": settings.default_retriever,
            "chat_corpus": settings.chat_corpus,
            "top_k": settings.top_k,
            "alphas": list(ALPHAS),
            "alpha": settings.alpha,
            "delta": settings.delta,
            "corpora": {name: _passage_count(settings, name) for name in CORPORA},
            "spend": get_engines().llm().spend(),
            "budget_usd": settings.budget_usd,
        }

    @app.get("/api/warmup")
    def warmup() -> dict[str, Any]:
        """Load what the first question would otherwise wait for; report each part separately."""
        e = get_engines()
        # Verifying one claim against a one-sentence passage is the cheapest call that makes
        # the verifier load its NLI model (with no evidence at all it never does).
        probe = Hit(
            passage=Passage(passage_id="warmup", doc_id="warmup", text="Warmup.", source="pdf"),
            score=0.0,
            rank=1,
            retriever="warmup",
        )
        steps = {
            "concepts": lambda: e.link_concepts("warmup"),
            "retriever": lambda: e.retriever(settings.default_retriever, settings.chat_corpus),
            "nli": lambda: e.verify([Claim(text="Warmup.")], [probe]),
            "deployment_bundle": e.bundle,
        }
        loaded: dict[str, Any] = {}
        for name, step in steps.items():
            started = time.perf_counter()
            try:
                step()
            except Exception as exc:
                stage = "retrieve" if name in ("concepts", "retriever") else name
                error = _engine_error(exc, stage)
                loaded[name] = {
                    "ok": False,
                    "error": {"kind": error.kind, "message": error.message},
                }
            else:
                loaded[name] = {"ok": True, "ms": round((time.perf_counter() - started) * 1000, 1)}
        return {"loaded": loaded}

    # -------------------------------------------------------------- pipeline

    @app.post("/api/ask")
    async def ask(body: AskRequest) -> EventSourceResponse:
        question, gold = body.question, None
        if body.benchmark_qid is not None:
            items = await run_in_threadpool(load_benchmark)
            example = items.get(body.benchmark_qid)
            if example is None:
                raise ApiError(404, "not_found", f"no PubMedQA question {body.benchmark_qid!r}")
            # The dataset's wording is authoritative: the gold label belongs to it.
            question, gold = example.question, example.label
        mode = body.mode or ("benchmark" if body.benchmark_qid else "chat")
        events = run_pipeline(
            question,
            mode=mode,
            retriever=body.retriever,
            alpha=body.alpha,
            gold_label=gold,
            benchmark_qid=body.benchmark_qid,
            engines=get_engines(),
        )
        return EventSourceResponse(_sse(events), ping=15)

    @app.post("/api/retrieve")
    async def retrieve(body: RetrieveRequest) -> dict[str, Any]:
        e = get_engines()
        corpus = body.corpus or settings.chat_corpus

        def search() -> dict[str, Any]:
            # One retriever at a time: they may each load a model, and memory is the limit.
            results = {}
            for name in body.retrievers:
                hits = e.retriever(name, corpus).search(body.query, k=body.k)
                results[name] = [hit.model_dump(mode="json") for hit in hits]
            return {"results": results, **_concept_annotations(e, body.query), "corpus": corpus}

        try:
            return await run_in_threadpool(search)
        except Exception as exc:
            raise _engine_error(exc, "retrieve") from exc

    @app.get("/api/benchmark/questions")
    def benchmark_questions(
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
        offset: Annotated[int, Query(ge=0)] = 0,
        split: Literal["all", "heldout"] = "all",
        reveal: bool = False,
    ) -> dict[str, Any]:
        # "heldout": the questions the deployment bundle kept out of both its fit and its
        # calibration rows, the only ones for which the live guarantee can apply. Without a
        # bundle no question is held out, so the list is empty.
        examples = list(load_benchmark().values())
        if split == "heldout":
            bundle = get_engines().bundle()
            heldout = set(bundle.heldout_qids) if bundle is not None else set()
            examples = [ex for ex in examples if ex.qid in heldout]
        page = examples[offset : offset + limit]
        items = [
            {"qid": ex.qid, "question": ex.question, **({"label": ex.label} if reveal else {})}
            for ex in page
        ]
        return {"items": items, "total": len(examples), "offset": offset, "limit": limit}

    # -------------------------------------------------------------- results

    def results_dir_of(run: str) -> Path:
        try:
            return run_dir(settings.results_dir, run)
        except UnknownRun as exc:
            raise ApiError(404, "not_found", str(exc.args[0])) from exc

    def bundle_for(run: str, generator: str | None) -> DeploymentBundle | None:
        """The deployment bundle RQ3 saved for `generator`: the live one for the live model."""
        if generator is None:
            # An RQ3 file that does not name its generator: only the main run's file (older
            # layout) is assumed to be the live model's; any other run shows its own thresholds.
            return get_engines().bundle() if run == DEFAULT_RUN else None
        if generator == settings.llm_model:
            return get_engines().bundle()
        # The name comes from a results file, not the request; still, it must stay one path part.
        if not is_safe_model_name(generator):
            return None
        directory = settings.deployment_dir_for(generator)
        if not (directory / BUNDLE_FILENAME).is_file():
            return None
        bundle = DeploymentBundle.load(directory)
        return bundle if bundle.llm_model == generator else None

    @app.get("/api/experiments")
    def experiments(run: str = DEFAULT_RUN) -> dict[str, Any]:
        results_dir_of(run)
        return experiments_index(settings.results_dir, run)

    @app.get("/api/experiments/{name}")
    def experiment(name: str, run: str = DEFAULT_RUN) -> FileResponse:
        try:
            path = experiment_path(results_dir_of(run), name)
        except (KeyError, ResultMissing) as exc:
            raise ApiError(404, "not_found", str(exc.args[0])) from exc
        return FileResponse(path, media_type="application/json")

    @app.get("/api/calibration")
    def calibration(run: str = DEFAULT_RUN) -> dict[str, Any]:
        directory = results_dir_of(run)
        try:
            generator = experiment_generator(directory, "rq3_conformal")
        except ResultMissing as exc:
            raise ApiError(404, "not_found", str(exc)) from exc
        bundle = bundle_for(run, generator)
        thresholds = delta = None
        if bundle is not None:
            # The thresholds the deployed rule for this generator applies, so the explainer
            # shows the real rule; without a bundle they come from the results file.
            thresholds = {f"{g.alpha:.2f}": g.threshold for g in bundle.guarantees}
            delta = bundle.delta
        try:
            payload = calibration_data(directory, thresholds=thresholds, delta=delta)
        except ResultMissing as exc:
            raise ApiError(404, "not_found", str(exc)) from exc
        return {**payload, "run": run, "llm": generator}

    @app.get("/api/traces")
    def traces(limit: Annotated[int, Query(ge=1, le=500)] = 50) -> dict[str, Any]:
        return {"items": get_engines().traces.list(limit)}

    @app.get("/api/traces/{trace_id}")
    def trace(trace_id: str) -> dict[str, Any]:
        payload = get_engines().traces.get(trace_id)
        if payload is None:
            raise ApiError(404, "not_found", f"no trace {trace_id!r}")
        return payload

    return app


# ---------------------------------------------------------------------------- helpers


class _JsonInternalErrors:
    """Answer an exception no handler anticipated with the JSON error body, inside CORS.

    A handler registered for `Exception` runs in Starlette's outermost middleware, outside
    `CORSMiddleware`, so its 500 carries no `Access-Control-Allow-Origin` header. The browser
    then withholds the body from the page, and the web app can only report the API as
    unreachable. Catching the exception here, below the CORS layer, gives the 500 its header.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def send_tracked(message: Message) -> None:
            nonlocal started
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, receive, send_tracked)
        except Exception as exc:
            if started:
                # Mid-response (a stream): a status line was sent, so no error body can follow.
                raise
            error = _engine_error(exc, "request")
            response = _error_response(error.status, error.kind, error.message)
            await response(scope, receive, send)


async def _sse(events: AsyncIterator[Event]) -> AsyncIterator[dict[str, str]]:
    async for event in events:
        # Payloads are JSON-native already; allow_nan=False makes a regression fail loudly
        # instead of sending `NaN`, which browsers' JSON.parse rejects.
        yield {"event": event.type, "data": json.dumps(event.data, allow_nan=False)}


def _concept_annotations(engines: Engines, query: str) -> dict[str, Any]:
    """Recognised concepts and the expanded query, shown beside the rankings.

    They need the `kb` concept dictionary, which an install that built only `pubmedqa` or
    `nfcorpus` does not have; the rankings are still valid then, so they are returned with
    `concepts_available: false` instead of being discarded. A `+concepts` retriever does need
    the dictionary and still fails with `no_index`.
    """
    try:
        concepts = engines.link_concepts(query)
        expanded = engines.expand_query(query)
    except CorpusNotBuilt:
        return {"concepts": [], "expanded_query": query, "concepts_available": False}
    return {
        "concepts": [c.model_dump(mode="json") for c in concepts],
        "expanded_query": expanded,
        "concepts_available": True,
    }


def _error_response(status: int, kind: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"kind": kind, "message": message}})


def _manifest(settings: Settings, corpus: str) -> dict[str, Any] | None:
    # The data layer writes manifest.json last, so its presence means the corpus is complete.
    path = settings.processed_dir / corpus / "manifest.json"
    return json.loads(path.read_text("utf-8")) if path.is_file() else None


def _passage_count(settings: Settings, corpus: str) -> int | None:
    manifest = _manifest(settings, corpus)
    return None if manifest is None else manifest.get("n_passages")


def _built_indexes(settings: Settings) -> dict[str, list[str]]:
    """`{corpus: [retriever component]}` for every index whose build finished (meta.json)."""
    return {
        corpus_dir.name: sorted(
            d.name for d in corpus_dir.iterdir() if d.is_dir() and (d / "meta.json").is_file()
        )
        for corpus_dir in sorted(settings.index_dir.glob("*"))
        if corpus_dir.is_dir()
    }


app = create_app()
