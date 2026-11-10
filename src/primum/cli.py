"""Command line interface (SPEC §3.10): build data and indexes, train, run experiments, ask, serve.

Every engine is imported inside the command that uses it, so `primum --help` and `primum spend`
start instantly instead of importing torch.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, help="Primum: selective, evidence-verified clinical QA.")
data_app = typer.Typer(no_args_is_help=True, help="Download sources and build the corpora.")
index_app = typer.Typer(no_args_is_help=True, help="Build retrieval indexes.")
train_app = typer.Typer(no_args_is_help=True, help="Train models.")
app.add_typer(data_app, name="data")
app.add_typer(index_app, name="index")
app.add_typer(train_app, name="train")

console = Console()


class CorpusChoice(StrEnum):
    pubmedqa = "pubmedqa"
    nfcorpus = "nfcorpus"
    kb = "kb"
    all = "all"


class Experiment(StrEnum):
    pubmedqa = "pubmedqa"
    rq1 = "rq1"
    rq2 = "rq2"
    rq3 = "rq3"
    report = "report"
    all = "all"


class Mode(StrEnum):
    chat = "chat"
    benchmark = "benchmark"


# Experiment name -> module in `primum.experiments`, in the order `exp all` runs them: the
# PubMedQA runs feed RQ2 and RQ3, and the report reads every results file.
EXPERIMENT_MODULES = {
    "pubmedqa": "run_pubmedqa",
    "rq1": "rq1_retrieval",
    "rq2": "rq2_confidence",
    "rq3": "rq3_conformal",
    "report": "report",
}


def _now() -> str:
    """The `created` stamp for provenance; taken once here so library code never reads the clock."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def _user_errors() -> Iterator[None]:
    """Report the failures a user fixes by running a command or editing `.env` as one line.

    A missing key, an exhausted budget and a missing corpus, index, encoder or results file
    each raise an error whose message names the fix. Anything else is a bug and keeps its
    traceback.
    """
    from primum.llm import BudgetExceeded, LLMUnavailable

    try:
        yield
    except (BudgetExceeded, LLMUnavailable, FileNotFoundError) as exc:
        console.print("[bold red]error:[/]", end=" ")
        console.print(str(exc), markup=False)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------- data


@data_app.command("build")
def data_build(
    corpus: Annotated[CorpusChoice, typer.Argument(help="Corpus to build, or 'all'.")],
    force: Annotated[bool, typer.Option(help="Rebuild even if already built.")] = False,
    max_distractors: Annotated[
        int | None, typer.Option(help="Cap on PubMedQA distractor abstracts (default: all).")
    ] = None,
) -> None:
    """Download the open sources and build `data/processed/<corpus>/`."""
    from primum.data import build_corpus

    names = ["pubmedqa", "nfcorpus", "kb"] if corpus == CorpusChoice.all else [corpus.value]
    for name in names:
        path = build_corpus(name, force=force, max_distractors=max_distractors)
        console.print(f"[green]built[/] {name} -> {path}")


@data_app.command("pairs")
def data_pairs(
    n_pubmedqa: Annotated[int, typer.Option(help="PubMedQA-artificial pairs to draw.")] = 40000,
    force: Annotated[bool, typer.Option(help="Rebuild even if already built.")] = False,
) -> None:
    """Build the contrastive training pairs (no PQA-L question or NFCorpus test query)."""
    from primum.data import build_contrastive_pairs

    console.print(f"[green]built[/] pairs -> {build_contrastive_pairs(n_pubmedqa, force=force)}")


# ---------------------------------------------------------------------------- index / train


@index_app.command("build")
def index_build(
    corpus: Annotated[str, typer.Argument(help="pubmedqa, nfcorpus or kb.")],
    retriever: Annotated[
        list[str] | None,
        typer.Option("--retriever", "-r", help="Retriever to build (repeatable). Default: bm25."),
    ] = None,
) -> None:
    """Build (or load and check) the indexes behind the named retrievers."""
    from primum.retrieval import build_retriever

    for name in retriever or ["bm25"]:
        started = time.perf_counter()
        with _user_errors():
            build_retriever(name, corpus)
        console.print(f"[green]ready[/] {name} on {corpus} ({time.perf_counter() - started:.1f} s)")


@train_app.command("encoder")
def train_encoder(
    epochs: Annotated[int, typer.Option(min=1)] = 1,
    batch_size: Annotated[int, typer.Option(min=2)] = 64,
    max_pairs: Annotated[
        int | None, typer.Option(help="Train on a seeded random sample of N pairs.")
    ] = None,
    lr: float = 2e-5,
    temperature: Annotated[float, typer.Option(help="InfoNCE temperature tau.")] = 0.05,
    hard_negatives: Annotated[
        bool, typer.Option(help="One round of BM25-mined negatives.")
    ] = False,
) -> None:
    """Fine-tune the bi-encoder contrastively; writes the model and results/encoder_training.json."""
    from primum.retrieval import train_encoder as train

    with _user_errors():
        path = train(
            epochs=epochs,
            batch_size=batch_size,
            max_pairs=max_pairs,
            lr=lr,
            temperature=temperature,
            hard_negatives=hard_negatives,
            created=_now(),
        )
    console.print(f"[green]trained[/] encoder -> {path}")


# ---------------------------------------------------------------------------- experiments


@app.command("exp")
def exp(
    name: Annotated[Experiment, typer.Argument(help="Experiment to run, or 'all'.")],
    limit: Annotated[
        int | None, typer.Option(min=1, help="pubmedqa, rq1: only the first N questions.")
    ] = None,
    repeats: Annotated[
        int | None, typer.Option(min=1, help="rq2, rq3: number of repeated splits.")
    ] = None,
) -> None:
    """Run an experiment script; each writes one file in results/."""
    names = list(EXPERIMENT_MODULES) if name == Experiment.all else [name.value]
    options = {k: v for k, v in {"limit": limit, "repeats": repeats}.items() if v is not None}
    modules = {
        n: importlib.import_module(f"primum.experiments.{EXPERIMENT_MODULES[n]}") for n in names
    }
    # Each option goes to the experiments whose `run` takes it; one that none of the selected
    # experiments takes is a mistake worth stopping for, not something to ignore silently.
    accepted = {
        n: set(inspect.signature(m.run).parameters) & set(options) for n, m in modules.items()
    }
    unused = set(options) - set().union(*accepted.values())
    if unused:
        raise typer.BadParameter(f"{name.value} does not take --{', --'.join(sorted(unused))}")
    created = _now()  # one stamp for the whole invocation, so an `all` run is identifiable
    for experiment, module in modules.items():
        console.print(f"[bold]{experiment}[/] ...")
        # A budget stop in `pubmedqa` has already written the completed records (marked
        # incomplete); the message says how to continue.
        with _user_errors():
            path = module.run(created=created, **{k: options[k] for k in accepted[experiment]})
        console.print(f"[green]wrote[/] {path}")


# ---------------------------------------------------------------------------- ask / serve / spend


@app.command("ask")
def ask(
    question: Annotated[str, typer.Argument(help="The question, in quotes.")],
    mode: Mode = Mode.chat,
    retriever: Annotated[
        str | None, typer.Option(help="Default: settings.default_retriever.")
    ] = None,
    alpha: Annotated[float | None, typer.Option(help="Target selective risk.")] = None,
) -> None:
    """Run the full pipeline on one question and print each stage as it completes."""
    ok = asyncio.run(_print_pipeline(question, mode.value, retriever, alpha))
    if not ok:
        raise typer.Exit(code=1)


async def _print_pipeline(
    question: str, mode: str, retriever: str | None, alpha: float | None
) -> bool:
    from primum.pipeline import run_pipeline

    async for event in run_pipeline(question, mode=mode, retriever=retriever, alpha=alpha):
        data = event.data
        match event.type:
            case "stage" if data["status"] == "done":
                console.print(f"[dim]  {data['name']:<11}{data['ms']:>9.1f} ms[/]")
            case "triage":
                style = {"emergency": "bold red", "urgent": "yellow"}.get(data["level"], "dim")
                console.print(f"[{style}]triage: {data['level']}[/]")
                if data["message"]:
                    console.print(data["message"], markup=False)
            case "concepts" if data["concepts"]:
                names = ", ".join(c["preferred"] for c in data["concepts"])
                console.print(f"concepts: {names}", markup=False)
            case "retrieval":
                console.print(f"evidence ({data['retriever']}):")
                for number, hit in enumerate(data["hits"], start=1):
                    passage = hit["passage"]
                    label = passage["title"] or passage["passage_id"]
                    line = f"  [{number}] {label} ({passage['source']}) score={hit['score']:.3f}"
                    console.print(line, markup=False, highlight=False)
            case "token":
                console.print(data["text"], end="", markup=False, highlight=False)
            case "answer":
                # Chat answers were already printed token by token.
                body = data["text"] if "decision" in data else ""
                console.print(body, markup=False, highlight=False)
            case "verdicts":
                colours = {"supported": "green", "contradicted": "red", "unsupported": "yellow"}
                for verdict in data["verdicts"]:
                    label = verdict["label"]
                    console.print(f"  [{colours[label]}]{label:<12}[/]", end="")
                    console.print(verdict["claim"]["text"], markup=False, highlight=False)
            case "confidence":
                _print_confidence(data)
            case "done":
                usage = data["usage"]
                console.print(f"[dim]trace {data['trace_id']}  cost ${usage['cost_usd']:.4f}[/]")
            case "error":
                console.print(f"[bold red]error ({data['kind']}) in {data['stage']}:[/]", end=" ")
                console.print(data["message"], markup=False)
                return False
    return True


def _print_confidence(data: dict) -> None:
    if data["uncalibrated"]:
        console.print(
            "[yellow]No deployment bundle: answer shown without a guarantee "
            "(run `primum exp rq3` to fit one).[/]"
        )
        return
    threshold = data["threshold"]
    verdict = "[green]released[/]" if data["answered"] else "[yellow]withheld[/]"
    gate = "none certified" if threshold is None else f"{threshold:.3f}"
    console.print(
        f"P(correct) = {data['p_correct']:.3f}, threshold at alpha={data['alpha']:.2f}: {gate} "
        f"-> {verdict}"
    )
    if not data["guarantee_applies"]:
        console.print(
            "[dim]The threshold is an operating point here: the certificate covers only the "
            "benchmark questions the deployment bundle held out, asked by id and answered by "
            "the calibrated pipeline.[/]"
        )


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Restart on code changes.")] = False,
) -> None:
    """Start the API (models load on first use or on GET /api/warmup)."""
    import uvicorn

    uvicorn.run("primum.api.app:app", host=host, port=port, reload=reload)


@app.command("spend")
def spend() -> None:
    """LLM spend recorded in the ledger, by tag, against the budget."""
    from primum.llm import get_llm

    summary = get_llm().spend()
    table = Table(title="LLM spend")
    table.add_column("tag")
    table.add_column("USD", justify="right")
    for tag, usd in sorted(summary["by_tag"].items()):
        table.add_row(tag, f"{usd:.4f}")
    table.add_row("[bold]total[/]", f"[bold]{summary['total_usd']:.4f}[/]")
    console.print(table)
    console.print(
        f"budget ${summary['budget_usd']:.2f}; {summary['calls']} calls, "
        f"{summary['cached_calls']} served from the cache"
    )


if __name__ == "__main__":
    app()
