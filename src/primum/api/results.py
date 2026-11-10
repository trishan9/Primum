"""Read-only access to experiment outputs in `results/` for the Experiments view and the explainer.

The API never computes a metric: it serves files the experiment scripts wrote, or says that
they have not been written. That is the integrity rule of SPEC §1 applied to the server.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

# Results file stem -> the command that produces it (quoted in "not run yet" messages).
EXPERIMENTS: dict[str, str] = {
    "rq1_retrieval": "primum exp rq1",
    "rq2_confidence": "primum exp rq2",
    "rq3_conformal": "primum exp rq3",
    "encoder_training": "primum train encoder",
    "pubmedqa_runs": "primum exp pubmedqa",
}


# Named runs: the `run` query value -> its subdirectory of `results/` ("" is `results/` itself).
# A fixed whitelist, so a request names a run but never a path. The main run holds every file;
# the others hold the generator-dependent files (pubmedqa_runs, rq2, rq3) of a second generator.
RUNS: dict[str, str] = {"main": "", "sonnet": "sonnet", "sonnet_sc300": "sonnet_sc300"}
DEFAULT_RUN = "main"


class ResultMissing(LookupError):
    """A results file (or the part of it a caller needs) does not exist yet."""


class UnknownRun(LookupError):
    """A `run` that is not in `RUNS`."""


def run_dir(results_dir: Path, run: str) -> Path:
    """The directory of a named run; `UnknownRun` for a name outside the whitelist."""
    if run not in RUNS:
        raise UnknownRun(f"unknown run {run!r}; expected one of {list(RUNS)}")
    return results_dir / RUNS[run] if RUNS[run] else results_dir


def experiments_index(results_dir: Path, run: str = DEFAULT_RUN) -> dict[str, Any]:
    """The files that exist for `run`, and every run that has any files.

    `{run, available: [...], files: {name: {updated, bytes}}, runs: [...]}`; each entry of `runs`
    is `run_info` for that run.
    """
    files = {}
    directory = run_dir(results_dir, run)
    for name in EXPERIMENTS:
        path = directory / f"{name}.json"
        if path.is_file():
            stat = path.stat()
            updated = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds")
            files[name] = {"updated": updated, "bytes": stat.st_size}
    runs = [info for name in RUNS if (info := run_info(results_dir, name)) is not None]
    for info in runs:
        info["subset_of"] = _subset_of(results_dir, info, runs)
    return {"run": run, "available": list(files), "files": files, "runs": runs}


def _subset_of(
    results_dir: Path, info: dict[str, Any], runs: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """How a run's questions relate to a larger run of the same generator, from the records.

    `{run, prefix, answered_in_both, first_sample_matches, failed_only_here}`: `prefix` is true
    when this run's questions are the first ones of the other run in dataset order;
    `first_sample_matches` counts questions answered in both whose (first-sample) decision
    agrees; `failed_only_here` counts questions that failed here but were answered there.
    None when no other run of the same generator covers all of this run's questions.
    """
    if "pubmedqa_runs" not in info["available"] or info["generator"] is None:
        return None
    mine = _records_of(run_dir(results_dir, info["run"]) / "pubmedqa_runs.json")
    for other in runs:
        if (
            other["run"] == info["run"]
            or other["generator"] != info["generator"]
            or "pubmedqa_runs" not in other["available"]
        ):
            continue
        theirs = _records_of(run_dir(results_dir, other["run"]) / "pubmedqa_runs.json")
        if len(mine[0]) >= len(theirs[0]) or not set(mine[0]) <= set(theirs[0]):
            continue
        both = [q for q in mine[1] if q in theirs[1]]
        return {
            "run": other["run"],
            "prefix": theirs[0][: len(mine[0])] == mine[0],
            "answered_in_both": len(both),
            "first_sample_matches": sum(mine[1][q] == theirs[1][q] for q in both),
            "failed_only_here": sum(q in theirs[1] for q in mine[2]),
        }
    return None


def _records_of(path: Path) -> tuple[tuple[str, ...], dict[str, Any], frozenset[str]]:
    """(`rag` qids in file order, qid -> decision of answered ones, failed qids) of a runs file."""
    stat = path.stat()
    return _records_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=8)
def _records_cached(
    path: str, _mtime_ns: int, _size: int
) -> tuple[tuple[str, ...], dict[str, Any], frozenset[str]]:
    data = json.loads(Path(path).read_text("utf-8"))
    records = data.get("records") if isinstance(data, dict) else None
    rag = [
        r
        for r in (records if isinstance(records, list) else [])
        if isinstance(r, dict) and r.get("condition", "rag") == "rag" and "qid" in r
    ]
    answered = {r["qid"]: r.get("decision") for r in rag if "error" not in r}
    failed = frozenset(r["qid"] for r in rag if "error" in r)
    return tuple(r["qid"] for r in rag), answered, failed


def run_info(results_dir: Path, run: str) -> dict[str, Any] | None:
    """What a run is, read from its files' `meta`; None when the run has no results files.

    `generator` is `meta.models.llm`; `n` (questions attempted), `self_consistency_samples`,
    `conditions` and `n_failed` (per condition) come from `pubmedqa_runs.json` when it exists,
    else `n` and `n_failed` (as `{"rag": ...}`) from the RQ2 or RQ3 file. No metric is computed;
    `experiments_index` adds `subset_of`, counted from the records.
    """
    directory = run_dir(results_dir, run)
    available = [name for name in EXPERIMENTS if (directory / f"{name}.json").is_file()]
    if not available:
        return None
    info: dict[str, Any] = {
        "run": run,
        "generator": None,
        "n": None,
        "self_consistency_samples": None,
        "conditions": None,
        "n_failed": None,
        "available": available,
    }
    for name in ("pubmedqa_runs", "rq2_confidence", "rq3_conformal"):
        if name not in available:
            continue
        meta, failed = _meta_of(directory / f"{name}.json")
        models = meta.get("models")
        info["generator"] = models.get("llm") if isinstance(models, dict) else None
        info["n"] = meta.get("n")
        if name == "pubmedqa_runs":
            info["self_consistency_samples"] = meta.get("self_consistency_samples")
            info["conditions"] = meta.get("conditions")
            info["n_failed"] = failed
        elif "n_failed_excluded" in meta:
            info["n_failed"] = {"rag": meta["n_failed_excluded"]}
        break
    return info


def experiment_generator(results_dir: Path, name: str) -> str | None:
    """`meta.models.llm` of an existing results file, or None when it records none."""
    meta, _ = _meta_of(experiment_path(results_dir, name))
    models = meta.get("models")
    llm = models.get("llm") if isinstance(models, dict) else None
    return llm if isinstance(llm, str) else None


def _meta_of(path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """`meta`, and per-condition `n_failed` from `summary.pubmedqa`, of a results file.

    Cached on the file's modification time and size: `pubmedqa_runs.json` is several megabytes
    (it holds every per-question record) and the index is requested on every page load.
    """
    stat = path.stat()
    return _meta_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=32)
def _meta_cached(
    path: str, _mtime_ns: int, _size: int
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    data = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(data, dict):
        return {}, None
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    summary = data.get("summary")
    pubmedqa = summary.get("pubmedqa") if isinstance(summary, dict) else None
    failed = None
    if isinstance(pubmedqa, dict):
        failed = {
            condition: block["n_failed"]
            for condition, block in pubmedqa.items()
            if isinstance(block, dict) and "n_failed" in block
        }
    return meta, failed


def experiment_path(results_dir: Path, name: str) -> Path:
    """Path of an existing results file; `KeyError` for an unknown name, `ResultMissing` if absent."""
    if name not in EXPERIMENTS:
        raise KeyError(f"unknown experiment {name!r}; expected one of {list(EXPERIMENTS)}")
    path = results_dir / f"{name}.json"
    if not path.is_file():
        raise ResultMissing(f"{name} has not been run yet; run `{EXPERIMENTS[name]}`")
    return path


def calibration_data(
    results_dir: Path, *, thresholds: dict[str, float | None] | None, delta: float | None
) -> dict[str, Any]:
    """Per-item `{p_correct, correct}` from the RQ3 results, plus certified thresholds.

    The top-level arrays are the rows the thresholds were certified on (held out from the
    model fit, not from the threshold choice); `heldout`, when present, holds rows held out
    from both.

    RQ3 owns the layout of its file; what this needs from it is one object holding two equal-
    length arrays named `p_correct` and `correct`, at any depth outside `meta`. `thresholds`
    and `delta` should come from the deployment bundle (the rule the live system applies);
    when it does not exist they are taken from that same object or the file's `meta`.
    """
    results = json.loads(experiment_path(results_dir, "rq3_conformal").read_text("utf-8"))
    found = _find_calibration(results)
    if found is None:
        raise ResultMissing(
            "rq3_conformal.json holds no per-item {p_correct, correct} arrays; re-run "
            "`primum exp rq3`"
        )
    meta = results.get("meta") if isinstance(results, dict) else None
    if thresholds is None:
        thresholds = found.get("thresholds") or {}
    if delta is None:
        delta = found.get("delta", (meta or {}).get("delta"))
    payload = {
        "p_correct": [float(p) for p in found["p_correct"]],
        "correct": [int(bool(c)) for c in found["correct"]],
        "thresholds": thresholds,
        "delta": delta,
        "n": len(found["p_correct"]),
    }
    # The certification settings, when RQ3 recorded them: a client that recomputes the
    # certificate from these rows needs the same candidate grid and minimum selection size
    # to arrive at the same thresholds. `heldout` holds rows nothing was fitted or certified
    # on, where coverage and risk at a threshold are out-of-sample.
    for key in ("n_grid", "min_selected", "heldout", "source"):
        if key in found:
            payload[key] = found[key]
    return payload


def _find_calibration(node: Any) -> dict[str, Any] | None:
    """Depth-first search for the first object with equal-length `p_correct`/`correct` lists."""
    if isinstance(node, dict):
        p, c = node.get("p_correct"), node.get("correct")
        if isinstance(p, list) and isinstance(c, list) and p and len(p) == len(c):
            return node
        children = [value for key, value in node.items() if key != "meta"]
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        if (found := _find_calibration(child)) is not None:
            return found
    return None
