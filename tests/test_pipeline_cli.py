"""CLI tests: command dispatch and the `ask` printer, with stub engines and fake experiments."""

from __future__ import annotations

import sys
import types
from datetime import datetime
from pathlib import Path

import pytest
from test_pipeline import EMERGENCY, ROUTINE, FakeLLM, make_engines
from typer.testing import CliRunner

import primum.llm
import primum.pipeline.orchestrator as orchestrator
from primum.cli import EXPERIMENT_MODULES, app
from primum.llm import BudgetExceeded

runner = CliRunner()


# The options each real experiment's `run` takes, as far as the CLI forwards them.
FAKE_OPTIONS = {
    "run_pubmedqa": "limit",
    "rq1_retrieval": "limit",
    "rq2_confidence": "repeats",
    "rq3_conformal": "repeats",
    "report": None,
}


@pytest.fixture
def fake_experiments(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict]]:
    """Install a stub module per experiment; returns the (module, created, options) calls."""
    calls: list[tuple[str, str, dict]] = []
    for module_name, option in FAKE_OPTIONS.items():
        module = types.ModuleType(f"primum.experiments.{module_name}")

        def record(name: str, created: str, **options) -> Path:
            calls.append((name, created, options))
            return Path(f"results/{name}.json")

        if option == "limit":
            module.run = lambda *, created, limit=None, _n=module_name: record(
                _n, created, **({} if limit is None else {"limit": limit})
            )
        elif option == "repeats":
            module.run = lambda *, created, repeats=None, _n=module_name: record(
                _n, created, **({} if repeats is None else {"repeats": repeats})
            )
        else:
            module.run = lambda *, created, _n=module_name: record(_n, created)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def test_fakes_cover_every_experiment() -> None:
    assert set(FAKE_OPTIONS) == set(EXPERIMENT_MODULES.values())


def test_exp_runs_one_module_with_an_iso_timestamp(fake_experiments) -> None:
    result = runner.invoke(app, ["exp", "rq1"])
    assert result.exit_code == 0, result.output
    ((module, created, options),) = fake_experiments
    assert module == "rq1_retrieval" and options == {}
    assert datetime.fromisoformat(created).tzinfo is not None


def test_exp_all_runs_every_experiment_in_dependency_order(fake_experiments) -> None:
    assert runner.invoke(app, ["exp", "all", "--repeats", "3"]).exit_code == 0
    assert [module for module, _, _ in fake_experiments] == [
        "run_pubmedqa",
        "rq1_retrieval",
        "rq2_confidence",
        "rq3_conformal",
        "report",
    ]
    assert len({created for _, created, _ in fake_experiments}) == 1
    # --repeats reaches only the experiments that take it.
    assert {m: o for m, _, o in fake_experiments if o} == {
        "rq2_confidence": {"repeats": 3},
        "rq3_conformal": {"repeats": 3},
    }


def test_exp_forwards_limit_and_rejects_an_option_it_cannot_use(fake_experiments) -> None:
    assert runner.invoke(app, ["exp", "pubmedqa", "--limit", "5"]).exit_code == 0
    ((module, _, options),) = fake_experiments
    assert module == "run_pubmedqa" and options == {"limit": 5}
    result = runner.invoke(app, ["exp", "report", "--limit", "5"])
    assert result.exit_code == 2 and len(fake_experiments) == 1


@pytest.mark.parametrize(
    "error",
    [
        BudgetExceeded("recorded spend has reached the budget; raise PRIMUM_BUDGET_USD"),
        FileNotFoundError("results/pubmedqa_runs.json does not exist; run `primum exp pubmedqa`"),
    ],
)
def test_exp_reports_a_fixable_failure_in_one_line(fake_experiments, monkeypatch, error) -> None:
    def fail(*, created, repeats=None):
        raise error

    monkeypatch.setattr(sys.modules["primum.experiments.rq2_confidence"], "run", fail)
    result = runner.invoke(app, ["exp", "rq2"])
    assert result.exit_code == 1
    assert "error:" in result.output and str(error).split(";")[0] in result.output
    assert "Traceback" not in result.output


def test_unknown_names_are_usage_errors() -> None:
    assert runner.invoke(app, ["exp", "rq9"]).exit_code == 2
    assert runner.invoke(app, ["data", "build", "wikipedia"]).exit_code == 2


def test_ask_prints_the_staged_pipeline(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator, "default_engines", lambda: make_engines(tmp_path))
    result = runner.invoke(app, ["ask", ROUTINE])
    assert result.exit_code == 0, result.output
    for text in (
        "triage: routine",
        "evidence",
        "[1]",
        "supported",
        "No deployment bundle",
        "trace",
    ):
        assert text in result.output


def test_ask_emergency_prints_the_message_and_nothing_clinical(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(orchestrator, "default_engines", lambda: make_engines(tmp_path))
    result = runner.invoke(app, ["ask", EMERGENCY])
    assert result.exit_code == 0
    assert "triage: emergency" in result.output and "evidence" not in result.output


def test_ask_exits_non_zero_on_a_pipeline_error(tmp_path, monkeypatch) -> None:
    engines = make_engines(tmp_path, llm=FakeLLM(fail=BudgetExceeded("budget reached")))
    monkeypatch.setattr(orchestrator, "default_engines", lambda: engines)
    result = runner.invoke(app, ["ask", ROUTINE])
    assert result.exit_code == 1
    assert "error (budget) in generate" in result.output and "budget reached" in result.output


def test_spend_prints_the_ledger_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(primum.llm, "get_llm", lambda: FakeLLM())
    result = runner.invoke(app, ["spend"])
    assert result.exit_code == 0 and "total" in result.output and "budget $1.00" in result.output
