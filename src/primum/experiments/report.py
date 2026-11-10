"""Figures and `docs/RESULTS.md`, generated only from the JSON files in `results/`.

Every number in the output is read from a results file written by an experiment; nothing is
typed here. A missing file is reported as "not run" with the command that produces it, and its
figures are not drawn.

Figure style (one look across all figures, meant for a paper): a single sans face at small
sizes; hairline solid gridlines one step off the background; left and bottom axes only; 1.5pt
lines and small round markers with a background-coloured ring so overlapping marks stay
legible; categorical colours assigned in a fixed order (method i always gets colour i), with a
distinct marker per series as a second channel so no identity depends on colour alone; and a
legend whenever there are two or more series. Each figure is written as PNG and SVG, and
every plotted value also appears in a table in RESULTS.md.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from primum.config import Settings, get_settings
from primum.experiments._common import read_json, resolve_results_dir

# Light reference palette (validated with the dataviz skill's palette checker: adjacent-pair
# CVD and normal-vision separation pass for the first five slots; three slots are below 3:1
# contrast on the background, which is why every figure has a table twin in RESULTS.md).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")

STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "medium",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.labelsize": 8.5,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "xtick.color": AXIS,
    "ytick.color": AXIS,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.labelcolor": INK_2,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
    "svg.fonttype": "none",
}

FILES = {
    "pubmedqa_runs": "primum exp pubmedqa",
    "encoder_training": "primum train encoder",
    "rq1_retrieval": "primum exp rq1",
    "rq2_confidence": "primum exp rq2",
    "rq3_conformal": "primum exp rq3",
}
RQ3_METHODS = ("certified", "certified_hb", "plug_in")
# A second generator's results live in subdirectories of `results/`: the full run (all questions)
# and an optional subset run with several samples per question (self-consistency).
SECOND_GENERATOR_TITLE = "Second generator: Claude Sonnet 5.5"
SECOND_GENERATOR_DIRS = ("sonnet", "sonnet_sc300")
SC_SIGNALS = ("signal:sc_agreement", "signal:sc_entropy")


def run(*, created: str, results_dir: Path | None = None, docs_path: Path | None = None) -> Path:
    """Draw `results/figures/*.{png,svg}` and write `docs/RESULTS.md`; returns the latter."""
    settings = get_settings()
    directory = resolve_results_dir(results_dir)
    docs = docs_path or settings.root / "docs" / "RESULTS.md"
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, dict[str, Any] | None] = {}
    for name in FILES:
        path = directory / f"{name}.json"
        loaded[name] = read_json(path) if path.exists() else None
        if loaded[name] is None:
            print(f"[report] skipping {path.name}: not found (run `{FILES[name]}`)")

    second: dict[str, dict[str, Any] | None] = {}
    for sub in SECOND_GENERATOR_DIRS:
        for name in FILES:
            path = directory / sub / f"{name}.json"
            if path.exists():
                second[f"{sub}/{name}"] = read_json(path)

    report = _Report(figures, docs.parent)
    with matplotlib.rc_context(STYLE):
        sections = [
            _header(created, {**loaded, **second}),
            _section(loaded, "pubmedqa_runs", "PubMedQA benchmark runs", report, _pubmedqa),
            _section(loaded, "rq1_retrieval", "RQ1: retrieval", report, _rq1),
            _section(loaded, "encoder_training", "Encoder fine-tuning", report, _encoder),
            _section(loaded, "rq2_confidence", "RQ2: confidence signals", report, _rq2),
            _section(loaded, "rq3_conformal", "RQ3: certified abstention", report, _rq3),
            _second_generator(loaded, second, report, settings),
        ]
    docs.parent.mkdir(parents=True, exist_ok=True)
    docs.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
    return docs


# --------------------------------------------------------------------------------------------
# Document structure
# --------------------------------------------------------------------------------------------


class _Report:
    """Where figures go and how RESULTS.md links to them."""

    def __init__(self, figures: Path, docs_dir: Path) -> None:
        self.figures = figures
        self.docs_dir = docs_dir

    def save(self, fig: Figure, name: str, alt: str) -> str:
        """Write PNG and SVG; return the Markdown image link (relative to RESULTS.md)."""
        for suffix in ("png", "svg"):
            fig.savefig(self.figures / f"{name}.{suffix}")
        link = os.path.relpath(self.figures / f"{name}.svg", self.docs_dir)
        return f"![{alt}]({Path(link).as_posix()})"


def _header(created: str, loaded: dict[str, dict[str, Any] | None]) -> str:
    rows = [
        [f"`results/{name}.json`", _meta(data, "created") if data else "not run", _meta(data, "n")]
        for name, data in loaded.items()
    ]
    return "\n".join(
        [
            "# Results",
            "",
            f"Generated by `primum exp report` on {created} from the JSON files in `results/`. "
            "Every number on this page is read from those files; none is typed by hand. "
            "Do not edit this file: re-run the experiments and the report instead.",
            "",
            "Intervals are 95%. Their kind differs by quantity and is stated in each section: "
            "bootstrap intervals over questions or queries, the central range over repeated "
            "random splits, or exact Clopper-Pearson intervals.",
            "",
            "Created is the `created` value recorded in each file's `meta`: the time of the run "
            "when started from the CLI, or a fixed label when the caller passed one, so it does "
            "not order the files in time.",
            "",
            _table(["File", "Created", "n"], rows),
        ]
    )


def _section(loaded, name: str, title: str, report: _Report, body) -> str:
    data = loaded[name]
    if data is None:
        return (
            f"## {title}\n\nNot run: `results/{name}.json` does not exist. "
            f"Run `{FILES[name]}` to produce it."
        )
    return f"## {title}\n\n" + body(data, report)


def _meta(data: dict[str, Any] | None, key: str) -> str:
    value = (data or {}).get("meta", {}).get(key)
    return "" if value is None else str(value)


# --------------------------------------------------------------------------------------------
# PubMedQA runs
# --------------------------------------------------------------------------------------------


def _pubmedqa(data: dict[str, Any], report: _Report) -> str:
    meta = data["meta"]
    summary = data["summary"]["pubmedqa"]
    conditions = list(summary)
    out = [
        f"{meta['n']} questions; generator `{meta['models']['llm']}`, retriever "
        f"`{meta['models']['retriever']}` (top {meta['top_k']}), "
        f"{meta['self_consistency_samples']} samples per `rag` question.",
    ]
    if not meta.get("complete", False):
        out.append(f"**This run stopped early ({meta.get('stopped')}); the figures are partial.**")
    rows = []
    for condition, entry in summary.items():
        rows.append(
            [
                condition,
                str(entry["n"]),
                str(entry["n_failed"]),
                _fmt(entry.get("accuracy")),
                _fmt(entry.get("macro_f1")),
                _pct(entry.get("supported_claim_rate")),
                _pct(entry.get("gold_in_hits_rate")),
            ]
        )
    out.append(
        _table(
            [
                "Condition",
                "n",
                "Failed",
                "Accuracy",
                "Macro-F1",
                "Supported claims",
                "Gold in evidence",
            ],
            rows,
        )
    )
    out.append("Accuracy and macro-F1 (yes/no/maybe) with 95% bootstrap intervals over questions.")

    tests = data.get("paired_tests", {})
    if tests:
        out.append(
            _table(
                ["Comparison", "Pairs", "Accuracy difference", "p (paired permutation)"],
                [
                    [
                        name.replace("_", " "),
                        str(t["n_pairs"]),
                        _fmt(t["accuracy_diff"]),
                        _p(t["p_value"]),
                    ]
                    for name, t in tests.items()
                ],
            )
        )

    fig = Figure(figsize=(6.4, 0.5 + 0.32 * len(conditions)), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    for ax, metric, label in zip(
        axes, ("accuracy", "macro_f1"), ("Accuracy", "Macro-F1"), strict=True
    ):
        _forest(ax, conditions, [summary[c].get(metric) for c in conditions])
        ax.set_title(label)
    out.append(report.save(fig, "pubmedqa_accuracy", "Accuracy and macro-F1 by condition"))
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------------
# RQ1
# --------------------------------------------------------------------------------------------

_DATASET_TITLES = {
    "nfcorpus": "NFCorpus test queries",
    "pubmedqa": "PubMedQA gold-abstract retrieval",
}


def _rq1(data: dict[str, Any], report: _Report) -> str:
    meta = data["meta"]
    out = []
    if meta.get("skipped"):
        skipped = "; ".join(f"`{s}` ({why})" for s, why in meta["skipped"].items())
        out.append(f"Systems not evaluated: {skipped}.")
    for dataset, title in _DATASET_TITLES.items():
        table = data.get(dataset)
        if not table:
            continue
        systems = list(table)
        metrics = list(next(iter(table.values())))
        out.append(f"### {title}\n\n{meta['n_queries'][dataset]} queries.")
        out.append(
            _table(
                ["System", *metrics],
                [[f"`{s}`", *(_fmt(table[s][m]) for m in metrics)] for s in systems],
            )
        )
        out.append("Mean over queries with 95% bootstrap intervals over queries.")
        for comparison, family in data.get("paired_tests", {}).get(dataset, {}).items():
            baseline = comparison.removeprefix("vs_")
            rows = [
                [
                    f"`{s}`",
                    *(
                        f"{_fmt(entry[f'{m}_diff'])}, p = {_p(entry[f'{m}_p_holm'])}"
                        for m in metrics
                    ),
                ]
                for s, entry in family.items()
            ]
            out.append(
                f"Difference from `{baseline}` (paired by query; Holm-adjusted permutation p):\n\n"
                + _table(["System", *metrics], rows)
            )

        fig = Figure(
            figsize=(2.3 * len(metrics) + 1.2, 0.5 + 0.3 * len(systems)), layout="constrained"
        )
        axes = np.atleast_1d(fig.subplots(1, len(metrics), sharey=True))
        for ax, metric in zip(axes, metrics, strict=True):
            _forest(ax, systems, [table[s][metric] for s in systems])
            ax.set_title(metric)
        out.append(report.save(fig, f"rq1_{dataset}", f"{title}: metrics by retriever"))
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------------
# Encoder training
# --------------------------------------------------------------------------------------------


def _encoder(data: dict[str, Any], report: _Report) -> str:
    out = [
        f"{data.get('n_train_pairs')} training pairs from {data.get('n_train_queries')} queries, "
        f"{data.get('steps')} steps on `{data.get('device')}` in {data.get('wall_time_s')} s."
    ]
    dev = data.get("dev")
    if dev:
        keys = [k for k in dev["before"] if k not in ("n_queries", "pool_size")]
        out.append(
            f"Held-out training-distribution check ({dev['before']['n_queries']} queries, "
            f"pool of {dev['before']['pool_size']} passages; a sanity check, not a benchmark):\n\n"
            + _table(
                ["", *keys],
                [[when, *(f"{dev[when][k]:.3f}" for k in keys)] for when in ("before", "after")],
            )
        )
    curve = data.get("loss_curve") or []
    if len(curve) >= 2:
        fig = Figure(figsize=(4.8, 2.6), layout="constrained")
        ax = fig.subplots()
        steps = [point["step"] for point in curve]
        ax.plot(steps, [point["loss"] for point in curve], color=SERIES[0])
        chance = data.get("loss_at_chance")
        if chance is not None:
            ax.axhline(chance, color=MUTED, linewidth=0.8)
            ax.annotate(
                "uniform scores",
                (steps[-1], chance),
                textcoords="offset points",
                xytext=(0, 3),
                ha="right",
                color=INK_2,
                fontsize=7.5,
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("InfoNCE loss")
        ax.set_title("Contrastive training loss")
        out.append(report.save(fig, "encoder_loss", "Contrastive training loss by step"))
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------------
# RQ2
# --------------------------------------------------------------------------------------------


def _rq2(data: dict[str, Any], report: _Report) -> str:
    meta = data["meta"]
    main_cal = meta["main_calibrator"]
    by_cal = data["estimators"]
    estimators = list(by_cal[main_cal])
    metrics = ["auroc", "aurc", "e_aurc", "ece", "brier"]
    out = [
        f"{meta['n']} `rag` answers ({meta['n_correct']} correct, accuracy "
        f"{meta['accuracy']:.3f}; {meta['n_failed_excluded']} failed generations excluded). "
        f"{meta['seeds']['repeats']} repeated stratified splits (train / calibration / test = "
        f"{meta['split_fractions']['train']:.0%} / {meta['split_fractions']['calibration']:.0%} / "
        f"{meta['split_fractions']['test']:.0%}). Values are the mean over repeats with the "
        "central 95% range of the repeats in brackets.",
        f"### All estimators, {main_cal} calibration",
        _table(
            ["Estimator", "AUROC", "AURC", "E-AURC", "ECE", "Brier"],
            [[f"`{e}`", *(_fmt(by_cal[main_cal][e][m]) for m in metrics)] for e in estimators],
        ),
        "### ECE by calibrator",
        _table(
            ["Estimator", *by_cal],
            [[f"`{e}`", *(_fmt(by_cal[c][e]["ece"]) for c in by_cal)] for e in estimators],
        ),
        f"### Feature-group ablation ({main_cal} calibration)",
        _table(
            ["Model", "AUROC", "AURC", "E-AURC", "ECE", "Brier"],
            [[name, *(_fmt(row[m]) for m in metrics)] for name, row in data["ablation"].items()],
        ),
    ]
    for comparison, block in data.get("comparisons", {}).items():
        reference = comparison.removeprefix("vs_")
        out.append(_comparison_table(block, reference, main_cal))
    out.append(f"Best single signal by mean AUROC: `{meta['best_single_signal']}`.")

    # Discrimination and calibration of every estimator; learned models are the emphasis.
    fig = Figure(figsize=(6.6, 0.6 + 0.24 * len(estimators)), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    learned = [e for e in estimators if e.split(":")[0] in ("logistic", "mlp")]
    colours = [SERIES[0] if e in learned else MUTED for e in estimators]
    for ax, metric, label in zip(axes, ("auroc", "ece"), ("AUROC", "ECE"), strict=True):
        _forest(ax, estimators, [by_cal[main_cal][e][metric] for e in estimators], colours)
        ax.set_title(label)
    axes[0].axvline(0.5, color=AXIS, linewidth=0.8, zorder=0)
    _legend(fig, [("Learned combination", SERIES[0], "o"), ("Single signal / stated", MUTED, "o")])
    out.append(report.save(fig, "rq2_estimators", "AUROC and ECE by confidence estimator"))

    reliability = data.get("reliability", {})
    if reliability:
        names = list(reliability)
        cols = 3 if len(names) > 4 else min(2, len(names))
        rows = int(np.ceil(len(names) / cols))
        fig = Figure(figsize=(2.3 * cols, 2.2 * rows), layout="constrained")
        axes = np.atleast_1d(fig.subplots(rows, cols, sharex=True, sharey=True)).ravel()
        for i, (ax, name) in enumerate(zip(axes, names, strict=False)):
            bins = reliability[name]
            ax.plot([0, 1], [0, 1], color=AXIS, linewidth=0.8, zorder=0)
            ax.plot(
                [b["confidence"] for b in bins],
                [b["accuracy"] for b in bins],
                color=SERIES[i % len(SERIES)],
                marker=MARKERS[i % len(MARKERS)],
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
            )
            ax.set_title(name, fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        for ax in axes[len(names) :]:
            ax.set_visible(False)
        fig.supxlabel("Mean predicted P(correct)", fontsize=8.5, color=INK_2)
        fig.supylabel("Observed accuracy", fontsize=8.5, color=INK_2)
        out.append(
            report.save(fig, "rq2_reliability", "Reliability diagrams of the main estimators")
        )
        out.append(
            _table(
                ["Estimator", "Bins (mean confidence → accuracy, count)"],
                [
                    [
                        f"`{n}`",
                        "; ".join(
                            f"{b['confidence']:.2f} → {b['accuracy']:.2f} ({b['count']})"
                            for b in reliability[n]
                        ),
                    ]
                    for n in names
                ],
            )
        )

    curves = data.get("risk_coverage", {})
    if curves:
        fig = Figure(figsize=(4.8, 3.0), layout="constrained")
        ax = fig.subplots()
        for i, (name, curve) in enumerate(curves.items()):
            ax.plot(
                curve["coverage"],
                curve["risk"],
                color=SERIES[i % len(SERIES)],
                marker=MARKERS[i % len(MARKERS)],
                markevery=10,
                markeredgecolor=SURFACE,
                markeredgewidth=1.0,
                label=name,
            )
        ax.set_xlabel("Coverage (share of questions answered)")
        ax.set_ylabel("Selective risk (error rate)")
        ax.set_xlim(0, 1)
        ax.set_title("Risk-coverage, mean over repeats")
        ax.legend(loc="upper left")
        out.append(report.save(fig, "rq2_risk_coverage", "Risk-coverage curves"))
        grid = (0.2, 0.4, 0.6, 0.8, 1.0)
        out.append(
            _table(
                ["Estimator", *(f"risk at {c:.0%}" for c in grid)],
                [
                    [f"`{n}`", *(f"{_at(c['coverage'], c['risk'], g):.3f}" for g in grid)]
                    for n, c in curves.items()
                ],
            )
        )
    return "\n\n".join(out)


# --------------------------------------------------------------------------------------------
# RQ3
# --------------------------------------------------------------------------------------------


def _rq3(data: dict[str, Any], report: _Report) -> str:
    meta = data["meta"]
    delta = meta["delta"]
    alphas = [f"{a:.2f}" for a in meta["alphas"]]
    out = [
        f"{meta['n']} `rag` answers, {meta['seeds']['repeats']} repeated splits, "
        f"n_cal = {meta['n_cal']}, δ = {delta}, min_selected = "
        f"{meta['min_selected']['value']} ({meta['min_selected']['fraction_of_n_cal']:.0%} of "
        f"n_cal). Confidence: `{meta['models']['confidence']}` with "
        f"`{meta['models']['calibrator']}` calibration, both fitted on the training split only.",
        f"Guarantee: {meta['guarantee']}",
        "Violation rate = share of repeats whose test selective risk exceeds α, with a "
        "Clopper-Pearson interval over repeats. Test risk is itself estimated on the test "
        "split, so near the boundary this rate overstates the probability that the true risk "
        "exceeds α. Coverage and risk: mean over repeats [central 95% range]; risk is "
        "averaged over the repeats that answered at least one question.",
    ]
    out.append(_selective_table(data["selective"], alphas))
    if data.get("min_selected_sensitivity"):
        out.append("Sensitivity to `min_selected` (certified, binomial):")
        out.append(_selective_table(data["min_selected_sensitivity"], alphas))
    if data.get("single_signal"):
        out.append("Each single signal as the confidence, same certified procedure:")
        out.append(_selective_table(data["single_signal"], alphas))
    sets = data.get("prediction_sets", {})
    if sets:
        rows = [
            [
                f"{method.upper()} {metric.replace('_', ' ')}",
                *(_fmt(block[a]) for a in alphas),
            ]
            for method, by_metric in sets.items()
            for metric, block in by_metric.items()
        ]
        out.append(
            "Conformal prediction sets over the stated yes/no/maybe probabilities "
            "(target coverage 1 − α):\n\n" + _table(["", *(f"α = {a}" for a in alphas)], rows)
        )
    deployment = data.get("deployment")
    if deployment:
        out.append(
            f"Deployment bundle ({'saved' if deployment['saved'] else 'not saved'}): trained on "
            f"{deployment['n_train']} answers, certified on {deployment['n_cal']}; thresholds "
            f"{_thresholds(deployment)}."
        )

    methods = [m for m in RQ3_METHODS if m in data["selective"]]
    x = np.array([float(a) for a in alphas])
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2) * 0.006
    fig = Figure(figsize=(6.6, 2.8), layout="constrained")
    ax_v, ax_c = fig.subplots(1, 2)
    for i, method in enumerate(methods):
        rows = [data["selective"][method][a] for a in alphas]
        style = {
            "color": SERIES[i],
            "marker": MARKERS[i],
            "markeredgecolor": SURFACE,
            "markeredgewidth": 1.0,
        }
        rate = np.array([r["violation_rate"] for r in rows])
        ci = np.array([r["violation_ci"] for r in rows])
        ax_v.errorbar(
            x + offsets[i],
            rate,
            yerr=[np.clip(rate - ci[:, 0], 0, None), np.clip(ci[:, 1] - rate, 0, None)],
            linewidth=1.5,
            elinewidth=1.0,
            capsize=0,
            label=method,
            **style,
        )
        coverage = [r["coverage"] for r in rows]
        mean = np.array([c["mean"] for c in coverage])
        lo, hi = np.array([c["ci"][0] for c in coverage]), np.array([c["ci"][1] for c in coverage])
        # The interval is the 2.5-97.5 percentile spread of per-split coverage, not a CI of
        # the mean: when most splits certify nothing (coverage 0) the mean can sit above the
        # upper percentile, so each side is clipped at zero length instead of going negative.
        ax_c.errorbar(
            x + offsets[i],
            mean,
            yerr=[np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)],
            linewidth=1.5,
            elinewidth=1.0,
            capsize=0,
            label=method,
            **style,
        )
    ax_v.axhline(delta, color=INK_2, linewidth=0.8, zorder=0)
    ax_v.annotate(
        f"δ = {delta}",
        (x[0], delta),
        textcoords="offset points",
        xytext=(-12, 3),
        ha="left",
        color=INK_2,
        fontsize=7.5,
    )
    for ax in (ax_v, ax_c):
        ax.set_xticks(x, alphas)
        ax.set_xlabel("Target risk α")
        ax.set_ylim(bottom=0)
    ax_v.set_ylabel("P(test risk > α)")
    ax_v.set_title("Violation rate")
    ax_c.set_ylabel("Share of questions answered")
    ax_c.set_title("Coverage")
    fig.legend(*ax_c.get_legend_handles_labels(), loc="outside upper center", ncols=len(methods))
    out.append(report.save(fig, "rq3_selective", "Violation rate and coverage by alpha"))
    return "\n\n".join(out)


def _selective_table(block: dict[str, Any], alphas: Sequence[str]) -> str:
    rows = []
    for method, by_alpha in block.items():
        for a in alphas:
            r = by_alpha[a]
            lo, hi = r["violation_ci"]
            rows.append(
                [
                    f"`{method}`",
                    a,
                    f"{r['violation_rate']:.1%} [{lo:.1%}, {hi:.1%}]",
                    _fmt(r["coverage"]),
                    _fmt(r["risk"]),
                    _pct(r["certified_rate"]),
                ]
            )
    return _table(
        ["Method", "α", "Violation rate", "Coverage", "Selective risk", "Certified in"], rows
    )


# --------------------------------------------------------------------------------------------
# Second generator
# --------------------------------------------------------------------------------------------


def _second_generator(
    loaded: dict[str, dict[str, Any] | None],
    second: dict[str, dict[str, Any] | None],
    report: _Report,
    settings: Settings,
) -> str:
    """The second generator's runs beside the main ones. Every comparison here is descriptive."""
    full_dir, subset_dir = SECOND_GENERATOR_DIRS
    title = f"## {SECOND_GENERATOR_TITLE}"
    runs = second.get(f"{full_dir}/pubmedqa_runs")
    if runs is None:
        return (
            f"{title}\n\nNot run: `results/{full_dir}/pubmedqa_runs.json` does not exist; "
            "this section is generated when it does."
        )
    main_runs = loaded["pubmedqa_runs"]
    subset_runs = second.get(f"{subset_dir}/pubmedqa_runs")
    out = [_second_intro(main_runs, runs)]

    # (1) Benchmark: the `rag` condition of each run.
    subset_label = "subset"
    if subset_runs is not None:
        sub_q = _rag_qids(subset_runs)
        if sub_q == _rag_qids(runs)[: len(sub_q)]:
            subset_label = f"first {len(sub_q)} (not random)"
    rows = []
    for label, data in (
        ("all", main_runs),
        ("all", runs),
        (subset_label, subset_runs),
    ):
        entry = (data or {}).get("summary", {}).get("pubmedqa", {}).get("rag")
        if entry is None:
            continue
        rows.append(
            [
                f"`{data['meta']['models']['llm']}`",
                label,
                str(data["meta"]["self_consistency_samples"]),
                str(entry["n"]),
                str(entry["n_failed"]),
                _fmt(entry.get("accuracy")),
                _fmt(entry.get("macro_f1")),
                _pct(entry.get("supported_claim_rate")),
                _pct(entry.get("gold_in_hits_rate")),
            ]
        )
    if rows:
        out.append("### `rag` benchmark")
        out.append(
            _table(
                [
                    "Generator",
                    "Questions",
                    "Samples",
                    "n",
                    "Failed",
                    "Accuracy",
                    "Macro-F1",
                    "Supported claims",
                    "Gold in evidence",
                ],
                rows,
            )
        )
        out.append(
            "n counts answered questions (failed generations excluded). Accuracy and macro-F1 "
            "with 95% bootstrap intervals over questions, each within its own run."
        )

    # (2) RQ2, both generators on all questions.
    rq2_runs = _present(loaded["rq2_confidence"], second.get(f"{full_dir}/rq2_confidence"))
    if rq2_runs:
        out.append("### RQ2: confidence signals")
        rows = []
        for data in rq2_runs:
            best = data["meta"]["best_single_signal"]
            names = _unique(["stated:verbal_conf", best, "logistic:deploy"])
            rows += _estimator_rows(data, names, best)
        calibrators = sorted({d["meta"]["main_calibrator"] for d in rq2_runs})
        out.append(
            _table(["Generator", "n", "Estimator", "AUROC", "ECE", "Brier"], rows)
            + f"\n\n{' / '.join(calibrators)} calibration; mean over repeated splits with the "
            "central 95% range of the repeats in brackets. Best single signal by mean AUROC is "
            "marked (best)."
        )
        for data in rq2_runs:
            out.append(_unavailable(data))
        second_rq2 = second.get(f"{full_dir}/rq2_confidence")
        if second_rq2:
            block = second_rq2.get("comparisons", {}).get("vs_stated:verbal_conf")
            if block:
                out.append(
                    f"`{_llm(second_rq2)}`: "
                    + _comparison_table(
                        block, "stated:verbal_conf", second_rq2["meta"]["main_calibrator"]
                    )
                )

    # (3) RQ2 on the subset with several samples per question.
    subset_rq2 = second.get(f"{subset_dir}/rq2_confidence")
    if subset_rq2 is None:
        out.append(
            f"Self-consistency subset: `results/{subset_dir}/rq2_confidence.json` does not exist."
        )
    else:
        out.append(
            _second_subset(subset_rq2, subset_runs, runs, second.get(f"{full_dir}/rq2_confidence"))
        )

    # (4) RQ3, both generators.
    rq3_runs = _present(loaded["rq3_conformal"], second.get(f"{full_dir}/rq3_conformal"))
    if rq3_runs:
        out.append(_second_rq3(rq3_runs, settings))
        out.append(_generators_figure(rq3_runs, report))
    return f"{title}\n\n" + "\n\n".join(out)


def _second_intro(main_runs: dict[str, Any] | None, runs: dict[str, Any]) -> str:
    meta = runs["meta"]
    llm, k = meta["models"]["llm"], meta["self_consistency_samples"]
    conditions = ", ".join(f"`{c}`" for c in meta["conditions"])
    text = [
        f"Generator `{llm}`: {meta['n']} questions, condition(s) {conditions}, {k} "
        f"sample{'s' if k != 1 else ''} per question, retriever `{meta['models']['retriever']}` "
        f"(top {meta['top_k']})."
    ]
    if main_runs is not None:
        main = main_runs["meta"]
        text.append(
            f"The main generator `{main['models']['llm']}` answered {main['n']} questions with "
            f"{main['self_consistency_samples']} samples per `rag` question, retriever "
            f"`{main['models']['retriever']}` (top {main['top_k']})."
        )
        same = _rag_qids(main_runs) == _rag_qids(runs)
        text.append(
            "Both runs cover the same questions."
            if same
            else "The two runs do not cover the same questions."
        )
        missing = [c for c in main["conditions"] if c not in meta["conditions"]]
        if missing:
            text.append(
                f"`{llm}` was not run in {' or '.join(f'`{c}`' for c in missing)}, so there is "
                "no such comparison for it."
            )
    if k == 1:
        text.append(
            "With one sample per question its self-consistency features were not measured; "
            "RQ2 leaves them out (listed below)."
        )
    text.append(
        "Comparisons between generators on this page are descriptive: each interval describes "
        "one run, and no paired test between generators was computed (the RQ3 rows come from "
        "two separate sets of repeated splits)."
    )
    return " ".join(text)


def _second_subset(
    data: dict[str, Any],
    subset_runs: dict[str, Any] | None,
    full_runs: dict[str, Any],
    full_rq2: dict[str, Any] | None,
) -> str:
    meta = data["meta"]
    llm = _llm(data)
    out = ["### RQ2 on the self-consistency subset"]
    text = [f"`{llm}` with {_samples(subset_runs)} samples per question on"]
    if subset_runs is not None:
        sub_q, full_q = _rag_qids(subset_runs), _rag_qids(full_runs)
        if sub_q == full_q[: len(sub_q)]:
            text.append(
                f"the first {len(sub_q)} questions in dataset order. This is not a random "
                "sample of the full question set."
            )
        else:
            text.append(f"{len(sub_q)} questions; their selection is not checked to be random.")
        matched, total = _first_sample_matches(subset_runs, full_runs)
        text.append(
            f"On the {total} questions answered in both `{llm}` runs, the subset's first-sample "
            f"decision equals the one-sample run's decision for {matched}."
        )
        only_here = _failed_only_in(subset_runs, full_runs)
        if only_here:
            text.append(
                "With several samples per question, a question counts as failed if any of its "
                f"samples is invalid, so {len(only_here)} question(s) failed here but were "
                f"answered in the one-sample run ({', '.join(f'`{q}`' for q in only_here)})."
            )
    else:
        text.append(
            f"{meta['n']} questions (its records file is missing, so how they were chosen "
            "is not checked)."
        )
    text.append(
        f"{meta['n']} answers ({meta['n_failed_excluded']} failed generations excluded), "
        f"accuracy {meta['accuracy']:.3f}"
    )
    if full_rq2 is not None:
        text[-1] += (
            f" against {full_rq2['meta']['accuracy']:.3f} for the same generator on all "
            f"{full_rq2['meta']['n']} answers; subset numbers need not carry over to the full set."
        )
    else:
        text[-1] += "."
    out.append(" ".join(text))

    best = meta["best_single_signal"]
    table = data["estimators"][meta["main_calibrator"]]
    names = _unique(["stated:verbal_conf", *SC_SIGNALS, best, "logistic:deploy", "logistic:full"])
    out.append(
        _table(
            ["Generator", "n", "Estimator", "AUROC", "ECE", "Brier"],
            _estimator_rows(data, [n for n in names if n in table], best),
        )
    )
    stated = table.get("stated:verbal_conf")
    lines = [
        f"`{s}`: mean AUROC {table[s]['auroc']['mean']:.3f} against "
        f"{stated['auroc']['mean']:.3f} for `stated:verbal_conf`"
        for s in SC_SIGNALS
        if s in table and stated
    ]
    if lines:
        out.append(
            "; ".join(lines)
            + ". These are means over the same repeated splits, set side by side; the paired "
            "comparisons in this file are for the learned models:"
        )
    block = data.get("comparisons", {}).get("vs_stated:verbal_conf")
    if block:
        out.append(_comparison_table(block, "stated:verbal_conf", meta["main_calibrator"]))
    out.append(_unavailable(data))
    return "\n\n".join(out)


def _second_rq3(rq3_runs: Sequence[dict[str, Any]], settings: Settings) -> str:
    alphas = _common_alphas(rq3_runs)
    labels = [f"`{_llm(d)}`" for d in rq3_runs]
    out = ["### RQ3: certified abstention"]
    out.append(
        " ".join(
            f"{label}: {d['meta']['n']} answers, n_cal = {d['meta']['n_cal']}, "
            f"δ = {d['meta']['delta']}, {d['meta']['seeds']['repeats']} repeats, confidence "
            f"`{d['meta']['models']['confidence']}` ({d['meta']['models']['calibrator']})."
            for label, d in zip(labels, rq3_runs, strict=True)
        )
    )

    def violation(d: dict[str, Any], method: str, a: str) -> str:
        r = d["selective"][method][a]
        lo, hi = r["violation_ci"]
        return f"{r['violation_rate']:.1%} [{lo:.1%}, {hi:.1%}]"

    out.append(
        "Violation rate (share of repeats with test selective risk above α, Clopper-Pearson "
        "interval over repeats):\n\n"
        + _table(
            ["α", *(f"{m} {lab}" for m in ("certified", "plug_in") for lab in labels)],
            [
                [a, *(violation(d, m, a) for m in ("certified", "plug_in") for d in rq3_runs)]
                for a in alphas
            ],
        )
    )
    out.append(
        "Certified method: share of repeats that certified a threshold, and coverage (mean "
        "over repeats [central 95% range]; when most repeats certify nothing the mean can lie "
        "above the range):\n\n"
        + _table(
            [
                "α",
                *(f"certified in {lab}" for lab in labels),
                *(f"coverage {lab}" for lab in labels),
            ],
            [
                [
                    a,
                    *(_pct(d["selective"]["certified"][a]["certified_rate"]) for d in rq3_runs),
                    *(_fmt(d["selective"]["certified"][a]["coverage"]) for d in rq3_runs),
                ]
                for a in alphas
            ],
        )
    )
    signal = "signal:verbal_conf"
    with_signal = [d for d in rq3_runs if signal in d.get("single_signal", {})]
    if with_signal:
        signal_labels = [f"`{_llm(d)}`" for d in with_signal]
        out.append(
            f"`{signal}` alone as the confidence, same certified procedure:\n\n"
            + _table(
                [
                    "α",
                    *(f"certified in {lab}" for lab in signal_labels),
                    *(f"coverage {lab}" for lab in signal_labels),
                ],
                [
                    [
                        a,
                        *(
                            _pct(d["single_signal"][signal][a]["certified_rate"])
                            for d in with_signal
                        ),
                        *(_fmt(d["single_signal"][signal][a]["coverage"]) for d in with_signal),
                    ]
                    for a in alphas
                ],
            )
        )
    lines = []
    for label, d in zip(labels, rq3_runs, strict=True):
        deployment = d.get("deployment")
        if not deployment:
            continue
        bundle = settings.deployment_dir_for(_llm(d)) / "bundle.json"
        try:
            shown = bundle.relative_to(settings.root).as_posix()
        except ValueError:
            shown = bundle.as_posix()
        lines.append(
            f"- {label} (`{shown}`, {'saved' if deployment['saved'] else 'not saved'} by this "
            f"run): trained on {deployment['n_train']} answers, certified on "
            f"{deployment['n_cal']}, {deployment.get('n_heldout', '–')} held out; thresholds "
            f"{_thresholds(deployment)}."
        )
    if lines:
        out.append("Deployment bundles, one per generator:\n\n" + "\n".join(lines))
    return "\n\n".join(out)


def _generators_figure(rq3_runs: Sequence[dict[str, Any]], report: _Report) -> str:
    alphas = _common_alphas(rq3_runs)
    x = np.array([float(a) for a in alphas])
    fig = Figure(figsize=(6.6, 2.8), layout="constrained")
    ax_c, ax_v = fig.subplots(1, 2)
    series = [(g, m) for g in range(len(rq3_runs)) for m in ("certified", "plug_in")]
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2) * 0.004
    for i, (g, method) in enumerate(series):
        data = rq3_runs[g]
        rows = [data["selective"][method][a] for a in alphas]
        style = {
            "color": SERIES[g],
            "marker": MARKERS[g],
            "markeredgecolor": SERIES[g] if method == "plug_in" else SURFACE,
            "markerfacecolor": SURFACE if method == "plug_in" else SERIES[g],
            "markeredgewidth": 1.0,
            "linestyle": "--" if method == "plug_in" else "-",
            "linewidth": 1.5,
            "elinewidth": 1.0,
            "capsize": 0,
            "label": f"{_llm(data)} {method}",
        }
        rate = np.array([r["violation_rate"] for r in rows])
        ci = np.array([r["violation_ci"] for r in rows])
        ax_v.errorbar(
            x + offsets[i],
            rate,
            yerr=[np.clip(rate - ci[:, 0], 0, None), np.clip(ci[:, 1] - rate, 0, None)],
            **style,
        )
        if method == "certified":
            mean = np.array([r["coverage"]["mean"] for r in rows])
            lo = np.array([r["coverage"]["ci"][0] for r in rows])
            hi = np.array([r["coverage"]["ci"][1] for r in rows])
            # Percentile spread of per-split coverage, clipped as in the RQ3 figure.
            ax_c.errorbar(
                x + offsets[i],
                mean,
                yerr=[np.clip(mean - lo, 0, None), np.clip(hi - mean, 0, None)],
                **style,
            )
    deltas = sorted({d["meta"]["delta"] for d in rq3_runs})
    for delta in deltas:
        ax_v.axhline(delta, color=INK_2, linewidth=0.8, zorder=0)
        ax_v.annotate(
            f"δ = {delta}",
            (x[0], delta),
            textcoords="offset points",
            xytext=(-12, 3),
            ha="left",
            color=INK_2,
            fontsize=7.5,
        )
    for ax in (ax_c, ax_v):
        ax.set_xticks(x, alphas)
        ax.set_xlabel("Target risk α")
        ax.set_ylim(bottom=0)
    ax_c.set_ylabel("Share of questions answered")
    ax_c.set_title("Certified coverage")
    ax_v.set_ylabel("P(test risk > α)")
    ax_v.set_title("Violation rate")
    from matplotlib.lines import Line2D

    # Plain line handles: errorbar handles draw the whisker through the marker.
    handles = [
        Line2D(
            [],
            [],
            color=SERIES[g],
            marker=MARKERS[g],
            linestyle="--" if method == "plug_in" else "-",
            linewidth=1.5,
            markeredgecolor=SERIES[g] if method == "plug_in" else SURFACE,
            markerfacecolor=SURFACE if method == "plug_in" else SERIES[g],
            label=f"{_llm(rq3_runs[g])} {method}",
        )
        for g, method in series
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=2, handlelength=3.5)
    return report.save(
        fig, "generators_rq3", "Certified coverage and violation rate by alpha, per generator"
    )


def _estimator_rows(data: dict[str, Any], names: Sequence[str], best: str) -> list[list[str]]:
    meta = data["meta"]
    table = data["estimators"][meta["main_calibrator"]]
    return [
        [
            f"`{_llm(data)}`",
            str(meta["n"]),
            f"`{name}`" + (" (best)" if name == best else ""),
            _fmt(table[name]["auroc"]),
            _fmt(table[name]["ece"]),
            _fmt(table[name]["brier"]),
        ]
        for name in names
        if name in table
    ]


def _unavailable(data: dict[str, Any]) -> str:
    meta = data["meta"]
    features = meta.get("unavailable_features") or []
    skipped = meta.get("skipped_feature_sets") or []
    if not features:
        return f"`{_llm(data)}` ({meta['n']} answers): every feature was measured."
    return (
        f"`{_llm(data)}` ({meta['n']} answers): features not measured and left out: "
        + ", ".join(f"`{f}`" for f in features)
        + (f"; feature sets skipped: {', '.join(f'`{s}`' for s in skipped)}" if skipped else "")
        + "."
    )


def _rag_qids(runs: dict[str, Any]) -> list[str]:
    return [r["qid"] for r in runs.get("records", []) if r.get("condition", "rag") == "rag"]


def _first_sample_matches(subset: dict[str, Any], full: dict[str, Any]) -> tuple[int, int]:
    """(questions whose first-sample decision agrees, questions answered in both runs)."""
    answered = {
        r["qid"]: r["decision"]
        for r in full.get("records", [])
        if r.get("condition", "rag") == "rag" and "error" not in r
    }
    pairs = [
        (r["decision"], answered[r["qid"]])
        for r in subset.get("records", [])
        if r.get("condition", "rag") == "rag" and "error" not in r and r["qid"] in answered
    ]
    return sum(a == b for a, b in pairs), len(pairs)


def _failed_only_in(subset: dict[str, Any], full: dict[str, Any]) -> list[str]:
    """Questions that failed in `subset` but were answered in `full` (rag condition)."""
    answered = {
        r["qid"]
        for r in full.get("records", [])
        if r.get("condition", "rag") == "rag" and "error" not in r
    }
    return [
        r["qid"]
        for r in subset.get("records", [])
        if r.get("condition", "rag") == "rag" and "error" in r and r["qid"] in answered
    ]


def _samples(runs: dict[str, Any] | None) -> str:
    return "–" if runs is None else str(runs["meta"]["self_consistency_samples"])


def _common_alphas(runs: Sequence[dict[str, Any]]) -> list[str]:
    alphas = [f"{a:.2f}" for a in runs[0]["meta"]["alphas"]]
    return [a for a in alphas if all(a in d["selective"]["certified"] for d in runs)]


def _present(*items: dict[str, Any] | None) -> list[dict[str, Any]]:
    return [item for item in items if item is not None]


def _unique(names: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(names))


def _llm(data: dict[str, Any]) -> str:
    return data["meta"]["models"]["llm"]


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def _comparison_table(block: dict[str, Any], reference: str, calibrator: str) -> str:
    return (
        f"Learned models against `{reference}` ({calibrator} calibration). Brier difference: "
        "per-question test loss, 95% bootstrap over questions, paired permutation p. "
        "AUROC difference: per repeat.\n\n"
        + _table(
            ["Model", "Brier difference", "p", "AUROC difference", "Repeats with higher AUROC"],
            [
                [
                    f"`{m}`",
                    _fmt(c["brier_diff"]),
                    _p(c["brier_p"]),
                    _fmt(c["auroc_diff"]),
                    _pct(c["auroc_better_share"]),
                ]
                for m, c in block.items()
            ],
        )
    )


def _thresholds(deployment: dict[str, Any]) -> str:
    return ", ".join(
        f"α = {a}: {'none certified' if t is None else f'{t:.3f}'}"
        for a, t in deployment["thresholds"].items()
    )


def _forest(
    ax: Axes,
    labels: Sequence[str],
    stats: Sequence[dict[str, Any] | None],
    colours: Sequence[str] | None = None,
) -> None:
    """Point estimates with interval whiskers, first label at the top."""
    y = np.arange(len(labels))[::-1]
    for i, (row, stat) in enumerate(zip(y, stats, strict=True)):
        if stat is None:
            continue
        colour = colours[i] if colours else SERIES[0]
        lo, hi = stat["ci"]
        ax.hlines(row, lo, hi, color=colour, linewidth=1.5)
        ax.plot(stat["mean"], row, "o", color=colour, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.set_yticks(y, labels)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(labels) - 0.4)


def _legend(fig: Figure, entries: Sequence[tuple[str, str, str]]) -> None:
    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color=c, marker=m, linewidth=1.5, markeredgecolor=SURFACE, label=label)
        for label, c, m in entries
    ]
    fig.legend(handles=handles, loc="outside upper center", ncols=len(handles))


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _fmt(stat: dict[str, Any] | None) -> str:
    if not stat:
        return "–"
    lo, hi = stat["ci"]
    return f"{stat['mean']:.3f} [{lo:.3f}, {hi:.3f}]"


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.1%}"


def _p(value: float) -> str:
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.1e}"


def _at(xs: Sequence[float], ys: Sequence[float], x: float) -> float:
    return float(np.interp(x, xs, ys))
