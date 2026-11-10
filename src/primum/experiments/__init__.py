"""Experiments (SPEC §3.9). Each module has one entry point, `run(*, created, **options) -> Path`,
which writes one results file and returns its path; `primum exp <name>` calls it.

    run_pubmedqa    results/pubmedqa_runs.json   benchmark records RQ2 and RQ3 are computed from
    rq1_retrieval   results/rq1_retrieval.json   retrieval quality of every retriever
    rq2_confidence  results/rq2_confidence.json  which signals predict correctness, calibration
    rq3_conformal   results/rq3_conformal.json   certified abstention; saves the deployment bundle
    report          results/figures/, docs/RESULTS.md, drawn only from the files above

The modules are imported by name (`from primum.experiments import rq2_confidence`); this package
imports nothing itself, so loading one experiment does not load the others' dependencies.
"""
