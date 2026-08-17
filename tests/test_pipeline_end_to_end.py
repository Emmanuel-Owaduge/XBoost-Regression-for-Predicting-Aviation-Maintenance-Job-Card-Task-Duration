"""End-to-end test for src/pipeline.py's orchestration wiring.

pipeline.main() calls generate_dataset()/fit_tuned() with no arguments, so
it always uses the real cfg.N_RECORDS/cfg.CV_FOLDS/cfg.RF_PARAM_GRID/
cfg.XGB_PARAM_GRID defaults (n_records=800, full grids) -- those can't be
shrunk via monkeypatching here (cfg.N_RECORDS and cfg.CV_FOLDS are bound as
default-argument values at each module's import time, not looked up at call
time; see the test-skill report). This is therefore the one genuinely slow
test in the suite (real generation + 2 real grid searches + SHAP, ~10-15s),
run as a real integration test rather than shrunk with fakes.

Side effect worth flagging: pipeline.main() writes to the real
cfg.DATA_DIR/cfg.METRICS_DIR/cfg.FIGURES_DIR paths (not a tmp_path) because
those paths are hardcoded module-level constants, not parameters this test
can redirect. Running this test overwrites the project's outputs/ and
data/generated/ artifacts with this run's output -- harmless since it uses
the same seed/config as a normal `python -m src.pipeline` run and produces
identical results, but not hermetically isolated the way the other tests
are.
"""

import json

from src import config as cfg
from src import pipeline


def test_pipeline_main_runs_end_to_end_and_produces_expected_artifacts():
    result = pipeline.main()

    assert set(result.keys()) == {"sanity", "results", "test_metrics", "coverage", "comparison"}
    assert result["sanity"].verdict == "ok"

    result_names = {r.name for r in result["results"]}
    assert result_names == {"linear_regression", "random_forest", "xgboost"}
    assert set(result["test_metrics"].keys()) == result_names
    for name, m in result["test_metrics"].items():
        assert set(m.keys()) == {"rmse", "mae", "mape"}
        assert m["rmse"] > 0

    assert set(result["comparison"]["model"]) == result_names
    assert len(result["comparison"]) == 3

    coverage = result["coverage"]
    assert coverage["nominal_coverage"] == 1 - cfg.CONFORMAL_ALPHA
    assert 0.0 <= coverage["empirical_coverage"] <= 1.0
    assert 0.0 <= coverage["empirical_coverage_first_half"] <= 1.0
    assert 0.0 <= coverage["empirical_coverage_second_half"] <= 1.0

    comparison_path = cfg.METRICS_DIR / "comparison.csv"
    coverage_path = cfg.METRICS_DIR / "conformal_coverage.json"
    assert comparison_path.exists()
    assert coverage_path.exists()
    with open(coverage_path) as f:
        written_coverage = json.load(f)
    assert written_coverage == coverage

    assert (cfg.FIGURES_DIR / "shap_global_importance.png").exists()
    assert (cfg.FIGURES_DIR / "shap_individual_0.png").exists()
