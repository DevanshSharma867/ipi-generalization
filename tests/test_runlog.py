"""Frozen-path guard and run layout of src/runlog.py. Run: python -m pytest tests/"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import runlog


@pytest.mark.parametrize("path", ["data/processed/test.csv", "data/processed", "checkpoints/tfidf_lr_model.pkl",
                                  "results/tfidf_lr_results.json", "results/scores/new.csv",
                                  "results/deberta_predictions/x/y.csv", "results"])
def test_frozen_paths_refused(path):
    with pytest.raises(PermissionError):
        runlog.assert_not_frozen(ROOT / path)


@pytest.mark.parametrize("path", ["results/v3/exp/run/metrics.json", "results/v2_std/x.csv", "results/REGISTRY.md",
                                  "data/v3/train.csv", "checkpoints/v3/model.pkl"])
def test_new_paths_allowed(path):
    runlog.assert_not_frozen(ROOT / path)


def test_frozen_version_refused():
    with pytest.raises(PermissionError):
        runlog.start_run(version="scores", exp="x", seed=1, args={}, data_version="v3")


def test_run_layout(tmp_path):
    data = tmp_path / "data.csv"
    data.write_text("a\n1\n")
    run = runlog.Run("v3", "unit", 7, {"lr": 0.1}, "v3", [data], results_root=tmp_path / "results")
    run.log("hello")
    run.save_metrics({"f1": 0.5, "nan": float("nan")}, headline=("f1", 0.5))
    run.save_predictions(pd.DataFrame({"id": ["a"], "label": [1], "prob": [0.9], "pred": [1], "threshold": [0.5],
                                       "model": ["m"], "eval_set": ["s"]}))
    run.finish()
    cfg = json.loads((run.dir / "config.json").read_text())
    assert cfg["seed"] == 7 and cfg["args"] == {"lr": 0.1}
    assert cfg["data_files_sha256"][str(data)] == runlog.sha256(data)
    assert len(cfg["git_commit"]) == 40 and isinstance(cfg["git_dirty"], bool)
    assert json.loads((run.dir / "metrics.json").read_text())["nan"] is None
    assert list(pd.read_csv(run.dir / "predictions.csv").columns) == runlog.PRED_COLUMNS
    assert "hello" in (run.dir / "log.txt").read_text()
    assert run.run_id.endswith("_seed7") and run.run_id in (tmp_path / "results" / "REGISTRY.md").read_text()


def test_code_dirty_ignores_results(monkeypatch):
    outputs = "?? results/v2_probes/score_x/2026/log.txt\n M results/REGISTRY.md"
    monkeypatch.setattr(runlog, "_git", lambda *a: outputs)
    assert runlog.code_dirty() is False
    monkeypatch.setattr(runlog, "_git", lambda *a: outputs + "\n M src/metrics.py")
    assert runlog.code_dirty() is True


def test_validate_predictions_rejects_extra_columns():
    with pytest.raises(ValueError):
        runlog.validate_predictions(pd.DataFrame({"id": [1], "label": [1], "prob": [1], "pred": [1], "threshold": [.5],
                                                  "model": ["m"], "eval_set": ["s"], "bogus": [0]}))
