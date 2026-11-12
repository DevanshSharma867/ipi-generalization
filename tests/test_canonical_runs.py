"""Reports read the run listed in configs/canonical_runs.json, never the latest run on disk."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import v3_baselines_report as rep  # noqa: E402


def make_run(root, rel, seed, dirty=False, finished=True):
    d = root / rel
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"git_dirty": dirty, "seed": seed}))
    if finished:
        (d / "finished.json").write_text("{}")
    return d


def use(monkeypatch, root, listing):
    (root / "canonical_runs.json").write_text(json.dumps(listing))
    monkeypatch.setattr(rep, "ROOT", root)
    monkeypatch.setattr(rep, "CANONICAL", root / "canonical_runs.json")


def test_listed_run_wins_over_newer(tmp_path, monkeypatch):
    old = make_run(tmp_path, "results/v3/score_x/20261111-000000_seed0", 0)
    make_run(tmp_path, "results/v3/score_x/20261112-000000_seed0", 0)  # newer, not listed
    use(monkeypatch, tmp_path, {"v3/score_x": {"0": "results/v3/score_x/20261111-000000_seed0"}})
    assert rep.canonical_runs("v3", "score_x") == {0: old}
    assert rep.canonical_runs("v3", "not_listed") == {}


@pytest.mark.parametrize("kw,rel", [({"dirty": True}, None), ({"finished": False}, None),
                                    ({}, "results/v3/score_x/20261111-000000_seed1")])
def test_bad_entries_refused(tmp_path, monkeypatch, kw, rel):
    rel = rel or "results/v3/score_x/20261111-000000_seed0"
    make_run(tmp_path, rel, 0, **kw)
    use(monkeypatch, tmp_path, {"v3/score_x": {"0": rel}})
    with pytest.raises(ValueError):
        rep.canonical_runs("v3", "score_x")


def test_repo_listing_valid():
    listing = json.loads(rep.CANONICAL.read_text())
    for key in (k for k in listing if not k.startswith("_")):
        version, exp = key.split("/")
        assert rep.canonical_runs(version, exp)
