"""
runlog.py
=========
Run bookkeeping for every new experiment.

    from runlog import start_run
    run = start_run(version="v3", exp="tfidf_lr", seed=42, args=vars(args),
                    data_version="v3", data_files=["data/v3/train.csv", ...])
    run.log("training ...")
    run.save_predictions(df)                      # standard schema, validated
    run.save_metrics(metrics, headline=("test_f1_macro", 0.81))
    run.finish()                                  # appends results/REGISTRY.md

Layout: results/<version>/<exp>/<YYYYmmdd-HHMMSS>_seed<k>/{config.json, metrics.json, predictions.csv, log.txt}

Frozen paths (tag v2-frozen) are refused: data/processed/, checkpoints/tfidf_lr_model.pkl and every file
or top-level directory under results/ that exists in the tag. `assert_not_frozen(path)` is the guard
other scripts should call before writing.

A run refuses to start when code or data outside results/ has uncommitted changes, so config.json's git_commit
identifies exactly what produced it. Scripts expose `--allow-dirty` via `add_run_args(parser)` to override; the
flag is recorded in config.json and the registry commit gets a "-dirty" suffix. finish() requires metrics.json.
"""

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REGISTRY = RESULTS / "REGISTRY.md"
FROZEN_TAG = "v2-frozen"

# Used when the git tag is unavailable (e.g. a clone without tags)
STATIC_FROZEN = [
    "data/processed", "checkpoints/tfidf_lr_model.pkl",
    "results/scores", "results/deberta_predictions", "results/confusion_matrices",
]

PRED_COLUMNS = [
    "id", "doc_id", "split_seed", "task", "label", "attack_category", "attack_source",
    "insertion_mode", "clean_type", "prob", "pred", "threshold", "model", "run_id", "eval_set",
]

LIBRARIES = ["numpy", "pandas", "scikit-learn", "scipy", "torch", "transformers", "tokenizers",
             "datasets", "nltk", "sentencepiece", "accelerate", "captum"]


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.rstrip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def code_dirty() -> bool:
    """Uncommitted changes outside results/. Run outputs (results/**, incl. REGISTRY.md appends) are not
    code or data drift, so they do not count."""
    for line in _git("status", "--porcelain", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1].strip('"')
        if not path.startswith("results/"):
            return True
    return False


def add_run_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--allow-dirty", action="store_true",
                        help="run despite uncommitted changes outside results/ (recorded in config.json)")
    return parser


def frozen_paths() -> list:
    """Repo-relative frozen paths: everything tracked in the tag under results/ (files and their
    top-level directories), data/processed/ and the TF-IDF checkpoint; static list as fallback."""
    listed = _git("ls-tree", "-r", "--name-only", FROZEN_TAG, "--", "results", "data/processed",
                  "checkpoints/tfidf_lr_model.pkl").splitlines()
    if not listed:
        return list(STATIC_FROZEN)
    paths = set(STATIC_FROZEN)
    for p in listed:
        paths.add(p)
        parts = p.split("/")
        if parts[0] == "results" and len(parts) > 2:
            paths.add("/".join(parts[:2]))
    return sorted(paths)


def is_frozen(path) -> bool:
    p = Path(path).resolve()
    if p == RESULTS or p == ROOT:
        return True
    for f in frozen_paths():
        fp = (ROOT / f).resolve()
        if p == fp or fp in p.parents:
            return True
    return False


def assert_not_frozen(path) -> None:
    if is_frozen(path):
        raise PermissionError(f"Refusing to write to frozen V2 path: {path} (frozen at tag {FROZEN_TAG})")


def validate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with exactly the standard columns, in order. Missing optional columns are added empty."""
    required = {"id", "label", "prob", "pred", "threshold", "model", "eval_set"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {sorted(missing)}")
    extra = set(df.columns) - set(PRED_COLUMNS)
    if extra:
        raise ValueError(f"predictions have non-standard columns: {sorted(extra)}")
    out = df.copy()
    for c in PRED_COLUMNS:
        if c not in out:
            out[c] = ""
    return out[PRED_COLUMNS]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> Dict[str, Optional[str]]:
    out = {"python": platform.python_version()}
    for lib in LIBRARIES:
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = None
    return out


def _hardware() -> Dict:
    hw = {"platform": platform.platform(), "machine": platform.machine(), "node": platform.node(),
          "cpu_count": os.cpu_count(), "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    if "torch" in sys.modules:  # only report devices when the run already uses torch
        torch = sys.modules["torch"]
        hw["cuda"] = torch.cuda.is_available()
        hw["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        hw["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return hw


def jsonable(obj):
    """NaN/inf -> None, numpy scalars -> Python, recursively."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path, obj) -> None:
    assert_not_frozen(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=2)


class Run:
    def __init__(self, version: str, exp: str, seed: int, args: Dict, data_version: str,
                 data_files: Iterable, results_root: Path = RESULTS, allow_dirty: bool = False):
        if not version or "/" in version or not exp or "/" in exp:
            raise ValueError("version and exp must be single non-empty path components")
        self.version, self.exp, self.seed, self.data_version = version, exp, seed, data_version
        self.run_id = f"{_dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_seed{seed}"
        self.dir = Path(results_root) / version / exp / self.run_id
        self.registry = Path(results_root) / "REGISTRY.md"
        assert_not_frozen(self.dir)
        dirty = code_dirty()
        if dirty and not allow_dirty:
            raise RuntimeError("uncommitted changes outside results/ (git status); commit them or pass --allow-dirty")
        if self.dir.exists():
            raise FileExistsError(self.dir)
        self.dir.mkdir(parents=True)
        self.headline: Optional[Tuple[str, float]] = None
        self._log = open(self.dir / "log.txt", "a")
        files = {str(f): sha256(f) for f in data_files}
        config = {
            "run_id": self.run_id, "version": version, "exp": exp, "seed": seed, "args": args,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": dirty, "allow_dirty": allow_dirty,
            "data_version": data_version, "data_files_sha256": files,
            "library_versions": _versions(), "hardware": _hardware(),
            "argv": sys.argv, "started": _dt.datetime.now().isoformat(timespec="seconds"),
        }
        write_json(self.dir / "config.json", config)
        self.config = config

    def log(self, msg: str) -> None:
        line = f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        self._log.write(line + "\n")
        self._log.flush()

    def save_metrics(self, metrics: Dict, headline: Optional[Tuple[str, float]] = None) -> None:
        write_json(self.dir / "metrics.json", metrics)
        if headline:
            self.headline = headline

    def save_predictions(self, df: pd.DataFrame) -> Path:
        df = validate_predictions(df.assign(run_id=self.run_id))
        path = self.dir / "predictions.csv"
        df.to_csv(path, index=False)
        return path

    def finish(self) -> None:
        self._log.close()
        if not (self.dir / "metrics.json").exists():
            raise RuntimeError(f"{self.dir}: finish() without save_metrics(); every run directory needs metrics.json")
        append_registry(self)


def start_run(version: str, exp: str, seed: int, args: Dict, data_version: str, data_files: Iterable = (),
              allow_dirty: bool = False) -> Run:
    return Run(version, exp, seed, args, data_version, data_files, allow_dirty=allow_dirty)


def append_registry(run: Run) -> None:
    if not run.registry.exists():
        run.registry.write_text(
            "# Run registry\n\nOne line per run (appended by src/runlog.py). Runs before tag v2-frozen are not listed;\n"
            "see CODEBASE_STATE.md §6.\n\n"
            "| date | version | exp | run_id | commit | data version | headline metric | path |\n"
            "|---|---|---|---|---|---|---|---|\n")
    head = f"{run.headline[0]}={run.headline[1]:.4f}" if run.headline else ""
    commit = run.config["git_commit"][:7] + ("-dirty" if run.config["git_dirty"] else "")
    rel = run.dir.relative_to(ROOT) if ROOT in run.dir.parents else run.dir
    with open(run.registry, "a") as f:
        f.write(f"| {_dt.date.today().isoformat()} | {run.version} | {run.exp} | {run.run_id} | {commit} | "
                f"{run.data_version} | {head} | {rel} |\n")
