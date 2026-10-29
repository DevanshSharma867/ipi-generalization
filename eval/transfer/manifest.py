"""
manifest.py
===========
Provenance of the rebuilt transfer sets (scripts/build_transfer_sets.sh): upstream and dependency commits, the
Python environments that ran the upstream scripts (pip freeze), the HF dataset revisions current at build time
(the general corpus streams them unpinned), and per-file sha256, row counts and label counts.
Writes <sets>/manifest.json and prints the data_version string used by every transfer run.

    python eval/transfer/manifest.py --sets data/transfer/sets --agentsec data/transfer/agentsec \\
        --upstream data/raw/benign-instruction-bench --venv agentdojo=.venv-agentdojo --venv liu=.venv-liu
"""

import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
from collections import Counter
from pathlib import Path

FILES = {  # file -> set name (paper_plan §5)
    "agentdojo_benign_outputs.jsonl": "ad_clean",
    "agentdojo_pairs.jsonl": "ad_matched",
    "agentdojo_pairs_shift.jsonl": "ad_shift",
    "taubench_clean.jsonl": "tb_clean",
    "taubench_pairs.jsonl": "tb_pairs",
    "taubench_eval.jsonl": "tb_matched",
    "bipia_pairs.jsonl": "bipia",
    "corpus.jsonl": "general",
}
HF_DATASETS = ["h1alexbel/github-readmes", "Yale-LILY/aeslc", "HuggingFaceFW/fineweb-edu"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def freeze(venv: Path) -> dict:
    py = venv / "bin" / "python"
    version = subprocess.run([str(py), "--version"], capture_output=True, text=True).stdout.strip()
    pkgs = subprocess.run(["uv", "pip", "freeze", "-p", str(py)], capture_output=True, text=True).stdout.split()
    return {"python": version, "packages": sorted(pkgs)}


def file_stats(path: Path) -> dict:
    rows = [json.loads(line) for line in open(path)]
    out = {"sha256": sha256(path), "rows": len(rows)}
    if rows and "label" in rows[0]:
        out["labels"] = dict(sorted(Counter(str(r["label"]) for r in rows).items()))
        out["distinct_texts_by_label"] = {lab: len({r["text"] for r in rows if str(r["label"]) == lab})
                                          for lab in out["labels"]}
    else:
        out["distinct_texts"] = len({r["text"] for r in rows})
    if rows and "suite" in rows[0] and "task" in rows[0]:
        out["tasks"] = len({(r["suite"], r["task"]) for r in rows})
    return out


def hf_revisions() -> dict:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        return {name: api.dataset_info(name).sha for name in HF_DATASETS}
    except Exception as e:  # recorded, not fatal: the corpus is not part of the reproduction gate
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, required=True)
    ap.add_argument("--agentsec", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--venv", action="append", default=[], help="name=path of an environment that ran upstream scripts")
    ap.add_argument("--agentdojo", default="")
    a = ap.parse_args()
    files = {name: file_stats(a.sets / f) for f, name in FILES.items() if (a.sets / f).exists()}
    missing = [f for f in FILES if not (a.sets / f).exists()]
    m = {
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "upstream": {"repo": "https://github.com/lzwhehe/benign-instruction-bench", "commit": git_head(a.upstream)},
        "dependencies": {p.name: git_head(p) for p in sorted(a.agentsec.iterdir()) if (p / ".git").exists()},
        "agentdojo": a.agentdojo, "agentdojo_suites": "v1.2", "pythonhashseed": 0,
        "environments": {k: freeze(Path(v)) for k, v in (x.split("=", 1) for x in a.venv)},
        "hf_dataset_revisions_at_build": hf_revisions(),
        "files": files, "missing": missing,
    }
    content = hashlib.sha256(json.dumps({k: v["sha256"] for k, v in files.items()}, sort_keys=True).encode())
    m["data_version"] = (f"transfer-bib{m['upstream']['commit'][:7]}-agentdojo{a.agentdojo.split('==')[-1]}"
                         f"-tb{m['dependencies'].get('tau-bench', '')[:7]}-files{content.hexdigest()[:8]}")
    (a.sets / "manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    print(m["data_version"])
    for name, s in files.items():
        print(f"  {name:12s} rows {s['rows']:5d} labels {s.get('labels', '-')} tasks {s.get('tasks', '-')}")
    if missing:
        print("MISSING:", missing)


if __name__ == "__main__":
    main()
