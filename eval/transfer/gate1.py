"""
gate1.py
========
Gate 1 of the transfer harness (paper_plan A17): the rebuilt sets must line up row for row with the score files
Liu (2026) released at commit 5d0ae7c. For every set and every released score file of that set (results/many/*
and the hfjudge_* files): same number of rows, and, where the release carries labels, the same label sequence.
Also checks the set sizes the paper reports. Logged through runlog to results/transfer/set_build_gate1/<run_id>/.

    python eval/transfer/gate1.py [--sets data/transfer/sets] [--upstream data/raw/benign-instruction-bench]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from runlog import add_run_args, start_run

# rebuilt file -> (Liu's set key in results/many, his judge-file stem, sizes the paper reports {label: rows})
SETS = {
    "agentdojo_benign_outputs.jsonl": ("adclean", "agentdojo_clean", {None: 339}),
    "agentdojo_pairs.jsonl": ("admatched", "agentdojo_pairs", {0: 585, 1: 432}),
    "agentdojo_pairs_shift.jsonl": ("adshift", "agentdojo_pairs_shift", {0: 975, 1: 720}),
    "taubench_eval.jsonl": ("tbeval", "taubench_eval", {0: 1533, 1: 584}),
    "bipia_pairs.jsonl": ("bipia", "bipia_pairs", {0: 1200, "1:test": 2435}),
    "corpus.jsonl": ("general", None, {None: 7504}),
}


def released_files(upstream: Path, key: str, stem):
    files = sorted((upstream / "results" / "many").glob(f"*__{key}.json"))
    if stem:   # *_agentdojo_pairs.json must not pick up *_agentdojo_pairs_shift.json
        files += sorted(f for f in (upstream / "results").glob(f"hfjudge_*{stem}.json") if f.name.endswith(f"_{stem}.json"))
    return files


def check_set(rows, files, expected):
    labels = [r.get("label") for r in rows]
    problems, compared = [], []
    for f in files:
        d = json.load(open(f))
        probs, rel = d.get("probs", []), d.get("labels")
        ok_len = len(probs) == len(rows)
        if rel is None:
            ok_lab = True
        elif all(x is None for x in labels):          # unlabelled benign set: the release stores zeros
            ok_lab = all(int(x) == 0 for x in rel)
        else:
            ok_lab = [int(x) for x in rel] == [int(x) for x in labels]
        compared.append({"file": f.name, "rows": len(probs), "labels_released": rel is not None,
                         "rows_match": ok_len, "labels_match": ok_lab})
        if not (ok_len and ok_lab):
            problems.append(f"{f.name}: rows {len(probs)} vs {len(rows)}, labels match {ok_lab}")
    counts = {}
    for lab, n in expected.items():
        if lab is None:
            got = len(rows)
        elif lab == "1:test":                          # injected rows with a test-half attack (BIPIA)
            got = sum(1 for r in rows if r.get("label") == 1 and r.get("attack_split") == "test")
        else:
            got = sum(1 for x in labels if x == lab)
        counts[str(lab)] = {"expected": n, "rebuilt": got}
        if got != n:
            problems.append(f"label {lab}: {got} rows, paper reports {n}")
    if not files:
        problems.append("no released score file to compare against")
    return {"released_files": compared, "paper_counts": counts, "problems": problems, "pass": not problems}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=Path, default=ROOT / "data" / "transfer" / "sets")
    ap.add_argument("--upstream", type=Path, default=ROOT / "data" / "raw" / "benign-instruction-bench")
    a = add_run_args(ap).parse_args()
    manifest = json.load(open(a.sets / "manifest.json"))
    run = start_run(version="transfer", exp="set_build_gate1", seed=0,
                    args={"sets": "data/transfer/sets", "upstream_commit": manifest["upstream"]["commit"],
                          "expected": {f: {str(k): v for k, v in e.items()} for f, (_, _, e) in SETS.items()}},
                    data_version=manifest["data_version"],
                    data_files=[a.sets / f for f in SETS] + [a.sets / "manifest.json"], allow_dirty=a.allow_dirty)
    out = {}
    for fname, (key, stem, expected) in SETS.items():
        rows = [json.loads(line) for line in open(a.sets / fname)]
        res = check_set(rows, released_files(a.upstream, key, stem), expected)
        out[fname] = res
        run.log(f"{fname:32s} rows {len(rows):5d} vs {len(res['released_files'])} released files: "
                f"{'PASS' if res['pass'] else 'FAIL ' + '; '.join(res['problems'])}")
    ok = all(r["pass"] for r in out.values())
    run.log(f"gate1: {'PASS' if ok else 'FAIL'}")
    run.save_metrics({"sets": out, "pass": ok, "manifest": manifest}, headline=("gate1_pass", float(ok)))
    run.finish()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
