"""
score_probes.py
===============
Score the V2 probe sets (data/probes_v2/) and the V2 test set with one detector, reusing the scorers in
score_detectors.py unchanged (tfidf / deberta / hf / judge; the judge prompt is score_detectors.JUDGE_SYSTEM,
frozen). Writes standard-schema predictions (eval_set = probe name) through runlog to
results/v2_probes/score_<name>/<run_id>/.

    python scripts/score_probes.py --model tfidf --name tfidf_c1
    python scripts/score_probes.py --model deberta --checkpoint checkpoints/deberta_s42 --name deberta_s42 \\
        --checkpoint_origin hpc_original
    python scripts/score_probes.py --model judge --checkpoint Qwen/Qwen2.5-7B-Instruct --name judge_qwen2.5_7b \\
        --sets p1_neutral_insertion p4_whitespace_only p2_benign_moved p3_attack_moved --doc_frac 0.25
"""

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import score_detectors as SD
from convert_v2_predictions import v2_fields
from runlog import sha256, start_run

PROBES = ROOT / "data" / "probes_v2"
PROBE_SETS = ["p1_neutral_insertion", "p2_benign_moved", "p3_attack_moved", "p4_whitespace_only"]
ALL_SETS = PROBE_SETS + ["v2_test"]
CLEAN_TYPE = {"p1_neutral_insertion": "neutral", "p2_benign_moved": "benign_imperative",
              "p3_attack_moved": "", "p4_whitespace_only": "whitespace_collapsed"}
SUBSAMPLE_SETS = ("p2_benign_moved", "p3_attack_moved")  # --doc_frac applies to these only


def load_set(name):
    if name == "v2_test":
        df = pd.read_csv(ROOT / "data/processed/test.csv")
        return df, v2_fields("test").loc[df["id"]].reset_index().rename(columns={"index": "id"})
    df = pd.read_csv(PROBES / f"{name}.csv", keep_default_na=False)
    fields = pd.DataFrame({
        "id": df["id"], "doc_id": df["doc_id"], "split_seed": 42, "task": df["task_type"], "label": df["label"],
        "attack_category": df["attack_type"].where(df["label"] == 1, ""), "attack_source": df["attack_source"],
        "insertion_mode": df["position"].where(df["position"] != "none", ""), "clean_type": CLEAN_TYPE[name]})
    return df, fields


def subsample_docs(df, frac, seed):
    docs = np.array(sorted(df["doc_id"].unique()))
    keep = np.random.RandomState(seed).choice(docs, size=max(1, int(round(frac * len(docs)))), replace=False)
    return df[df["doc_id"].isin(keep)].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(SD.SCORERS), required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--name", required=True, help="model name in predictions and run directory")
    ap.add_argument("--sets", nargs="+", default=ALL_SETS, choices=ALL_SETS)
    ap.add_argument("--doc_frac", type=float, default=1.0, help="fraction of docs scored in P2/P3 (judge budget)")
    ap.add_argument("--seed", type=int, default=42, help="doc subsample seed")
    ap.add_argument("--checkpoint_origin", default="", help="e.g. hpc_original / retrained_identical_config")
    a = ap.parse_args()

    files = [PROBES / f"{s}.csv" for s in a.sets if s != "v2_test"]
    files += [ROOT / "data/processed/test.csv"] if "v2_test" in a.sets else []
    if a.model == "tfidf":
        files.append(ROOT / "checkpoints/tfidf_lr_model.pkl")
    args = {**vars(a), "probe_manifest_sha256": sha256(PROBES / "manifest.json")}
    if a.model == "judge":
        args["judge_prompt_sha256"] = hashlib.sha256(inspect.getsource(SD.score_judge).encode()
                                                     + SD.JUDGE_SYSTEM.encode()).hexdigest()
        args["judge_system_prompt"] = SD.JUDGE_SYSTEM
    run = start_run(version="v2_probes", exp=f"score_{a.name}", seed=a.seed, args=args,
                    data_version="v2_probes", data_files=files)
    preds = []
    for s in a.sets:
        df, fields = load_set(s)
        if s in SUBSAMPLE_SETS and a.doc_frac < 1.0:
            df = subsample_docs(df, a.doc_frac, a.seed)
            fields = fields.set_index("id").loc[df["id"]].reset_index()
        score = SD.SCORERS[a.model](df, a.checkpoint)
        out = fields.assign(prob=np.asarray(score, float), pred=(np.asarray(score) >= 0.5).astype(int),
                            threshold=0.5, model=a.name, eval_set=s)
        run.log(f"{s}: {len(out)} rows, flag rate {out['pred'].mean():.4f}")
        preds.append(out)
    run.save_predictions(pd.concat(preds, ignore_index=True))
    run.save_metrics({"rows_per_set": {p["eval_set"].iloc[0]: len(p) for p in preds}})
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
