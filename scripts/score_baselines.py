"""
score_baselines.py
==================
Scores a trained baseline (paper_plan §7 step 4, A30) on the V3 eval sets of one split seed, optionally its V3
position probes, and optionally the agent transfer sets, after selection is fixed. Writes one run directory
results/<version>/<exp>/<run_id>/ with predictions.csv.gz (runlog PRED_COLUMNS; eval_set = set name) and
metrics.json (rows, flag rate at 0.5 and the fraction of inputs over 512 tokens, per set). Metrics with CIs are
computed later from the predictions (scripts/v3_baselines_report.py).

Input to every model: build_combined_text, i.e. "Intent: <user_intent> | Context: <text>". On the agent sets
the intent is the task's user request and the context is the tool output (paper_plan §3 Inputs). Inputs over 512
tokens keep the intent and lose the end of the context (DeBERTa tokenizer truncation; §3 Truncation).

    python scripts/score_baselines.py --model deberta --ckpt checkpoints/v3/deberta_v3/split_seed_0_seed0 \\
        --exp score_deberta_v3 --seed 0 --v3_split_dir data/v3/split_seed_0 [--probes data/v3_probes/split_seed_0.csv] \\
        [--agent_dir data/transfer/sets_adliu --agent_sets admatched adshift tbeval adclean tbclean calpool]
    python scripts/score_baselines.py --model tfidf --ckpt checkpoints/v3/tfidf_lr/split_seed_0.pkl ...
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "models")]
from preprocess import build_combined_text, preprocess_for_tfidf  # noqa: E402
from runlog import add_run_args, start_run  # noqa: E402

V3_SETS = ("test", "val", "val_id", "raw_clean", "ambiguous", "long")
# eval/transfer/detectors.py SETS (Liu's set keys -> rebuilt files in data/transfer/sets_adliu)
AGENT_FILES = {"adclean": "agentdojo_benign_outputs.jsonl", "admatched": "agentdojo_pairs.jsonl",
               "adshift": "agentdojo_pairs_shift.jsonl", "tbeval": "taubench_eval.jsonl",
               "tbclean": "taubench_clean.jsonl", "calpool": "calibration_pool.jsonl"}
MAX_LEN = 512


def v3_frame(df: pd.DataFrame, eval_set: str) -> pd.DataFrame:
    """A V3 CSV (or probe CSV) -> rows with the scoring input and the PRED metadata columns."""
    out = pd.DataFrame({
        "id": df["id"], "doc_id": df["doc_id"], "split_seed": df.get("split_seed", ""), "task": df["task_type"],
        "label": pd.to_numeric(df["label"], errors="coerce").astype("Int64"),  # ambiguous: no label
        "attack_category": df.get("attack_category", ""),
        "attack_source": df.get("attack_source", ""), "insertion_mode": df.get("insertion_mode", ""),
        "clean_type": df.get("clean_type", df.get("row_type", "")), "eval_set": eval_set})
    out["input"] = df.apply(build_combined_text, axis=1).values
    return out


def load_v3(split_dir: Path, probes=None) -> pd.DataFrame:
    frames = [v3_frame(pd.read_csv(split_dir / f"{s}.csv", keep_default_na=False), f"v3_{s}") for s in V3_SETS]
    if probes:
        p = pd.read_csv(probes, keep_default_na=False)
        frames += [v3_frame(g, f"v3probe_{name}") for name, g in p.groupby("probe", sort=True)]
    return pd.concat(frames, ignore_index=True)


def load_agent(agent_dir: Path, sets) -> pd.DataFrame:
    frames = []
    for s in sets:
        rows = [json.loads(line) for line in open(agent_dir / AGENT_FILES[s])]
        df = pd.DataFrame(rows)
        lab = df["label"].astype(int) if "label" in df else pd.Series(0, index=df.index)
        cl = df["suite"].astype(str) + "/" + df["task"].astype(str)  # user-task clusters (transfer_metrics)
        frames.append(pd.DataFrame({
            "id": [f"{s}_{i:05d}" for i in range(len(df))], "doc_id": cl, "split_seed": "", "task": df["suite"],
            "label": lab, "attack_category": df.get("attack", pd.Series("", index=df.index)).fillna(""),
            "attack_source": "", "insertion_mode": "", "clean_type": "", "eval_set": s,
            "input": [build_combined_text(pd.Series({"user_intent": r["user_request"], "text": r["text"]}))
                      for r in rows]}))
    return pd.concat(frames, ignore_index=True)


def score_deberta(ckpt: str, texts, bs: int = 16):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
    from deberta import get_device
    dev = get_device()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, dtype=torch.float32).to(dev).eval()
    n_tok = [len(tok(t, truncation=False)["input_ids"]) for t in texts]
    enc = [tok(t, truncation=True, max_length=MAX_LEN) for t in texts]
    loader = DataLoader([{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in enc],
                        batch_size=bs, shuffle=False, collate_fn=DataCollatorWithPadding(tok))
    probs = []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev)).logits
            probs.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy().tolist())
    return np.array(probs), np.array(n_tok) > MAX_LEN


def score_tfidf(ckpt: str, texts):
    pipe = pickle.load(open(ckpt, "rb"))
    return pipe.predict_proba(preprocess_for_tfidf(list(texts)))[:, 1], np.zeros(len(texts), bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["deberta", "tfidf"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seed", type=int, required=True, help="training seed of the checkpoint (run id suffix)")
    ap.add_argument("--model_name", default=None, help="model column (default: <model>_<exp>_s<seed>)")
    ap.add_argument("--v3_split_dir", type=Path, default=None)
    ap.add_argument("--probes", type=Path, default=None)
    ap.add_argument("--agent_dir", type=Path, default=None)
    ap.add_argument("--agent_sets", nargs="*", default=[])
    ap.add_argument("--data_version", default="v3")
    a = add_run_args(ap).parse_args()
    frames, files = [], []
    if a.v3_split_dir:
        frames.append(load_v3(a.v3_split_dir, a.probes))
        files += [a.v3_split_dir / f"{s}.csv" for s in V3_SETS] + ([a.probes] if a.probes else [])
    if a.agent_sets:
        frames.append(load_agent(a.agent_dir, a.agent_sets))
        files += [a.agent_dir / AGENT_FILES[s] for s in a.agent_sets]
    if not frames:
        sys.exit("nothing to score: give --v3_split_dir and/or --agent_sets")
    df = pd.concat(frames, ignore_index=True)
    run = start_run(version=a.version, exp=a.exp, seed=a.seed,
                    args={k: str(v) for k, v in vars(a).items()}, data_version=a.data_version,
                    data_files=[f for f in files if Path(f).exists() and Path(f).stat().st_size < 2e9],
                    allow_dirty=a.allow_dirty)
    run.log(f"scoring {len(df)} rows with {a.model} {a.ckpt}")
    uniq = pd.unique(df["input"])  # each distinct input scored once
    fn = score_deberta if a.model == "deberta" else score_tfidf
    p, over = fn(a.ckpt, uniq)
    pmap, omap = dict(zip(uniq, p)), dict(zip(uniq, over))
    df["prob"] = df["input"].map(pmap).astype(float)
    df["over_512"] = df["input"].map(omap).astype(bool)
    df["threshold"] = 0.5
    df["pred"] = (df["prob"] >= 0.5).astype(int)
    df["model"] = a.model_name or f"{a.model}_{a.exp}_s{a.seed}"
    summary = {s: {"n": int(len(g)), "n_injected": int(g["label"].fillna(0).sum()), "flag_rate_at_0.5": float(g["pred"].mean()),
                   "frac_over_512_tokens": float(g["over_512"].mean())} for s, g in df.groupby("eval_set")}
    run.save_metrics({"checkpoint": a.ckpt, "model": a.model, "n_unique_inputs": int(len(uniq)), "sets": summary})
    run.save_predictions(df.drop(columns=["input", "over_512"]))
    for s, v in summary.items():
        run.log(f"{s:28s} n={v['n']:5d} flag@0.5={v['flag_rate_at_0.5']:.3f} over512={v['frac_over_512_tokens']:.3f}")
    run.finish()


if __name__ == "__main__":
    main()
