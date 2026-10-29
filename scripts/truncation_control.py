"""
truncation_control.py
=====================
Do the position effects survive when every inserted span is fully visible to the model?

For every P1/P2/P3 row and every model family with a context limit, decide with the model's OWN tokenizer
(copied from the HPC files used for scoring into hf_cache/tokenizers_hpc/) whether the inserted span lies
fully inside the visible window:
  deberta   (our V2 checkpoints)  input "Intent: .. | Context: ..", truncation at 512 tokens (incl. specials)
  protectai / piguard             input = document text only, truncation at 512 tokens
  qwen judge                      chat-template prompt, NO truncation in score_detectors.score_judge;
                                  visible iff the prompt fits Qwen2.5-7B's 32768 max_position_embeddings
  tfidf                           no limit (all spans visible)
A span is visible iff its last character falls before the end offset of the last kept non-special token.

Matched subset: a source (same doc + same inserted text) is kept only if its span is visible at ALL three
positions; position cells and paired contrasts are then recomputed with probe_report.model_report.

Logged with runlog to results/v2_probes/truncation_control/<run_id>/ (metrics.json + visibility.csv).

    python scripts/truncation_control.py
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
import score_detectors as SD
from preprocess import build_combined_text
from probe_report import MEAN_SEEDS, latest_predictions, mean_f1, mean_flag_rate, model_report, seed_mean_frame
from runlog import add_run_args, start_run
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
TOK = ROOT / "hf_cache" / "tokenizers_hpc"
PROBES = ROOT / "data" / "probes_v2"
SETS = ["p1_neutral_insertion", "p2_benign_moved", "p3_attack_moved"]
QWEN_MAX_POSITIONS = 32768
FAMILY = {"tfidf_c1": None, "deberta_s42": "deberta", "deberta_s123": "deberta", "deberta_s999": "deberta",
          "deberta_mean": "deberta", "protectai": "protectai", "piguard": "piguard", "judge_qwen2.5_7b": "qwen",
          "judge_qwen2.5_7b_vsum": "qwen"}
CONTRAST_KEYS = ("p1_start_minus_middle", "p3_end_minus_start", "p2_start_minus_middle")


def model_input(row, family):
    """(input string, char offset of the span in it) exactly as the scorer builds the input."""
    if family == "deberta":
        text = build_combined_text(row)
        ctx = str(row["text"]).strip()
        lead = len(row["text"]) - len(row["text"].lstrip())
        off = len(text) - len(ctx) + int(row["span_start"]) - lead
    else:  # protectai / piguard see the raw document text
        text, off = row["text"], int(row["span_start"])
    assert text[off:off + int(row["span_len"])] == row["inserted_text"]
    return text, off


def visibility_512(df, tok, family):
    out = []
    for _, r in df.iterrows():
        text, off = model_input(r, family)
        full = tok(text, add_special_tokens=True)["input_ids"]
        enc = tok(text, truncation=True, max_length=512, add_special_tokens=True, return_offsets_mapping=True)
        vis_end = max(e for s, e in enc["offset_mapping"] if e > s)
        span_end = off + int(r["span_len"])
        out.append({"n_tokens": len(full), "truncated": len(full) > 512,
                    "visible": len(full) <= 512 or span_end <= vis_end})
    return pd.DataFrame(out, index=df.index)


def visibility_qwen(df, tok):
    out = []
    for _, r in df.iterrows():
        msgs = [{"role": "system", "content": SD.JUDGE_SYSTEM},
                {"role": "user", "content": f"User request:\n{r['user_intent']}\n\nDocument:\n{r['text']}\n\n"
                                            "Does the document contain a prompt injection relative to this request?"}]
        n = len(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
        out.append({"n_tokens": n, "truncated": n > QWEN_MAX_POSITIONS, "visible": n <= QWEN_MAX_POSITIONS})
    return pd.DataFrame(out, index=df.index)


def source_key(ids: pd.Series) -> pd.Series:
    return ids.str.rsplit("_", n=1).str[0]


def main():
    a = add_run_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    probes = pd.concat([pd.read_csv(PROBES / f"{s}.csv", keep_default_na=False).assign(eval_set=s) for s in SETS],
                       ignore_index=True)
    found = latest_predictions()
    tok_dirs = {"deberta": "deberta_v2_ckpt", "protectai": "protectai", "piguard": "piguard", "qwen": "qwen2.5_7b"}
    run = start_run(version="v2_probes", exp="truncation_control", seed=42,
                    args={"window_tokens": 512, "qwen_max_positions": QWEN_MAX_POSITIONS, "tokenizers": tok_dirs,
                          "judge_truncation": "none (score_detectors.score_judge passes the full chat-template prompt)",
                          "prediction_sources": {m: src for m, (_, src) in found.items()}},
                    data_version="v2_probes",
                    data_files=[PROBES / f"{s}.csv" for s in SETS]
                    + sorted(p for d in tok_dirs.values() for p in (TOK / d).iterdir()),
                    allow_dirty=a.allow_dirty)

    vis = []
    for fam, d in tok_dirs.items():
        tok = AutoTokenizer.from_pretrained(str(TOK / d))
        v = visibility_qwen(probes, tok) if fam == "qwen" else visibility_512(probes, tok, fam)
        vis.append(probes[["id", "doc_id", "task_type", "eval_set", "position"]].join(v).assign(family=fam))
        run.log(f"{fam}: max tokens {v['n_tokens'].max()}, rows with span not fully visible {(~v['visible']).sum()}")
    vis = pd.concat(vis, ignore_index=True)
    vis.to_csv(run.dir / "visibility.csv", index=False)

    vis["source"] = source_key(vis["id"])
    matched = vis.groupby(["family", "source"])["visible"].transform("all")
    vis["matched"] = matched
    survival, trunc_end = {}, {}
    for fam, g in vis.groupby("family"):
        survival[fam] = {}
        for (s, task), h in g.groupby(["eval_set", "task_type"]):
            survival[fam].setdefault(s, {})[task] = {
                "sources": int(h["source"].nunique()), "sources_kept": int(h.loc[h["matched"], "source"].nunique()),
                "docs": int(h["doc_id"].nunique()), "docs_with_kept_source": int(h.loc[h["matched"], "doc_id"].nunique()),
                "rows": len(h), "rows_kept": int(h["matched"].sum())}
        p3 = g[g["eval_set"] == "p3_attack_moved"]
        trunc_end[fam] = {pos: {t: float((~h["visible"]).mean()) for t, h in [("all", p3[p3["position"] == pos])]
                                + list(p3[p3["position"] == pos].groupby("task_type"))}
                          for pos in ("start", "middle", "end")}

    keep = {fam: set(g.loc[g["matched"], "id"]) for fam, g in vis.groupby("family")}
    results = {}
    frames = {m: df[df["eval_set"].isin(SETS)].reset_index(drop=True) for m, (df, _) in found.items()}
    if all(m in frames for m in MEAN_SEEDS):
        frames["deberta_mean"] = seed_mean_frame(frames)
    for m, df in frames.items():
        fam = FAMILY.get(m)
        sub = df if fam is None else df[df["id"].isin(keep[fam])].reset_index(drop=True)
        fns = (mean_flag_rate, mean_f1) if m == "deberta_mean" else ()
        full, mt = model_report(df, *fns), model_report(sub, *fns)
        results[m] = {"family": fam, "rows_full": len(df), "rows_matched": len(sub),
                      "full": full, "matched": mt,
                      "contrast_change_matched_minus_full": {
                          k: mt["contrasts"][k]["diff"] - full["contrasts"][k]["diff"]
                          for k in CONTRAST_KEYS if k in mt["contrasts"] and k in full["contrasts"]}}
        run.log(f"{m}: rows {len(sub)}/{len(df)}; P3 end-start full {full['contrasts']['p3_end_minus_start']['diff']:+.4f} "
                f"matched {mt['contrasts']['p3_end_minus_start']['diff']:+.4f}")

    run.save_metrics({"survival": survival, "p3_truncated_fraction_by_position": trunc_end,
                      "max_tokens": {fam: int(g["n_tokens"].max()) for fam, g in vis.groupby("family")},
                      "models": results},
                     headline=("deberta_mean_p3_end_minus_start_matched",
                               results.get("deberta_mean", {}).get("matched", {}).get("contrasts", {})
                               .get("p3_end_minus_start", {}).get("diff", float("nan"))))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
