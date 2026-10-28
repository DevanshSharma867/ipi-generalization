"""
v3_integrity.py
===============
Independent integrity check of built V3 split realizations, read back from the written CSVs (the builder's own
`data_pipeline_v3.check` runs in memory on the main frame only). Per split seed:

- doc_id, exact text, near-duplicate component: disjoint across train/val/test (main), and every eval-slice row
  belongs to a doc of the split it claims (val_id: val docs; raw_clean/ambiguous: val or test docs).
- attack strings (injected span_text) disjoint across train/val/test main splits; attack categories of train and
  val disjoint; test attacks only from the *_attack_test.json files, train/val only from *_attack_train.json.
  (val_id deliberately pairs val docs with train-category attacks; it is checked for exactly that.)
- benign items (clean_a span_source) and donor sentences/chunks (clean_b span_text) disjoint across splits;
  clean_b donor docs come from the same task and split; ambiguous items never appear as clean_a items.
- exact 1:1 balance per task x split; 4 rows per doc, 2 injected.
- <= 512 DeBERTa-v3 tokens for every main, val_id, raw_clean and ambiguous row, re-tokenized here (the `long` slice
  is over-length by design); the stored n_tokens column must match.
- every span recoverable from char_offset/span_len.

    python scripts/v3_integrity.py --data_dir data/v3 --split_seeds 0 1 2 3 4 [--data_version v3]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from data_pipeline_v3 import MAX_TOKENS, TOKENIZER
from preprocess import build_combined_text
from runlog import add_run_args, start_run

SPLITS = ("train", "val", "test")
SLICES = ("val_id", "raw_clean", "ambiguous", "long")


def load(d: Path):
    main = pd.concat([pd.read_csv(d / f"{s}.csv", keep_default_na=False) for s in SPLITS], ignore_index=True)
    sl = {n: pd.read_csv(d / f"{n}.csv", keep_default_na=False) for n in SLICES}
    return main, sl


def checks(main: pd.DataFrame, sl: dict, tok) -> dict:
    """{check_name: list of violation strings}; empty lists mean pass."""
    v = {}
    by = {s: main[main["split"] == s] for s in SPLITS}
    pairs = [(a, b) for i, a in enumerate(SPLITS) for b in SPLITS[i + 1:]]

    def disjoint(name, rows_fn, col):
        out = []
        for a, b in pairs:
            inter = set(rows_fn(by[a])[col]) & set(rows_fn(by[b])[col])
            if inter:
                out.append(f"{a}/{b}: {len(inter)} shared, e.g. {sorted(map(str, inter))[:2]}")
        v[name] = out

    disjoint("doc_id_disjoint", lambda d: d, "doc_id")
    disjoint("text_disjoint", lambda d: d, "text")
    disjoint("component_disjoint", lambda d: d, "component")
    disjoint("attack_string_disjoint", lambda d: d[d["row_type"] == "injected"], "span_text")
    disjoint("benign_item_disjoint", lambda d: d[d["row_type"] == "clean_a"], "span_source")
    disjoint("donor_text_disjoint", lambda d: d[d["row_type"] == "clean_b"], "span_text")

    inj = main[main["row_type"] == "injected"]
    cats = {s: set(inj[inj["split"] == s]["attack_category"]) for s in SPLITS}
    v["train_val_category_disjoint"] = [f"shared: {sorted(cats['train'] & cats['val'])}"] \
        if cats["train"] & cats["val"] else []
    src = inj.groupby("split")["attack_source"].agg(lambda x: sorted(set(x))).to_dict()
    v["attack_file_by_split"] = [f"{s}: {src.get(s)}" for s in SPLITS
                                 if any(("_test" in f) != (s == "test") for f in src.get(s, []))]

    split_of = main.drop_duplicates("doc_id").set_index("doc_id")["split"]
    donor_doc = main[main["row_type"] == "clean_b"]
    bad = []
    for r in donor_doc.itertuples():
        dd = r.span_source.removeprefix("donor:")
        if dd in split_of and split_of[dd] != r.split:
            bad.append(f"{r.id}: donor {dd} in {split_of[dd]}")
        if not dd.startswith(r.task_type):
            bad.append(f"{r.id}: donor {dd} from another task")
    v["donor_doc_same_split_task"] = bad[:5]

    amb_items = set(sl["ambiguous"]["span_source"])
    v["ambiguous_not_benign"] = sorted(amb_items & set(main[main["row_type"] == "clean_a"]["span_source"]))[:5]

    bal = main.groupby(["task_type", "split", "label"]).size().unstack("label")
    v["balance_1to1"] = [f"{i}: {r.to_dict()}" for i, r in bal.iterrows() if r.iloc[0] != r.iloc[1]]
    per_doc = main.groupby("doc_id").agg(n=("id", "size"), inj=("label", lambda x: (x.astype(int) == 1).sum()))
    v["four_rows_two_injected"] = per_doc[(per_doc["n"] != 4) | (per_doc["inj"] != 2)].index.tolist()[:5]

    slice_bad = []
    for name in ("val_id", "raw_clean", "ambiguous"):
        for r in sl[name].itertuples():
            if r.doc_id not in split_of or split_of[r.doc_id] != r.split:
                slice_bad.append(f"{name} {r.id}: doc split {split_of.get(r.doc_id)} vs row {r.split}")
    vi = sl["val_id"]
    slice_bad += [f"val_id {r.id}: attack {r.attack_source}" for r in vi.itertuples()
                  if r.split != "val" or "_test" in r.attack_source or r.attack_category in cats["val"]]
    v["eval_slice_membership"] = slice_bad[:5]

    tok_bad, n_mismatch, n_rows = [], 0, 0
    for name, df in [("main", main)] + [(n, sl[n]) for n in ("val_id", "raw_clean", "ambiguous")]:
        if df.empty:
            continue
        n = [len(x) for x in tok([build_combined_text(r) for r in df.to_dict("records")])["input_ids"]]
        n_rows += len(n)
        n_mismatch += int((pd.Series(n, index=df.index) != df["n_tokens"].astype(int)).sum())
        tok_bad += [f"{name} {i}: {k}" for i, k in zip(df["id"], n) if k > MAX_TOKENS]
    v["max_512_tokens"] = tok_bad[:5]
    v["n_tokens_column_matches"] = [f"{n_mismatch} of {n_rows} rows differ"] if n_mismatch else []

    spans = [r.id for df in (main, sl["val_id"], sl["ambiguous"]) for r in df.itertuples()
             if r.text[int(r.char_offset):int(r.char_offset) + int(r.span_len)] != r.span_text]
    v["span_recoverable"] = spans[:5]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "v3"))
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data_version", default="v3")
    ap.add_argument("--exp", default="integrity")
    a = add_run_args(ap).parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    dirs = {s: Path(a.data_dir) / f"split_seed_{s}" for s in a.split_seeds}
    files = [d / f"{n}.csv" for d in dirs.values() for n in SPLITS + SLICES] + [d / "manifest.json" for d in dirs.values()]
    run = start_run(version="v3", exp=a.exp, seed=a.split_seeds[0],
                    args={"data_dir": a.data_dir, "split_seeds": a.split_seeds, "max_tokens": MAX_TOKENS,
                          "tokenizer": str(TOKENIZER.relative_to(ROOT))},
                    data_version=a.data_version, data_files=files, allow_dirty=a.allow_dirty)
    out, ok = {}, True
    for s, d in dirs.items():
        main_df, sl = load(d)
        res = checks(main_df, sl, tok)
        out[str(s)] = {"violations": res, "pass": not any(res.values()),
                       "rows": {"main": len(main_df), **{n: len(df) for n, df in sl.items()}},
                       "main_by_task_split": {f"{t}/{sp}": int(n) for (t, sp), n in
                                              main_df.groupby(["task_type", "split"]).size().items()}}
        ok &= out[str(s)]["pass"]
        for name, viol in res.items():
            run.log(f"[{s}] {name:30s} {'PASS' if not viol else 'FAIL ' + '; '.join(map(str, viol))}")
    run.log(f"integrity: {'PASS' if ok else 'FAIL'}")
    run.save_metrics({"seeds": out, "pass": ok}, headline=("integrity_pass", float(ok)))
    run.finish()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
