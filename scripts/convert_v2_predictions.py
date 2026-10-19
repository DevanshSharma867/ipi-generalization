"""
convert_v2_predictions.py
=========================
Convert every frozen V2 predictions/scores CSV into the standard predictions schema
(src/runlog.PRED_COLUMNS) under results/v2_std/<eval_set>/<model>.csv. Originals are only read.

Missing fields are joined by id from the dataset each file was scored on
(data/processed/{test,val}.csv, data/pairs/pairs_v0.csv, data/external/*.csv).
run_id = "v2-frozen:<source path>". A manifest with source sha256 is written to results/v2_std/manifest.json.

    python scripts/convert_v2_predictions.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from runlog import assert_not_frozen, sha256, validate_predictions, write_json

OUT = ROOT / "results" / "v2_std"
RES = ROOT / "results"
SPLIT_SEED = 42  # data_pipeline.py default; the only V2 split


def v2_fields(split: str) -> pd.DataFrame:
    d = pd.read_csv(ROOT / f"data/processed/{split}.csv").set_index("id")
    raw = d["attack_type"] == "none"
    inj = d["label"] == 1
    return pd.DataFrame({
        "doc_id": d["doc_id"], "split_seed": SPLIT_SEED, "task": d["task_type"], "label": d["label"],
        "attack_category": d["attack_type"].where(inj, ""),
        "attack_source": d["attack_source"].where(~raw, ""),
        "insertion_mode": d["insertion_mode"].where(~raw, ""),
        "clean_type": pd.Series("", index=d.index).mask(~inj & raw, "raw").mask(~inj & ~raw, "benign_imperative"),
    })


def pairs_fields() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/pairs/pairs_v0.csv").set_index("id")
    return pd.DataFrame({
        "doc_id": d["doc_id"], "split_seed": SPLIT_SEED, "task": d["task_type"], "label": d["label"],
        "attack_category": "", "attack_source": "pair_specs_v0.json",
        "insertion_mode": "end",  # score_detectors.build_pairs appends the instruction after the document
        "clean_type": d["label"].map({0: "aligned_query", 1: ""}),
    })


def injecagent_fields() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/external/injecagent.csv").set_index("id")
    inj = d["label"] == 1
    return pd.DataFrame({
        "doc_id": "", "split_seed": "", "task": d["task_type"], "label": d["label"],
        "attack_category": d["attack_type"].where(inj, ""), "attack_source": d["label"].map({1: "InjecAgent", 0: ""}),
        "insertion_mode": "",
        "clean_type": d["attack_type"].map({"none": "benign_statement", "benign_imperative": "benign_imperative"})
                                      .where(~inj, ""),
    })


def deepset_fields() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data/external/deepset.csv").set_index("id")
    return pd.DataFrame({
        "doc_id": "", "split_seed": "", "task": d["task_type"], "label": d["label"],
        "attack_category": "", "attack_source": "", "insertion_mode": "",
        "clean_type": d["label"].map({0: "raw", 1: ""}),
    })


FIELDS = {"v2_test": lambda: v2_fields("test"), "v2_val": lambda: v2_fields("val"),
          "pairs_v0": pairs_fields, "injecagent": injecagent_fields, "deepset": deepset_fields}


def build(src: Path, ids, prob, pred, threshold, model, eval_set, fields_key) -> pd.DataFrame:
    f = FIELDS[fields_key]().loc[list(ids)]
    df = f.reset_index().rename(columns={"index": "id"})
    df["prob"] = list(prob)
    df["pred"] = [int(p) for p in pred]
    df["threshold"] = threshold
    df["model"] = model
    df["run_id"] = f"v2-frozen:{src.relative_to(ROOT)}"
    df["eval_set"] = eval_set
    return validate_predictions(df)


def sources():
    """Yield (source, label column check frame, builder kwargs) for every V2 predictions/scores file."""
    tf = json.load(open(RES / "tfidf_lr_results.json"))
    tuned_thr = tf["val_tuning"]["optimal_config"]["threshold"]
    best_c = int(tf["val_tuning"]["optimal_config"]["C"])  # tuned C; run_experiments also reuses it for LOTO

    src = RES / "test_predictions.csv"
    d = pd.read_csv(src)
    yield src, d, dict(prob=d.prob_injected, pred=d.pred_default, threshold=0.5, model="tfidf_c1",
                       eval_set="v2_test", fields_key="v2_test")
    yield src, d, dict(prob=d.prob_injected_tuned, pred=d.pred_tuned, threshold=tuned_thr,
                       model=f"tfidf_c{best_c}_tuned", eval_set="v2_test", fields_key="v2_test")

    for task in ("email", "table", "code"):
        src = RES / f"loto_predictions_{task}.csv"
        d = pd.read_csv(src)
        yield src, d, dict(prob=d.prob_injected, pred=d.pred, threshold=0.5, model=f"tfidf_c{best_c}_loto",
                           eval_set=f"v2_loto_test_{task}", fields_key="v2_test")

    for run_dir in sorted((RES / "deberta_predictions").iterdir()):
        name = run_dir.name  # s42 | loto_email | loto_email_s123 | sub496_s42
        parts = name.split("_")
        if name.startswith("loto_"):
            task, seed = parts[1], (parts[2][1:] if len(parts) > 2 else "42")
            model = f"deberta_loto_s{seed}"
        elif name.startswith("sub496_"):
            task, seed, model = None, parts[1][1:], f"deberta_sub496_s{parts[1][1:]}"
        else:
            task, seed, model = None, name[1:], f"deberta_s{name[1:]}"
        for split in ("test", "val"):
            src = run_dir / f"{split}_predictions.csv"
            if not src.exists():
                continue
            d = pd.read_csv(src)
            eval_set = f"v2_{split}" if task is None else (f"v2_loto_test_{task}" if split == "test" else f"v2_loto_val_{task}")
            yield src, d, dict(prob=d.prob_injected, pred=d.pred, threshold=0.5, model=model,
                               eval_set=eval_set, fields_key=f"v2_{split}")

    for ds_dir in sorted((RES / "scores").iterdir()):
        if not ds_dir.is_dir():
            continue
        ds = ds_dir.name
        eval_set = {"bipia_test": "v2_test", "pairs": "pairs_v0"}.get(ds, ds)
        for src in sorted(ds_dir.glob("*.csv")):
            d = pd.read_csv(src)
            model = src.stem + ("_rescore" if ds == "bipia_test" and src.stem.startswith(("deberta", "tfidf")) else "")
            model = {"tfidf_rescore": "tfidf_c1_rescore"}.get(model, model)
            yield src, d, dict(prob=d.score, pred=(d.score >= 0.5).astype(int), threshold=0.5, model=model,
                               eval_set=eval_set, fields_key=eval_set)


def main():
    assert_not_frozen(OUT)
    manifest = []
    for src, d, kw in sources():
        out = build(src, d["id"], **kw)
        if not (out["label"].values == d["label"].values).all():
            raise ValueError(f"label mismatch after join: {src}")
        dest = OUT / kw["eval_set"] / f"{kw['model']}.csv"
        assert_not_frozen(dest)
        if dest.exists() and any(m["dest"] == str(dest.relative_to(ROOT)) for m in manifest):
            raise FileExistsError(f"two sources map to {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        manifest.append({"source": str(src.relative_to(ROOT)), "source_sha256": sha256(src),
                         "dest": str(dest.relative_to(ROOT)), "rows": len(out), "model": kw["model"],
                         "eval_set": kw["eval_set"], "threshold": kw["threshold"]})
        print(f"{src.relative_to(ROOT)} -> {dest.relative_to(ROOT)} ({len(out)} rows)")
    write_json(OUT / "manifest.json", manifest)
    print(f"{len(manifest)} files converted")


if __name__ == "__main__":
    main()
