"""
artifact_baselines.py
=====================
Artifact-only (content-free) baselines on a test set; learned probes are trained on the matching train split only.

  position_oracle      pred = 1 iff insertion_mode == "start" (metadata)
  first_para_rule      pred = 1 iff the first "\\n\\n"-paragraph of the text has <= 30 words (untrained)
  prefix30_tfidf_lr    TF-IDF+LR (frozen V2 config, C=1) on the first 30 whitespace tokens only
  suffix30_tfidf_lr    same on the last 30 tokens
  span_form_lr         LR on the 9 V2 content-free features of the INSERTED span only
  span_form_you_lr     the same 9 features + "contains you/your" (D6)
  length_lr            LR on n_chars and n_words of the full text
  second_person_lr     LR on "span contains you/your" only (D6)
  boundary_kind_lr     LR on one-hot boundary_kind only (v3; V2 has no boundary_kind)

--dataset v2: data/processed, spans recovered with probes.extract_inserted_span; span probes are evaluated on the
216 test rows with a span. --dataset v3: <data_dir>/split_seed_<k>/{train,test}.csv; every main row has a span
(text[char_offset:char_offset+span_len]). Every probe is reported pooled and per task, each with a doc-cluster
bootstrap CI (B=1000, seed 42); for v3 the gates (macro-F1 <= 0.60 pooled and per task) are checked and listed.

    python scripts/artifact_baselines.py                                   # V2, results/v2_probes/
    python scripts/artifact_baselines.py --dataset v3 --split_seed 0 [--data_dir data/v3] [--exp artifact_baselines]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "models"), str(ROOT / "scripts")]
from metrics import df_metrics, doc_cluster_bootstrap
from preprocess import preprocess_for_tfidf
from probes import extract_inserted_span, form_tags
from runlog import start_run
from tfidf_lr import LR_PARAMS, TFIDF_PARAMS
from convert_v2_predictions import v2_fields

DATA = ROOT / "data" / "processed"
N_TOK = 30
GATE = 0.60
CI_KEYS = ["f1_macro", "roc_auc", "accuracy", "recall", "flag_rate_clean"]
SPAN_FEATURES = ["n_words", "n_chars", "ends_q", "ends_dot", "ends_excl", "has_quote", "has_digit",
                 "has_newline_or_code_char", "cap_ratio"]


def with_spans(df: pd.DataFrame) -> pd.DataFrame:
    ctx = df[df["attack_type"] == "none"].set_index("doc_id")["text"]
    df = df.assign(context=df["doc_id"].map(ctx))
    df["span"] = [None if m == "none" else extract_inserted_span(t, c, m)
                  for t, c, m in zip(df["text"], df["context"], df["insertion_mode"])]
    return df


def span_features(spans: pd.Series, names=SPAN_FEATURES) -> np.ndarray:
    return np.array([[float(form_tags(x)[k]) for k in names] for x in spans], dtype=float)


def first_n(text: str, n=N_TOK) -> str:
    return " ".join(text.split()[:n])


def last_n(text: str, n=N_TOK) -> str:
    return " ".join(text.split()[-n:])


def tfidf_probe(train, test, view):
    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr = vec.fit_transform(preprocess_for_tfidf([view(t) for t in train["text"]]))
    X_te = vec.transform(preprocess_for_tfidf([view(t) for t in test["text"]]))
    lr = LogisticRegression(**LR_PARAMS).fit(X_tr, train["label"])
    return lr.predict_proba(X_te)[:, 1]


def standard(test: pd.DataFrame, prob, pred, model: str, eval_set: str) -> pd.DataFrame:
    if "row_type" in test:  # v3 rows carry their own fields
        f = test[["id", "doc_id", "split_seed", "task_type", "label", "attack_category", "attack_source",
                  "insertion_mode", "clean_type"]].rename(columns={"task_type": "task"}).fillna("")
    else:
        f = v2_fields("test").loc[test["id"]].reset_index().rename(columns={"index": "id"})
    return f.reset_index(drop=True).assign(prob=np.asarray(prob, float), pred=np.asarray(pred, int), threshold=0.5,
                                           model=model, eval_set=eval_set)


def lr_probe(X_tr, y_tr, X_te, scale=True):
    clf = make_pipeline(StandardScaler(), LogisticRegression(**{**LR_PARAMS, "n_jobs": None})) if scale \
        else LogisticRegression(**{**LR_PARAMS, "n_jobs": None})
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1], clf


def load(dataset, data_dir, split_seed):
    if dataset == "v2":
        train, test = (with_spans(pd.read_csv(DATA / f"{s}.csv")) for s in ("train", "test"))
        return train, test, [DATA / "train.csv", DATA / "test.csv"], "v2_test"
    d = Path(data_dir) / f"split_seed_{split_seed}"
    train, test = (pd.read_csv(d / f"{s}.csv", keep_default_na=False) for s in ("train", "test"))
    for df in (train, test):
        df["span"] = [t[int(o):int(o) + int(n)] for t, o, n in zip(df["text"], df["char_offset"], df["span_len"])]
        assert (df["span"] == df["span_text"]).all()
    return train, test, [d / "train.csv", d / "test.csv", d / "manifest.json"], "v3_test"


def evaluate(df):
    boot = doc_cluster_bootstrap(df, lambda d: {k: df_metrics(d)[k] for k in CI_KEYS})
    per_task = {t: {"n": len(g), "bootstrap": doc_cluster_bootstrap(g.reset_index(drop=True),
                                                                      lambda d: {"f1_macro": df_metrics(d)["f1_macro"]})}
                for t, g in df.groupby("task")}
    return {"eval_set": df["eval_set"].iloc[0], "n": len(df), "point": df_metrics(df), "bootstrap": boot,
            "per_task": per_task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("v2", "v3"), default="v2")
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "v3"))
    ap.add_argument("--data_version", default=None)
    ap.add_argument("--exp", default="artifact_baselines")
    a = ap.parse_args()
    train, test, files, eval_set = load(a.dataset, a.data_dir, a.split_seed)
    version = "v2_probes" if a.dataset == "v2" else "v3"
    run = start_run(version=version, exp=a.exp, seed=42 if a.dataset == "v2" else a.split_seed,
                    args={"dataset": a.dataset, "split_seed": a.split_seed, "data_dir": str(a.data_dir),
                          "n_tokens": N_TOK, "tfidf": TFIDF_PARAMS, "lr": LR_PARAMS, "span_features": SPAN_FEATURES,
                          "gate": GATE, "bootstrap": {"B": 1000, "seed": 42}},
                    data_version=a.data_version or a.dataset, data_files=files)
    run.log(f"rows with a span: train {train.span.notna().sum()}, test {test.span.notna().sum()}")

    preds = []
    oracle = (test["insertion_mode"] == "start").astype(int)
    preds.append(standard(test, oracle, oracle, "position_oracle", eval_set))
    rule = test["text"].apply(lambda t: len(t.split("\n\n")[0].split()) <= 30).astype(int)
    preds.append(standard(test, rule, rule, "first_para_rule", eval_set))
    for name, view in (("prefix30_tfidf_lr", first_n), ("suffix30_tfidf_lr", last_n)):
        p = tfidf_probe(train, test, view)
        preds.append(standard(test, p, p >= 0.5, name, eval_set))
    length = lambda d: np.array([[len(t), len(t.split())] for t in d["text"]], dtype=float)
    p, _ = lr_probe(length(train), train["label"], length(test))
    preds.append(standard(test, p, p >= 0.5, "length_lr", eval_set))

    tr_s, te_s = train[train.span.notna()].reset_index(drop=True), test[test.span.notna()].reset_index(drop=True)
    span_set = eval_set if a.dataset == "v3" else "v2_test_inserted_spans"
    coefs = {}
    for name, feats in (("span_form_lr", SPAN_FEATURES), ("span_form_you_lr", SPAN_FEATURES + ["has_you"]),
                        ("second_person_lr", ["has_you"])):
        p, clf = lr_probe(span_features(tr_s.span, feats), tr_s["label"], span_features(te_s.span, feats))
        preds.append(standard(te_s, p, p >= 0.5, name, span_set))
        coefs[name] = dict(zip(feats, clf[-1].coef_[0].round(4).tolist()))
    if "boundary_kind" in test:
        enc = OneHotEncoder(handle_unknown="ignore").fit(train[["boundary_kind"]])
        p, _ = lr_probe(enc.transform(train[["boundary_kind"]]), train["label"], enc.transform(test[["boundary_kind"]]),
                        scale=False)
        preds.append(standard(test, p, p >= 0.5, "boundary_kind_lr", eval_set))

    results = {}
    for df in preds:
        name = df["model"].iloc[0]
        results[name] = evaluate(df)
        b = results[name]["bootstrap"]["f1_macro"]
        tasks = "  ".join(f"{t} {v['bootstrap']['f1_macro']['point']:.4f}" for t, v in results[name]["per_task"].items())
        run.log(f"{name:20s} n={len(df):4d} macro-F1 {b['point']:.4f} [{b['ci'][0]:.4f}, {b['ci'][1]:.4f}]  {tasks}")
    for name, c in coefs.items():
        results[name]["standardized_coefficients"] = c
    if a.dataset == "v3":
        fails = [f"{n}/{t}" for n, r in results.items()
                 for t, v in [("pooled", r["bootstrap"]["f1_macro"]["point"])] +
                 [(t, x["bootstrap"]["f1_macro"]["point"]) for t, x in r["per_task"].items()] if v > GATE]
        results["gate"] = {"threshold": GATE, "failures": fails, "pass": not fails}
        # Diagnostics: where the window probes get their signal, and which span features differ by row type
        by_pos = {}
        for df in preds:
            name = df["model"].iloc[0]
            by_pos[name] = {f"{t}/{m}": float(df_metrics(g)["f1_macro"]) for (t, m), g in
                            df.assign(task=df["task"].astype(str)).groupby(["task", "insertion_mode"])}
        feats = SPAN_FEATURES + ["has_you"]
        means = pd.DataFrame(span_features(te_s.span, feats), columns=feats).assign(
            task=te_s["task_type"].values, row_type=te_s["row_type"].values).groupby(["task", "row_type"]).mean()
        results["diagnostics"] = {"f1_macro_by_task_and_position": by_pos,
                                  "test_span_feature_means": {f"{t}/{r}": row.round(4).to_dict()
                                                              for (t, r), row in means.iterrows()}}
        run.log(f"gates: {'PASS' if not fails else 'FAIL ' + ', '.join(fails)}")
    run.save_predictions(pd.concat(preds, ignore_index=True))
    run.save_metrics(results, headline=("position_oracle_f1_macro", results["position_oracle"]["point"]["f1_macro"]))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
