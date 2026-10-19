"""
artifact_baselines.py
=====================
Artifact-only baselines on the V2 test set. Learned probes are trained on V2 train only.

  a. position_oracle      pred = 1 iff insertion_mode == "start" (metadata; upper bound of the pure position signal)
  b. first_para_rule      pred = 1 iff the first "\\n\\n"-paragraph of the context has <= 30 words (untrained)
  c. prefix30_tfidf_lr    TF-IDF+LR (frozen V2 config, C=1) on the first 30 whitespace tokens of the context only
     suffix30_tfidf_lr    same on the last 30 tokens
  d. span_form_lr         LR on 9 content-free features of the INSERTED span only (benign template vs attack);
                          evaluated on the 216 test rows that have an inserted span

Spans are recovered exactly from each doc's raw context with probes.extract_inserted_span (asserted on every row).
Doc-cluster bootstrap CIs (B=1000, seed 42). Logged with runlog to results/v2_probes/artifact_baselines/<run_id>/.

    python scripts/artifact_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "models"), str(ROOT / "scripts")]
from metrics import df_metrics, doc_cluster_bootstrap
from preprocess import preprocess_for_tfidf
from probes import extract_inserted_span
from runlog import start_run
from tfidf_lr import LR_PARAMS, TFIDF_PARAMS
from convert_v2_predictions import v2_fields

DATA = ROOT / "data" / "processed"
N_TOK = 30
CODE_CHARS = set("{}()[];=<>`\\")
CI_KEYS = ["f1_macro", "roc_auc", "accuracy", "recall", "flag_rate_clean"]


def with_spans(df: pd.DataFrame) -> pd.DataFrame:
    ctx = df[df["attack_type"] == "none"].set_index("doc_id")["text"]
    df = df.assign(context=df["doc_id"].map(ctx))
    df["span"] = [None if m == "none" else extract_inserted_span(t, c, m)
                  for t, c, m in zip(df["text"], df["context"], df["insertion_mode"])]
    return df


def span_features(spans: pd.Series) -> np.ndarray:
    def f(s: str):
        st = s.strip()
        letters = [ch for ch in st if ch.isalpha()]
        return [len(st.split()), len(st), st.endswith("?"), st.endswith("."), st.endswith("!"),
                any(q in st for q in "\"'"), any(ch.isdigit() for ch in st),
                ("\n" in st) or any(ch in CODE_CHARS for ch in st),
                sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0]
    return np.array([f(s) for s in spans], dtype=float)


SPAN_FEATURES = ["n_words", "n_chars", "ends_q", "ends_dot", "ends_excl", "has_quote", "has_digit",
                 "has_newline_or_code_char", "cap_ratio"]


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
    f = v2_fields("test").loc[test["id"]].reset_index().rename(columns={"index": "id"})
    return f.assign(prob=np.asarray(prob, float), pred=np.asarray(pred, int), threshold=0.5,
                    model=model, eval_set=eval_set)


def main():
    train, test = (with_spans(pd.read_csv(DATA / f"{s}.csv")) for s in ("train", "test"))
    run = start_run(version="v2_probes", exp="artifact_baselines", seed=42,
                    args={"n_tokens": N_TOK, "tfidf": TFIDF_PARAMS, "lr": LR_PARAMS, "span_features": SPAN_FEATURES,
                          "bootstrap": {"B": 1000, "seed": 42}},
                    data_version="v2", data_files=[DATA / "train.csv", DATA / "test.csv"])
    run.log(f"spans recovered exactly: train {train.span.notna().sum()}, test {test.span.notna().sum()}")

    preds = []
    oracle = (test["insertion_mode"] == "start").astype(int)
    preds.append(standard(test, oracle, oracle, "position_oracle", "v2_test"))
    rule = test["text"].apply(lambda t: len(t.split("\n\n")[0].split()) <= 30).astype(int)
    preds.append(standard(test, rule, rule, "first_para_rule", "v2_test"))
    for name, view in (("prefix30_tfidf_lr", first_n), ("suffix30_tfidf_lr", last_n)):
        p = tfidf_probe(train, test, view)
        preds.append(standard(test, p, p >= 0.5, name, "v2_test"))

    tr_s, te_s = train[train.span.notna()], test[test.span.notna()]
    clf = make_pipeline(StandardScaler(), LogisticRegression(**{**LR_PARAMS, "n_jobs": None}))
    clf.fit(span_features(tr_s.span), tr_s["label"])
    p = clf.predict_proba(span_features(te_s.span))[:, 1]
    preds.append(standard(te_s, p, p >= 0.5, "span_form_lr", "v2_test_inserted_spans"))
    coefs = dict(zip(SPAN_FEATURES, clf[-1].coef_[0].round(4).tolist()))

    results = {}
    for df in preds:
        name = df["model"].iloc[0]
        boot = doc_cluster_bootstrap(df, lambda d: {k: df_metrics(d)[k] for k in CI_KEYS})
        results[name] = {"eval_set": df["eval_set"].iloc[0], "n": len(df), "point": df_metrics(df), "bootstrap": boot}
        run.log(f"{name:20s} n={len(df):3d} macro-F1 {boot['f1_macro']['point']:.4f} "
                f"[{boot['f1_macro']['ci'][0]:.4f}, {boot['f1_macro']['ci'][1]:.4f}]  ROC-AUC {boot['roc_auc']['point']:.4f} "
                f"[{boot['roc_auc']['ci'][0]:.4f}, {boot['roc_auc']['ci'][1]:.4f}]")
    results["span_form_lr"]["standardized_coefficients"] = coefs
    run.save_predictions(pd.concat(preds, ignore_index=True))
    run.save_metrics(results, headline=("position_oracle_f1_macro", results["position_oracle"]["point"]["f1_macro"]))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
