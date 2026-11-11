"""
v3_sanity.py
============
Sanity checks on the V3 DeBERTa result (read-only: no data or model is changed). Inputs: data/v3/split_seed_<k>
(v3-frozen) and the latest clean results/v3/score_deberta_v3/<run> per split seed.

1. Attack strings: unique injected span texts per split (train, val, val_id, test) and split seed; exact and
   near-duplicate overlap of test (and val) attack strings with the train attack strings (char 5-gram Jaccard
   >= 0.5 on lowercased, whitespace-collapsed text); V3 DeBERTa test macro-F1 (task-macro and pooled) on all test
   rows and with the injected rows whose attack has a train near-duplicate removed.
2. Per-attack-string recall on V3 test at 0.5 (rows of all split seeds), its distribution and per category; overall
   recall with a 95% CI from resampling attack strings (clusters = attack strings, B = 1000).
3. Style probe on the inserted span alone (also by feature group: function words, punctuation, casing and length): logistic regression on style features without content words
   (function-word rates from the NLTK English stopword list, punctuation and newline rates, casing, word / char /
   sentence lengths, digit rate), trained per split seed on train spans (benign-pool clean-A items vs attacks) and
   scored on that seed's test spans; also the variant with clean-B spans among the negatives (the same rows as the
   span-form gate, F024). Macro-F1 at 0.5 and AUROC, run-mean with doc-cluster CIs (stratified by task).
4. Val (held-out categories): exact / near-duplicate overlap of val attack strings with train attack strings,
   shared categories, and the V3 DeBERTa val macro-F1 per split seed (from the training runs' selection metric).

    python scripts/v3_sanity.py
"""

import argparse
import json
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import preprocess  # noqa: E402,F401  (sets the NLTK data path: the repo copy, as for TF-IDF)
from metrics import _resample_indices  # noqa: E402
from runlog import add_run_args, start_run  # noqa: E402
from v3_baselines_report import boot, latest_runs, run_mean, task_macro  # noqa: E402

SEEDS = range(5)
TOP_COEFS = []
JACCARD = 0.5
B, SEED = 1000, 42


def norm(s):
    return " ".join(str(s).lower().split())


def grams(s, n=5):
    s = norm(s)
    return frozenset(s[i:i + n] for i in range(max(1, len(s) - n + 1)))


def max_jaccard(queries, refs):
    """For each query string, the largest char 5-gram Jaccard to any reference string."""
    rg = [grams(r) for r in refs]
    out = {}
    for q in queries:
        g = grams(q)
        out[q] = max((len(g & r) / len(g | r) for r in rg), default=0.0)
    return out


def load(k):
    d = ROOT / f"data/v3/split_seed_{k}"
    main = pd.concat([pd.read_csv(d / f"{s}.csv", keep_default_na=False) for s in ("train", "val", "test")])
    vid = pd.read_csv(d / "val_id.csv", keep_default_na=False)
    return main, vid


def macro_f1(d):
    return float(f1_score(d["label"].astype(int), d["pred"].astype(int), average="macro"))


# ─── 3. style features (no content words) ────────────────────────────────────

def style_features(spans):
    from nltk.corpus import stopwords
    fw = sorted(set(stopwords.words("english")))
    punct = list(".,!?;:'\"()[]{}-*`/_#@&%$<>=|\\") + ["\n"]
    rows = []
    for s in spans:
        words = re.findall(r"[A-Za-z']+", s)
        lw = [w.lower() for w in words]
        nw = max(1, len(words))
        nc = max(1, len(s))
        sents = [x for x in re.split(r"[.!?]+(?:\s|$)", s) if x.strip()]
        f = {f"fw_{w}": lw.count(w) / nw for w in fw}
        f.update({f"p_{ord(c)}": s.count(c) / nc for c in punct})
        f.update({"upper_char": sum(ch.isupper() for ch in s) / nc,
                  "title_word": sum(w[:1].isupper() for w in words) / nw,
                  "caps_word": sum(w.isupper() and len(w) > 1 for w in words) / nw,
                  "starts_upper": float(s.strip()[:1].isupper()),
                  "n_words": len(words), "n_chars": len(s), "mean_word_len": np.mean([len(w) for w in words]) if words else 0.0,
                  "n_sents": len(sents), "mean_sent_words": len(words) / max(1, len(sents)),
                  "digit": sum(ch.isdigit() for ch in s) / nc, "space": sum(ch in string.whitespace for ch in s) / nc})
        rows.append(f)
    return pd.DataFrame(rows).fillna(0.0)


GROUPS = {"all": None, "function_words": "fw_", "punctuation": "p_",
          "casing_length": ("upper_char", "title_word", "caps_word", "starts_upper", "n_words", "n_chars",
                            "mean_word_len", "n_sents", "mean_sent_words", "digit", "space")}


def select(X, group):
    g = GROUPS[group]
    if g is None:
        return X
    cols = [c for c in X.columns if c.startswith(g)] if isinstance(g, str) else list(g)
    return X[cols]


def style_probe(main, with_clean_b, group="all"):
    def pick(split):
        d = main[(main["split"] == split) & (main["row_type"].isin(["injected", "clean_a"] + (["clean_b"] if with_clean_b else [])))]
        return d.reset_index(drop=True)
    tr, te = pick("train"), pick("test")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=0))
    clf.fit(select(style_features(tr["span_text"]), group), tr["label"].astype(int))
    p = clf.predict_proba(select(style_features(te["span_text"]), group))[:, 1]
    if group == "all" and not with_clean_b:  # the ten largest standardized coefficients, for interpretation
        lr = clf[-1]
        cols = select(style_features(tr["span_text"].head(1)), group).columns
        top = sorted(zip(cols, lr.coef_[0]), key=lambda x: -abs(x[1]))[:10]
        TOP_COEFS.append([(c, float(w)) for c, w in top])
    return pd.DataFrame({"doc_id": te["doc_id"], "task": te["task_type"], "label": te["label"].astype(int),
                         "prob": p, "pred": (p >= 0.5).astype(int)})


def auroc(d):
    return float(roc_auc_score(d["label"], d["prob"])) if d["label"].nunique() == 2 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a = add_run_args(ap).parse_args()
    runs = latest_runs("v3", "score_deberta_v3")
    trains = latest_runs("v3", "deberta_v3_train")
    if sorted(runs) != list(SEEDS) or sorted(trains) != list(SEEDS):
        sys.exit(f"need V3 DeBERTa scoring and training runs for split seeds 0-4: {sorted(runs)} {sorted(trains)}")
    files = [d / "predictions.csv.gz" for d in runs.values()] + [d / "metrics.json" for d in trains.values()]
    run = start_run(version="v3", exp="v3_sanity", seed=SEED,
                    args={"jaccard": JACCARD, "ngram": 5, "B": B, "seed": SEED,
                          "score_runs": {k: str(d.relative_to(ROOT)) for k, d in runs.items()},
                          "train_runs": {k: str(d.relative_to(ROOT)) for k, d in trains.items()}},
                    data_version="v3", data_files=files, allow_dirty=a.allow_dirty)
    out = {"attack_strings": {}, "val": {}, "style_probe": {}}
    test_frames, nodup_frames, strings = [], [], []
    probes = {False: [], True: []}
    groups = {g: [] for g in GROUPS if g != "all"}
    for k in SEEDS:
        main_, vid = load(k)
        inj = main_[main_["row_type"] == "injected"]
        att = {s: set(inj.loc[inj["split"] == s, "span_text"]) for s in ("train", "val", "test")}
        att["val_id"] = set(vid["span_text"])
        cats = {s: set(inj.loc[inj["split"] == s, "attack_category"]) for s in ("train", "val", "test")}
        jt = max_jaccard(att["test"], att["train"])
        jv = max_jaccard(att["val"], att["train"])
        near_test = {s for s, j in jt.items() if j >= JACCARD}
        out["attack_strings"][str(k)] = {
            "unique": {s: len(v) for s, v in att.items()},
            "test_exact_in_train": len(att["test"] & att["train"]),
            "test_near_dup_of_train": len(near_test),
            "test_max_jaccard_to_train": {"max": max(jt.values()), "median": float(np.median(list(jt.values())))},
            "test_rows_removed": int(((main_["split"] == "test") & main_["span_text"].isin(near_test)
                                      & (main_["row_type"] == "injected")).sum())}
        out["val"][str(k)] = {"val_exact_in_train": len(att["val"] & att["train"]),
                              "val_near_dup_of_train": int(sum(j >= JACCARD for j in jv.values())),
                              "val_max_jaccard_to_train": max(jv.values()),
                              "val_categories_shared_with_train": sorted(cats["val"] & cats["train"]),
                              "test_categories_shared_with_train": sorted(cats["test"] & cats["train"]),
                              "deberta_val_f1_macro": json.loads((trains[k] / "metrics.json").read_text())["val_f1_macro"]}
        p = pd.read_csv(runs[k] / "predictions.csv.gz", keep_default_na=False)
        p = p[p["eval_set"] == "v3_test"].merge(main_[["id", "span_text", "row_type"]], on="id", how="left")
        p = p.assign(run=k, label=p["label"].astype(int))
        test_frames.append(p)
        nodup_frames.append(p[~((p["row_type"] == "injected") & p["span_text"].isin(near_test))])
        strings.append(p[p["label"] == 1])
        for cb in (False, True):
            probes[cb].append(style_probe(main_, cb).assign(run=k))
        for g in groups:
            groups[g].append(style_probe(main_, False, g).assign(run=k))
    test, nodup = pd.concat(test_frames, ignore_index=True), pd.concat(nodup_frames, ignore_index=True)
    out["deberta_test_f1"] = {
        "all": {"task_macro": boot(test, run_mean(task_macro(macro_f1))), "pooled": boot(test, run_mean(macro_f1))},
        "near_dups_removed": {"task_macro": boot(nodup, run_mean(task_macro(macro_f1))),
                              "pooled": boot(nodup, run_mean(macro_f1)), "n_rows": int(len(nodup)),
                              "n_rows_all": int(len(test))}}
    # 2. per-attack-string recall (rows of every split seed), CI over attack strings
    inj = pd.concat(strings, ignore_index=True)
    per = inj.groupby("span_text").agg(n=("pred", "size"), recall=("pred", "mean"),
                                       category=("attack_category", "first"), task=("task", "first")).reset_index()
    rec = run_mean(lambda g: float(g["pred"].mean()))
    idx = _resample_indices(inj.reset_index(drop=True), B, SEED, "span_text", None)
    vals = np.array([rec(inj.iloc[i]) for i in idx])
    worst = per.sort_values(["recall", "n"], ascending=[True, False]).head(15)
    out["per_attack_recall"] = {
        "n_strings": int(len(per)), "rows_per_string": {"min": int(per["n"].min()), "median": float(per["n"].median()),
                                                        "max": int(per["n"].max())},
        "recall_run_mean": rec(inj), "ci_by_attack_string": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "string_recall": {"min": float(per["recall"].min()), "p10": float(per["recall"].quantile(0.1)),
                          "median": float(per["recall"].median()), "n_below_0.5": int((per["recall"] < 0.5).sum()),
                          "n_equal_1": int((per["recall"] == 1).sum())},
        "by_category": {c: {"n_strings": int(len(g)), "row_recall": float(inj.loc[inj["attack_category"] == c, "pred"].mean())}
                        for c, g in per.groupby("category")},
        "worst": [{"text": r.span_text[:160], "category": r.category, "task": r.task, "n": int(r.n), "recall": float(r.recall)}
                  for r in worst.itertuples()]}
    # 3. style probe
    for cb, frames in probes.items():
        d = pd.concat(frames, ignore_index=True)
        out["style_probe"]["clean_a_and_b" if cb else "clean_a_only"] = {
            "f1_macro_pooled": boot(d, run_mean(macro_f1)), "auroc_pooled": boot(d, run_mean(auroc)),
            "f1_macro_task_macro": boot(d, run_mean(task_macro(macro_f1))),
            "per_task_f1": {t: boot(d[d["task"] == t].reset_index(drop=True), run_mean(macro_f1), None)
                            for t in ("code", "email", "table")}}
    out["style_probe"]["clean_a_only_by_feature_group"] = {
        g: {"f1_macro_pooled": boot(d, run_mean(macro_f1)), "auroc_pooled": boot(d, run_mean(auroc))}
        for g, d in ((g, pd.concat(f, ignore_index=True)) for g, f in groups.items())}
    out["style_probe"]["clean_a_only_top_coefficients_by_seed"] = TOP_COEFS
    run.save_metrics(out, headline=("style_probe_clean_a_only_f1", out["style_probe"]["clean_a_only"]["f1_macro_pooled"]["point"]))
    run.finish()


if __name__ == "__main__":
    main()
