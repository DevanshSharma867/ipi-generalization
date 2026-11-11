"""
make_tables.py
==============
Generate LaTeX tables (booktabs), paper/numbers.tex (\\newcommand macros) and the paper figures from results
files only.

V2 Tables III-VI of the Phase 2 report are recomputed from results/v2_std/ with src/metrics.py and checked
against the frozen results/tfidf_lr_results.json, results/leave_one_task_out.json and results/ablations.json
(any difference above 1e-4 fails). Other tables read the latest run of each experiment in results/.

Provenance gate: every macro records the source it was computed from, and must be owned by exactly one entry
of docs/findings.src.md (its `macros:` globs) whose `source:` lists that source. Run-directory sources are
checked against their config.json (commit, data version) and against the git_dirty flag of every run they
consumed (the run's own args plus the entry's `upstream:`). Macros of "full" and "recomputed-match" entries go
to paper/numbers.tex; all others go to paper/legacy_numbers.tex, which the paper must not \\input.

    python scripts/make_tables.py            # writes paper/tables/*.tex, paper/numbers.tex, paper/legacy_numbers.tex
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from make_findings import LEGACY, NUMBERS, NUMBERS_OK, REQUIRED, STATUSES, owns, parse_entries
from metrics import df_metrics, doc_cluster_bootstrap, point_metrics, recall_at_fpr
from sklearn.metrics import roc_curve

STD = ROOT / "results" / "v2_std"
RES = ROOT / "results"
PAPER = ROOT / "paper"
TOL = 1e-4
CI_KEYS = ["accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc",
           "recall_at_5pct_fpr", "recall_at_10pct_fpr"]
TASKS = [("email", "EmailQA"), ("table", "TableQA"), ("code", "CodeQA")]

macros = {}
macro_src = {}  # macro -> the results file or run directory it was computed from
SRC = None
mismatches = []
n_checked = 0
RUN_RE = re.compile(r"results/[^/\s]+/[^/\s]+/\d{8}-\d{6}_seed\d+")


# ─── helpers ──────────────────────────────────────────────────────────────────

_DIGITS = "Zero One Two Three Four Five Six Seven Eight Nine".split()


def macro_name(*parts) -> str:
    words = re.split(r"[^A-Za-z0-9]+", "_".join(parts))
    name = "".join(w[:1].upper() + w[1:] for w in words if w)
    return "".join(_DIGITS[int(c)] if c.isdigit() else c for c in name)


def source(path) -> None:
    """Set the source recorded for the macros that follow (repo-relative file or run directory)."""
    global SRC
    SRC = str(Path(path).relative_to(ROOT)) if Path(path).is_absolute() else str(path)


def put(name, value) -> str:
    if name in macros and macros[name] != value:
        raise ValueError(f"macro {name} defined twice with different values")
    macros[name], macro_src[name] = value, SRC
    return value


def num(value, *parts, digits=4) -> str:
    return put(macro_name(*parts), f"{value:.{digits}f}")


def ci(lo_hi, *parts) -> str:
    lo, hi = (f"{v:.2f}" for v in lo_hi)
    put(macro_name(*parts, "lo"), lo)
    put(macro_name(*parts, "hi"), hi)
    return f"[{lo}, {hi}]"


def check(label, ours, ref):
    global n_checked
    n_checked += 1
    if ref is None or ours is None or abs(float(ours) - float(ref)) > TOL:
        mismatches.append(f"{label}: computed {ours} vs frozen {ref}")


def ci_metrics(df):
    m = df_metrics(df)
    return {k: m[k] for k in CI_KEYS}


def evaluate(df):
    """Point metrics plus doc-cluster bootstrap CIs (B=1000, seed 42)."""
    return df_metrics(df), doc_cluster_bootstrap(df, ci_metrics, B=1000, seed=42)


def load(eval_set, model):
    return pd.read_csv(STD / eval_set / f"{model}.csv", keep_default_na=False)


def write(name, body):
    path = PAPER / "tables" / f"{name}.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    print(f"wrote {path.relative_to(ROOT)}")


# ─── tables ───────────────────────────────────────────────────────────────────

def table_main(tf):
    rows = []
    for model, key, label in [("tfidf_c1", "default_evaluation", r"TF-IDF + LR (Default: $C=1.0, \tau=0.50$)"),
                              ("tfidf_c100_tuned", "tuned_evaluation", r"TF-IDF + LR (Tuned: $C=100.0, \tau=0.71$)")]:
        source(STD / "v2_test" / f"{model}.csv")
        m, b = evaluate(load("v2_test", model))
        ref, ref_ci = tf[key]["metrics"], tf[key]["doc_cluster_bootstrap_95ci"]
        for k in CI_KEYS:
            check(f"Tab III {model} {k}", m[k], ref[k])
            check(f"Tab III {model} {k} CI lo", b[k]["ci"][0], ref_ci[k][0])
            check(f"Tab III {model} {k} CI hi", b[k]["ci"][1], ref_ci[k][1])
        for t in ("5pct_fpr", "10pct_fpr"):
            check(f"Tab III {model} threshold@{t}", m[f"threshold_at_{t}"], ref[f"operating_threshold_at_{t}"])
        p = ("v2", model)
        cells = [f"{num(m[k], *p, k)} {ci(b[k]['ci'], *p, k, 'ci')}" for k in CI_KEYS[:6]]
        cells += [num(m["recall_at_5pct_fpr"], *p, "rec5"), num(m["recall_at_10pct_fpr"], *p, "rec10")]
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    write("v2_table3_main", "\n".join([
        r"% Generated by scripts/make_tables.py from results/v2_std/v2_test/ -- do not edit by hand",
        r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
        r"\textbf{Model / Configuration} & \textbf{Acc [95\% CI]} & \textbf{Macro-F1 [95\% CI]} & \textbf{Inj-F1 [95\% CI]} & "
        r"\textbf{Prec [95\% CI]} & \textbf{Rec [95\% CI]} & \textbf{ROC-AUC [95\% CI]} & \textbf{Rec@5\%FPR} & \textbf{Rec@10\%FPR} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def table_ablations(abl):
    source(STD / "v2_test" / "tfidf_c1.csv")
    test = load("v2_test", "tfidf_c1")
    # Majority class: sklearn DummyClassifier(most_frequent) breaks the 668/668 train tie towards class 0
    maj = point_metrics(test["label"], np.zeros(len(test)), pred=np.zeros(len(test), dtype=int))
    check("Tab IV majority accuracy", maj["accuracy"], abl["majority_class"]["accuracy"])
    check("Tab IV majority f1_macro", maj["f1_macro"], abl["majority_class"]["macro_f1"])
    fused = df_metrics(test)
    rows = [("Majority Class Baseline", "majority", maj)]
    # No per-row predictions were saved for these ablations; values are read from results/ablations.json
    for key, label in [("length_only", "Text Length Only (Chars/Words)"), ("intent_only", "Intent Only (User Query)"),
                       ("context_only", "Context Only (Document Body)")]:
        rows.append((label, key, abl[key]))
    body = []
    for label, key, m in rows:
        source(STD / "v2_test" / "tfidf_c1.csv" if key == "majority" else RES / "ablations.json")
        cells = [num(m[k], "v2", "abl", key, k) for k in ("accuracy", "f1_macro", "precision", "recall")]
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
    source(STD / "v2_test" / "tfidf_c1.csv")
    fcells = [num(fused[k], "v2", "abl", "fused", k) for k in ("accuracy", "f1_macro", "precision", "recall")]
    write("v2_table4_ablations", "\n".join([
        r"% Generated by scripts/make_tables.py. Majority and Fused rows recomputed from results/v2_std/v2_test/;",
        r"% Length/Intent/Context rows read from results/ablations.json (no per-row predictions were saved).",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"\textbf{Ablation Variant} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Precision} & \textbf{Recall} \\",
        r"\midrule", *body, r"\midrule",
        r"\textbf{Fused Intent + Context (Full)} & " + " & ".join(fcells) + r" \\",
        r"\bottomrule", r"\end{tabular}", ""]))


def table_per_task(tf):
    source(STD / "v2_test" / "tfidf_c1.csv")
    test = load("v2_test", "tfidf_c1")
    rows = []
    for task, label in TASKS:
        sub = test[test["task"] == task].reset_index(drop=True)
        m, b = evaluate(sub)
        ref = tf["per_task_breakdown"][task]
        for k in ("accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc"):
            check(f"Tab V {task} {k}", m[k], ref["metrics"][k])
        for k in CI_KEYS:
            check(f"Tab V {task} {k} CI lo", b[k]["ci"][0], ref["cluster_bootstrap_95ci"][k][0])
            check(f"Tab V {task} {k} CI hi", b[k]["ci"][1], ref["cluster_bootstrap_95ci"][k][1])
        p = ("v2", "task", task)
        n_docs = sub["doc_id"].nunique()
        put(macro_name(*p, "n"), str(len(sub)))
        put(macro_name(*p, "ndocs"), str(n_docs))
        rows.append(f"{label} & {len(sub)} ({n_docs}) & {num(m['accuracy'], *p, 'accuracy')} & "
                    f"{num(m['f1_macro'], *p, 'f1_macro')} & {ci(b['f1_macro']['ci'], *p, 'f1_macro', 'ci')} & "
                    f"{num(m['precision'], *p, 'precision')} & {num(m['recall'], *p, 'recall')} \\\\")
    write("v2_table5_per_task", "\n".join([
        r"% Generated by scripts/make_tables.py from results/v2_std/v2_test/tfidf_c1.csv",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"\textbf{Task Type} & \textbf{N (Docs)} & \textbf{Accuracy} & \textbf{Macro-F1} & \textbf{Macro-F1 95\% CI} & "
        r"\textbf{Prec} & \textbf{Rec} \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def table_loto(loto):
    train = pd.read_csv(ROOT / "data/processed/train.csv")
    names = dict(TASKS)
    rows = []
    for task, label in TASKS:
        source(STD / f"v2_loto_test_{task}" / "tfidf_c100_loto.csv")
        sub = load(f"v2_loto_test_{task}", "tfidf_c100_loto")
        m, b = evaluate(sub)
        ref = loto[f"held_out_{task}"]
        n_tr = int((train["task_type"] != task).sum())
        check(f"Tab VI {task} n_train", n_tr, ref["n_train_samples"])
        check(f"Tab VI {task} n_test", len(sub), ref["n_test_samples"])
        for k in ("accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc"):
            check(f"Tab VI {task} {k}", m[k], ref["metrics"][k])
        for k in CI_KEYS:
            check(f"Tab VI {task} {k} CI lo", b[k]["ci"][0], ref["doc_cluster_bootstrap_95ci"][k][0])
            check(f"Tab VI {task} {k} CI hi", b[k]["ci"][1], ref["doc_cluster_bootstrap_95ci"][k][1])
        p = ("v2", "loto", task)
        put(macro_name(*p, "ntrain"), str(n_tr))
        put(macro_name(*p, "ntest"), str(len(sub)))
        trained_on = " + ".join(names[t].replace("QA", "").capitalize() for t, _ in TASKS if t != task)
        rows.append(f"{trained_on} & \\textbf{{{label}}} & {n_tr} / {len(sub)} & {num(m['accuracy'], *p, 'accuracy')} & "
                    f"{num(m['f1_macro'], *p, 'f1_macro')} & {ci(b['f1_macro']['ci'], *p, 'f1_macro', 'ci')} & "
                    f"{num(m['roc_auc'], *p, 'roc_auc')} \\\\")
    write("v2_table6_loto", "\n".join([
        r"% Generated by scripts/make_tables.py from results/v2_std/v2_loto_test_*/tfidf_c100_loto.csv",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"\textbf{Train Tasks} & \textbf{Held-Out Task} & \textbf{N (tr/te)} & \textbf{Acc} & \textbf{Macro-F1} & "
        r"\textbf{Macro-F1 95\% CI} & \textbf{AUC} \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


# ─── V2 position probes (results/v2_probes/, latest run of each exp) ──────────

PROBE_COLS = [("p1_neutral_insertion", p) for p in ("start", "middle", "end")] + \
             [("p2_benign_moved", p) for p in ("start", "middle", "end")] + \
             [("p3_attack_moved", p) for p in ("start", "middle", "end")]


def latest_metrics(exp, version="v2_probes"):
    runs = sorted((RES / version / exp).glob("*/metrics.json"))
    if not runs:
        return None, None
    source(runs[-1].parent)
    return json.load(open(runs[-1])), runs[-1].parent


def cell(c, *parts, digits=2):
    if c is None:
        return "--"
    return f"{num(c['point'], *parts, digits=digits)} {{\\scriptsize{ci(c['ci'], *parts, 'ci')}}}"


def tex(s):
    return s.replace("_", r"\_")


def table_artifact_baselines():
    m, src = latest_metrics("artifact_baselines")
    if m is None:
        return
    rows = []
    for name in ("position_oracle", "first_para_rule", "prefix30_tfidf_lr", "suffix30_tfidf_lr", "span_form_lr"):
        r, b = m[name], m[name]["bootstrap"]
        p = ("v2", "artifact", name)
        rows.append(f"{tex(name)} & {tex(r['eval_set'])} & {r['n']} & {cell(b['f1_macro'], *p, 'f1_macro', digits=4)} & "
                    f"{cell(b['roc_auc'], *p, 'roc_auc', digits=4)} \\\\")
    write("v2_artifact_baselines", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"\textbf{Probe} & \textbf{Eval rows} & \textbf{N} & \textbf{Macro-F1 [95\% CI]} & \textbf{ROC-AUC [95\% CI]} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def table_v3_gates(stage="prefreeze"):
    """Content-free gates (v3_design §8, R2) and integrity checks on the 5 split realizations: "prefreeze" (draft
    benign pool) or "frozen" (the v3-frozen build of data/v3)."""
    m, src = latest_metrics(f"artifact_baselines_{stage}", version="v3")
    if m is None:
        return
    g = m["gate_summary"]
    rows = []
    for name, e in g["probes"].items():
        p = ("v3", stage, name)
        pooled = list(e["pooled_by_seed"].values())
        lo, hi = num(min(pooled), *p, "pooled_min"), num(max(pooled), *p, "pooled_max")
        tasks = [num(e["per_task_mean"][t], *p, "task_mean", t) for t, _ in TASKS]
        status = put(macro_name(*p, "status"), ("PASS" if e["pass"] else "FAIL") if e["gate"] else "diag")
        rows.append(f"{tex(name)} & {lo}--{hi} & {' & '.join(tasks)} & {status} \\\\")
    put(macro_name("v3", stage, "gate", "threshold"), f"{g['threshold']:.2f}")
    put(macro_name("v3", stage, "gates", "status"), "PASS" if g["pass"] else "FAIL")
    write(f"v3_{stage}_gates", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"\textbf{Probe} & \textbf{Pooled macro-F1 (seeds 0--4)} & \textbf{EmailQA} & \textbf{TableQA} & "
        r"\textbf{CodeQA} & \textbf{Gate} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    m, src = latest_metrics(f"integrity_{stage}", version="v3")
    if m is None:
        return
    put(macro_name("v3", stage, "integrity", "status"), "PASS" if m["pass"] else "FAIL")
    for s, r in m["seeds"].items():
        p = ("v3", stage, "integrity", "seed", s)
        put(macro_name(*p, "status"), "PASS" if r["pass"] else "FAIL")
        for k, n in r["rows"].items():
            put(macro_name(*p, "rows", k), str(n))
        for k, n in r["main_by_task_split"].items():
            put(macro_name(*p, "main", k), str(n))


def macros_v3_sanity():
    """Read-only sanity checks of the V3 DeBERTa result (scripts/v3_sanity.py): attack-string overlap, test F1
    without near-duplicates, per-attack-string recall, span style probe, val overlap."""
    m, src = latest_metrics("v3_sanity", version="v3")
    if m is None:
        return
    p = ("v3", "sanity")
    a = m["attack_strings"].values()
    for s in ("train", "val", "test", "val_id"):
        u = [v["unique"][s] for v in a]
        put(macro_name(*p, "unique", s, "min"), str(min(u)))
        put(macro_name(*p, "unique", s, "max"), str(max(u)))
    put(macro_name(*p, "test", "exact", "in", "train"), str(max(v["test_exact_in_train"] for v in a)))
    put(macro_name(*p, "test", "near", "dups"), str(max(v["test_near_dup_of_train"] for v in a)))
    num(max(v["test_max_jaccard_to_train"]["max"] for v in a), *p, "test", "max", "jaccard", digits=2)
    vv = m["val"].values()
    put(macro_name(*p, "val", "exact", "in", "train"), str(max(v["val_exact_in_train"] for v in vv)))
    put(macro_name(*p, "val", "near", "dups"), str(max(v["val_near_dup_of_train"] for v in vv)))
    num(max(v["val_max_jaccard_to_train"] for v in vv), *p, "val", "max", "jaccard", digits=2)
    put(macro_name(*p, "val", "shared", "categories"), str(max(len(v["val_categories_shared_with_train"]) for v in vv)))
    f1s = [v["deberta_val_f1_macro"] for v in vv]
    num(min(f1s), *p, "val", "f", "one", "min")
    num(max(f1s), *p, "val", "f", "one", "max")
    f = m["deberta_test_f1"]
    for which in ("all", "near_dups_removed"):
        for agg in ("task_macro", "pooled"):
            cell(f[which][agg], *p, "test", "f", "one", which, agg, digits=4)
    put(macro_name(*p, "test", "rows", "kept"), str(f["near_dups_removed"]["n_rows"]))
    put(macro_name(*p, "test", "rows", "all"), str(f["near_dups_removed"]["n_rows_all"]))
    r = m["per_attack_recall"]
    put(macro_name(*p, "attack", "strings"), str(r["n_strings"]))
    cell({"point": r["recall_run_mean"], "ci": r["ci_by_attack_string"]}, *p, "attack", "recall", digits=4)
    num(r["string_recall"]["min"], *p, "attack", "recall", "min")
    put(macro_name(*p, "attack", "below", "half"), str(r["string_recall"]["n_below_0.5"]))
    put(macro_name(*p, "attack", "perfect"), str(r["string_recall"]["n_equal_1"]))
    sp = m["style_probe"]
    for v in ("clean_a_only", "clean_a_and_b"):
        cell(sp[v]["f1_macro_pooled"], *p, "style", v, "f", "one", digits=4)
        cell(sp[v]["auroc_pooled"], *p, "style", v, "auroc", digits=4)
    for g, v in sp["clean_a_only_by_feature_group"].items():
        cell(v["f1_macro_pooled"], *p, "style", "group", g, "f", "one", digits=4)
        cell(v["auroc_pooled"], *p, "style", "group", g, "auroc", digits=4)


TRANSFER_SETS = {"agentdojo_benign_outputs.jsonl": "ad_clean", "agentdojo_pairs.jsonl": "ad_matched",
                 "agentdojo_pairs_shift.jsonl": "ad_shift", "taubench_eval.jsonl": "tb_matched",
                 "bipia_pairs.jsonl": "bipia", "corpus.jsonl": "general"}
COUNT_KEYS = {"None": "rows", "0": "benign", "1": "injected", "1:test": "injected_test"}


def macros_transfer_gate1():
    """Gate 1 (paper_plan A17): rebuilt Liu (2026) set sizes, which match his paper and released score files."""
    m, _ = latest_metrics("set_build_gate1", version="transfer")
    if m is None:
        return
    put(macro_name("transfer", "gate1", "status"), "PASS" if m["pass"] else "FAIL")
    for f, s in m["sets"].items():
        p = ("transfer", "gate1", TRANSFER_SETS[f])
        put(macro_name(*p, "files"), str(len(s["released_files"])))
        for k, c in s["paper_counts"].items():
            put(macro_name(*p, COUNT_KEYS[k]), str(c["rebuilt"]))


def sci(value, *parts) -> str:
    return put(macro_name(*parts), f"{value:.1e}")


def macros_transfer_gate2():
    """Gate 2 (paper_plan A19, A20; docs/deviations.md 2026-10-30): Liu (2026) reproduction in his environment with
    the recovered AgentDojo pairing, the judge's precision and run-to-run noise, and the library-version effect."""
    m, _ = latest_metrics("gate2_followup_pairing", version="transfer")
    if m is None:
        return
    p = ("transfer", "gate2")
    a19 = m["a19_with_recovered_admatched"]
    put(macro_name(*p, "status"), "PASS" if a19["pass"] else "FAIL")
    for k, c in a19["checks"].items():
        det, metric = k.split("/")
        num(c["ours"], *p, det, metric, "ours")
        num(c["liu"], *p, det, metric, "liu")
    put(macro_name(*p, "recovery", "status"), "PASS" if m["recovery_pass"] else "FAIL")
    for s, r in m["recovery"].items():
        put(macro_name(*p, s, "groups"), str(r["groups"]))
        put(macro_name(*p, s, "matched"), str(r["matched"]))
        put(macro_name(*p, s, "timedep"), str(r["time_dependent_rows"]))
    for k, v in m["validation_rows_vs_liu"].items():
        det, s = k.split("/")
        sci(v["max_abs"], *p, s, det, "maxdiff")
        put(macro_name(*p, s, det, "overbound"), str(v["n_gt_1e3"]))
    g = m["judge_bf16_vs_nf4_admatched_recovered"]
    num(g["gap_bf16_minus_nf4"]["admatched_auroc"], *p, "judge", "recovered", "gap")
    num(g["bf16"]["admatched_auroc"], *p, "judge", "recovered", "bf16")
    m, _ = latest_metrics("gate2_followup_residual", version="transfer")
    if m is not None:
        put(macro_name(*p, "residual", "status"), "PASS" if m["explained"] else "FAIL")
        for s in ("admatched", "adshift"):
            r = m[s]["piguard"]
            put(macro_name(*p, s, "multipart"), str(r["multi_participant_rows"]))
            sci(r["max_abs_other"], *p, s, "piguard", "maxother")
    m, _ = latest_metrics("gate2_followup_judge", version="transfer")
    if m is not None:
        j = m["judge_bipia"]
        for run in ("judge_repeat", "judge_6gib"):
            sci(j[run]["rows_vs_run2"]["max_abs"], *p, run, "vsrun")
            put(macro_name(*p, run, "flips"), str(j[run]["rows_vs_run2"]["flips_at_05"]))
        r = j["judge_repeat"]["rows_vs_liu"]
        num(r["max_abs"], *p, "judge", "vsliu", "max", digits=2)
        put(macro_name(*p, "judge", "vsliu", "overbound"), str(r["n_gt_1e3"]))
        put(macro_name(*p, "judge", "vsliu", "flips"), str(r["flips_at_05"]))
        put(macro_name(*p, "judge", "bipia", "rows"), str(r["n"]))
        g = m["judge_bf16_vs_nf4_admatched_seeded"]
        num(g["gap_bf16_minus_nf4"]["admatched_auroc"], *p, "judge", "seeded", "gap")
    m, _ = latest_metrics("gate2_followup_batch", version="transfer")
    if m is None:
        return
    p = ("transfer", "env")
    for det, d in m["batch_env_bipia500"].items():
        e = d["bs1_ours_vs_liu_env"]
        num(e["max_abs"], *p, det, "max", digits=2)
        put(macro_name(*p, det, "overbound"), str(e["n_gt_1e3"]))
        put(macro_name(*p, det, "flips"), str(e["flips_at_05"]))
        sci(max(d["liu_bs1_vs_bs16"]["max_abs"], d["ours_bs1_vs_bs16"]["max_abs"]), *p, det, "batch")
        sci(d["liu_bs1_vs_released"]["max_abs"], *p, det, "liuenv")
    put(macro_name(*p, "rows"), str(e["n"]))
    sci(max(v["max_abs"] for v in m["deberta_v2_bs1_vs_bs16"].values()), *p, "deberta", "batch")
    put(macro_name(*p, "deberta", "status"), "PASS" if m["batch_independent_ours_deberta"] else "FAIL")
    m, _ = latest_metrics("gate2_followup_envfix", version="transfer")
    if m is None:
        return
    put(macro_name(*p, "fix", "status"), "PASS" if m["explained"] else "FAIL")
    for det in ("piguard", "promptguard2-86m", "horizon-base"):
        sci(m[det]["fixed"]["max_abs"], *p, det, "fixed")
        put(macro_name(*p, det, "fixed", "overbound"), str(m[det]["fixed"]["n_gt_1e3"]))


TRANSFER_EVAL = (("admatched", "AD-Matched"), ("adshift", "AD-Shift"), ("tbeval", "TB-Matched"))
TRUNC_SETS = (*TRANSFER_EVAL, ("adclean", "AD-Clean"), ("tbclean", "TB-Clean"), ("calpool", "Cal.\\ pool"))
TRANSFER_NEG = {"admatched": "adclean", "adshift": "adclean", "tbeval": "tbclean"}  # clean rows each eval set uses
HIGH_TRUNCATION = 0.25  # table mark: more than this fraction of a set's rows truncated (F021 note, F023)


def high_truncation(v, s):
    """True when the detector truncates (not windows) more than HIGH_TRUNCATION of eval set s or its clean rows."""
    o = v.get("over_max_len")
    return bool(o and o["handling"] == "truncated"
                and max(o["fraction"][s], o["fraction"][TRANSFER_NEG[s]]) > HIGH_TRUNCATION)


def table_transfer_detectors():
    """Published detectors on the agent transfer sets (paper_plan §3, A3, A16): AUROC, calibrated TPR@1%FPR with the
    realized FPR on the evaluated clean rows (thresholds from the calibration pool), user-task cluster CIs."""
    m, src = latest_metrics("transfer_metrics", version="transfer")
    if m is None:
        return
    rows = []
    for key, v in m.items():
        det, tag = key.split("@")
        p0 = ("transfer", "det", det, tag)
        num(v["calibration_pool"]["flag_at_0.5"], *p0, "calpool", "flag", digits=3)
        cells = []
        for s, _ in TRANSFER_EVAL:
            e, p = v[s], (*p0, s)
            mark = r"$^\dagger$" if high_truncation(v, s) else ""
            cells.append(cell(e["auroc"], *p, "auroc", digits=3) + mark)
            num(e["fpr_at_0.5"]["point"], *p, "fpr", "default", digits=3)
            for k in ("1pct_fpr", "5pct_fpr"):
                c = e[f"calibrated@{k}"]
                c_tpr = cell(c["tpr"], *p, "cal", k, "tpr", digits=3)
                c_fpr = num(c["realized_fpr"]["point"], *p, "cal", k, "fpr", digits=3)
                ci(c["realized_fpr"]["ci"], *p, "cal", k, "fpr", "ci")
                o = e[f"oracle@{k}"]
                put(macro_name(*p, "oracle", k, "tpr"), f"{o['tpr']['point']:.3f}" if o["status"] == "ok" else "n/a")
                if k == "1pct_fpr":
                    cells.append(f"{c_tpr} & {c_fpr}")
                for proto in ("task_half", "calibration_pool"):  # secondary protocol, same evaluation half
                    h = e["task_halves"][proto]
                    hk = h[f"calibrated@{k}"]
                    if hk["status"] == "ok":
                        num(hk["tpr"]["point"], *p, "halves", proto, k, "tpr", digits=3)
                        num(hk["realized_fpr"]["point"], *p, "halves", proto, k, "fpr", digits=3)
                    else:
                        put(macro_name(*p, "halves", proto, k, "tpr"), "n/a")
                        put(macro_name(*p, "halves", proto, k, "fpr"), "n/a")
        for s in ("adclean", "tbclean"):
            for thr, b in v[f"{s}_task_block_rate"].items():
                if isinstance(b, dict):
                    cell(b, *p0, s, "block", thr, digits=3)
        rows.append(f"{tex(det)} ({tex(tag)}) & " + " & ".join(cells) + r" \\")
    write("transfer_detectors", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"% AUROC [95% CI]; calibrated TPR@1%FPR [CI] and realized FPR on the clean rows (AD: AD-Clean; TB: TB-Clean);",
        r"% thresholds from the calibration pool (A3); CIs resample user tasks (B=1000).",
        r"\begin{tabular}{l" + "rrr" * len(TRANSFER_EVAL) + "}", r"\toprule",
        r"\textbf{Detector} & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{\textbf{{{n}}}}}" for _, n in TRANSFER_EVAL)
        + r" \\",
        " & " + " & ".join(["AUROC & TPR@1\\% & FPR"] * len(TRANSFER_EVAL)) + r" \\",
        r"\midrule", *rows, r"\bottomrule",
        rf"\multicolumn{{{1 + 3 * len(TRANSFER_EVAL)}}}{{l}}{{\scriptsize $^\dagger$ More than "
        rf"{HIGH_TRUNCATION:.0%}".replace("%", r"\%") + r" of this set's injected or clean rows exceed the detector's "
        r"max length and are truncated (scored as shipped; Table~\ref{tab:transfer-truncation}).} \\",
        r"\end{tabular}", ""]))
    table_transfer_truncation(m, src)
    # calibration transfer: how often the calibration pool's threshold misses its target on the agent sets' clean rows
    for s, _ in TRANSFER_EVAL:
        for k, t in (("1pct_fpr", 0.01), ("5pct_fpr", 0.05)):
            f = [v[s][f"calibrated@{k}"]["realized_fpr"] for v in m.values()]
            p = ("transfer", "caltransfer", s, k)
            put(macro_name(*p, "n"), str(len(f)))
            put(macro_name(*p, "above"), str(sum(x["ci"][0] > t for x in f)))  # CI entirely above the target
            put(macro_name(*p, "below"), str(sum(x["ci"][1] < t for x in f)))
            num(float(np.median([x["point"] for x in f])), *p, "median", digits=3)
            for proto in ("task_half", "calibration_pool"):  # both on the evaluation half
                h = [v[s]["task_halves"][proto][f"calibrated@{k}"] for v in m.values()]
                h = [x["realized_fpr"]["point"] for x in h if x["status"] == "ok"]
                put(macro_name(*p, "halves", proto, "n"), str(len(h)))
                if h:
                    num(float(np.median(h)), *p, "halves", proto, "median", digits=3)


def table_transfer_truncation(m, src):
    """Fraction of each set's rows over each detector's max length (F023), with how such rows are handled."""
    rows, high = [], {s: [] for s, _ in TRANSFER_EVAL}
    for key, v in m.items():
        det, tag = key.split("@")
        o = v["over_max_len"]
        cells = ["--" if o["fraction"][s] is None
                 else num(o["fraction"][s], "transfer", "trunc", det, tag, s, digits=3) for s, _ in TRUNC_SETS]
        for s, _ in TRANSFER_EVAL:
            if high_truncation(v, s):
                high[s].append(det)
        rows.append(f"{tex(det)} ({tex(tag)}) & {o['max_len'] or '--'} & {o['handling'] or 'whole input'} & "
                    + " & ".join(cells) + r" \\")
    for s, _ in TRANSFER_EVAL:
        put(macro_name("transfer", "trunc", s, "high", "n"), str(len(high[s])))
    put(macro_name("transfer", "trunc", "threshold"), f"{HIGH_TRUNCATION:.2f}")
    write("transfer_truncation", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"% Fraction of rows over the detector's max length (tokens). truncated: tokens past the window are dropped;",
        r"% windowed: every window scored, maximum taken; judges read the whole input. Scoring is Liu's, as shipped.",
        r"\begin{tabular}{lrl" + "r" * len(TRUNC_SETS) + "}", r"\toprule",
        r"\textbf{Detector} & \textbf{Max len} & \textbf{Long inputs} & "
        + " & ".join(rf"\textbf{{{n}}}" for _, n in TRUNC_SETS) + r" \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def macros_agent(version):
    """Agent-style data (paper_plan A3): calibration pool size and sha256, overlap audit, content-free gates."""
    v = version.split("_")[1]  # agent_v1 -> v1
    m, src = latest_metrics("agent_overlap_audit", version=version)
    if m is not None:
        cfg = json.load(open(src / "config.json"))
        cal = [x for k, x in cfg["data_files_sha256"].items() if k.endswith("calibration_pool.csv")]
        put(macro_name("agent", v, "cal", "sha"), cal[0])
        put(macro_name("agent", v, "cal", "rows"), str(m["units"]["calibration_pool"]["agentdojo"]["n"]))
        put(macro_name("agent", v, "audit", "status"), "PASS" if m["verdict"]["pass"] else "FAIL")
        for u, cs in m["units"].items():
            for c, r in cs.items():
                put(macro_name("agent", v, "audit", u, c, "eight"), str(r["n_sharing_8gram"]))
                put(macro_name("agent", v, "audit", u, c, "five"), str(r["n_sharing_5gram"]))
                put(macro_name("agent", v, "audit", u, c, "contained"), str(r["n_containments"]))
            put(macro_name("agent", v, "audit", u, "n"), str(r["n"]))
    m, src = latest_metrics("artifact_baselines", version=version)
    if m is not None:
        g = m["gate_summary"]
        for name, e in g["probes"].items():
            p = ("agent", v, name)
            pooled = list(e["pooled_by_seed"].values())
            num(min(pooled), *p, "pooled_min")
            num(max(pooled), *p, "pooled_max")
            worst = max(e["per_task_mean"], key=e["per_task_mean"].get)
            num(e["per_task_mean"][worst], *p, "tool_max")
            put(macro_name(*p, "tool_max_name"), tex(worst))
            put(macro_name(*p, "status"), ("PASS" if e["pass"] else "FAIL") if e["gate"] else "diag")
            # transparency only, never pass/fail (docs/deviations.md 2026-10-31): pooled |AUROC - 0.5|, max over seeds
            num(max(abs(r[name]["bootstrap"]["roc_auc"]["point"] - 0.5) for r in m["seeds"].values()), *p, "auroc_dev_max")
        put(macro_name("agent", v, "gates", "status"), "PASS" if g["pass"] else "FAIL")
    m, src = latest_metrics("form_cv", version=version)
    if m is not None:  # within-train CV (scripts/agent_form_cv.py); named AgentFormCv* so F019's glob does not own it
        for name in m["summary"]:
            vals = [r[name] for r in m["seeds"].values()]
            p = ("agent", "form_cv", v, name)
            num(min(x["f1_macro"] for x in vals), *p, "f1_min")
            num(max(x["f1_macro"] for x in vals), *p, "f1_max")
            num(min(x["roc_auc"] for x in vals), *p, "auroc_min")
            num(max(x["roc_auc"] for x in vals), *p, "auroc_max")
        put(macro_name("agent", "form_cv", v, "flagged"), tex(", ".join(m["flagged"]) or "none"))


def tables_probe_report():
    m, src = latest_metrics("probe_report")
    if m is None:
        return
    models = m["models"]
    rel, con = [], []
    for name, r in models.items():
        cells = r["cells"]
        row = [cell(cells.get(f"{s}/{pos}"), "v2", "probe", name, s.split("_")[0], pos) for s, pos in PROBE_COLS]
        row.append(cell(cells.get("p4_whitespace_only"), "v2", "probe", name, "p4"))
        row.append(cell(cells.get("v2_test_f1_macro"), "v2", "probe", name, "v2test", "f1_macro", digits=4))
        rel.append(f"{tex(name)} & " + " & ".join(row) + r" \\")
        c = r["contrasts"]
        con.append(f"{tex(name)} & " + " & ".join(
            cell({"point": c[k]["diff"], "ci": c[k]["ci"]} if k in c else None, "v2", "probe", name, k, digits=4)
            for k in ("p1_start_minus_middle", "p3_end_minus_start", "p2_start_minus_middle")) + r" \\")
    head = f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json"
    write("v2_probe_reliance", "\n".join([
        head, r"% Cells: rate flagged as injected (P3: recall) [doc-cluster bootstrap 95% CI]",
        r"\begin{tabular}{l" + "r" * 11 + "}", r"\toprule",
        r" & \multicolumn{3}{c}{\textbf{P1 neutral (flag)}} & \multicolumn{3}{c}{\textbf{P2 benign (flag)}} & "
        r"\multicolumn{3}{c}{\textbf{P3 attack (recall)}} & \textbf{P4} & \textbf{V2 test} \\",
        r"\textbf{Model} & start & middle & end & start & middle & end & start & middle & end & flag & macro-F1 \\",
        r"\midrule", *rel, r"\bottomrule", r"\end{tabular}", ""]))
    write("v2_position_contrasts", "\n".join([
        head, r"% Paired within-doc differences [paired doc-cluster bootstrap 95% CI]",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{P1 start $-$ middle} & \textbf{P3 end $-$ start} & \textbf{P2 start $-$ middle} \\",
        r"\midrule", *con, r"\bottomrule", r"\end{tabular}", ""]))


def figure_probe_report():
    """paper/figures/position_flag_rates.{pdf,png}: copied from the latest probe_report run, which draws it."""
    m, src = latest_metrics("probe_report")
    if m is None:
        return
    out = PAPER / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        shutil.copy(src / f"position_flag_rates.{ext}", out / f"position_flag_rates.{ext}")
    print(f"wrote {(out / 'position_flag_rates').relative_to(ROOT)}.pdf/.png (from {src.relative_to(ROOT)})")


def diff_cell(c, *parts):
    return cell(None if c is None else {"point": c["diff"], "ci": c["ci"]}, *parts, digits=4)


def table_seed_shortcut_variance():
    m, src = latest_metrics("probe_report")
    if m is None:
        return
    rows = []
    for name in ("deberta_s42", "deberta_s123", "deberta_s999", "deberta_mean"):
        r = m["models"].get(name)
        if r is None:
            continue
        p = ("v2", "seedvar", name)
        rows.append(f"{tex(name)} & {cell(r['cells'].get('v2_test_f1_macro'), *p, 'f1_macro', digits=4)} & "
                    f"{cell(r['cells'].get('p1_neutral_insertion/start'), *p, 'p1start', digits=4)} & "
                    f"{diff_cell(r['contrasts'].get('p2_start_minus_middle'), *p, 'p2startminusmiddle')} \\\\")
    write("v2_seed_shortcut_variance", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"\textbf{DeBERTa seed} & \textbf{V2 test macro-F1} & \textbf{P1 start flag rate} & \textbf{P2 start $-$ middle} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def table_v2_test_groups():
    m, src = latest_metrics("probe_report")
    if m is None:
        return
    rows = []
    for name, r in m["models"].items():
        c, p = r["cells"], ("v2", "groups", name)
        for key, cell_key in (("recall", "v2_test_recall"), ("benignflag", "v2_test_benign_imperative_flag"),
                              ("rawcleanflag", "v2_test_raw_clean_flag"), ("pfourflag", "p4_whitespace_only")):
            if cell_key in c:
                put(macro_name(*p, key, "n"), str(c[cell_key]["n"]))
        if "p4_minus_raw_clean" in r["contrasts"]:
            put(macro_name(*p, "pfourminusrawclean", "npairs"), str(r["contrasts"]["p4_minus_raw_clean"]["n_pairs"]))
        rows.append(f"{tex(name)} & {cell(c.get('v2_test_recall'), *p, 'recall', digits=4)} & "
                    f"{cell(c.get('v2_test_benign_imperative_flag'), *p, 'benignflag', digits=4)} & "
                    f"{cell(c.get('v2_test_raw_clean_flag'), *p, 'rawcleanflag', digits=4)} & "
                    f"{cell(c.get('p4_whitespace_only'), *p, 'pfourflag', digits=4)} & "
                    f"{diff_cell(r['contrasts'].get('p4_minus_raw_clean'), *p, 'pfourminusrawclean')} \\\\")
    write("v2_test_groups", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"% V2 test (288 rows, 72 docs) by row group [doc-cluster bootstrap 95% CI]; P4 - raw clean is paired by doc",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Recall (injected)} & \textbf{Flag: benign imperative} & \textbf{Flag: raw clean} & "
        r"\textbf{Flag: P4} & \textbf{P4 $-$ raw clean} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def table_summary_json_bug():
    """Recall@5%FPR on pairs v0: the value stored in results/scores/summary.json (roc_curve with
    drop_intermediate=True) next to src/metrics.recall_at_fpr on the converted copy in results/v2_std/."""
    summary = json.load(open(RES / "scores" / "summary.json"))["pairs"]
    rows = []
    for name in ("piguard", "protectai"):
        source(STD / "pairs_v0" / f"{name}.csv")
        d = load("pairs_v0", name)
        ours = recall_at_fpr(d["label"], d["prob"], 0.05)[0]
        # Reproduce the stored value: score_detectors.recall_at_fpr (roc_curve with drop_intermediate=True)
        fpr, tpr, _ = roc_curve(d["label"], d["prob"])
        check(f"pairsbug {name} summary.json recall@5%fpr", float(tpr[fpr <= 0.05].max()), summary[name]["recall@5%fpr"])
        p = ("v2", "pairsbug", name)
        ours_s = num(ours, *p, "metrics")
        source(RES / "scores" / "summary.json")
        rows.append(f"{tex(name)} & {num(summary[name]['recall@5%fpr'], *p, 'summaryjson')} & {ours_s} \\\\")
    write("v2_pairs_recall5_provenance", "\n".join([
        "% Generated by scripts/make_tables.py from results/scores/summary.json and results/v2_std/pairs_v0/",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"\textbf{Model} & \textbf{summary.json} & \textbf{src/metrics.py} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))


def tables_truncation_control():
    m, src = latest_metrics("truncation_control")
    if m is None:
        return
    head = f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json"
    put(macro_name("v2", "trunc", "window_tokens"), str(json.load(open(src / "config.json"))["args"]["window_tokens"]))
    rec, con = [], []
    for name, r in m["models"].items():
        mt, full, fam = r["matched"], r["full"], r["family"]
        p = ("v2", "trunc", name)
        rec.append(f"{tex(name)} & {r['rows_matched']} / {r['rows_full']} & " + " & ".join(
            cell(mt["cells"].get(f"p3_attack_moved/{pos}"), *p, "p3", pos) for pos in ("start", "middle", "end")) + r" \\")
        frac = 0.0 if fam is None else m["p3_truncated_fraction_by_position"][fam]["end"]["all"]
        con.append(f"{tex(name)} & {num(frac, *p, 'p3endtruncated', digits=3)} & "
                   f"{diff_cell(full['contrasts'].get('p3_end_minus_start'), *p, 'p3endminusstart', 'full')} & "
                   f"{diff_cell(mt['contrasts'].get('p3_end_minus_start'), *p, 'p3endminusstart', 'matched')} & "
                   f"{diff_cell(mt['contrasts'].get('p2_start_minus_middle'), *p, 'p2startminusmiddle', 'matched')} & "
                   f"{diff_cell(mt['contrasts'].get('p1_start_minus_middle'), *p, 'p1startminusmiddle', 'matched')} \\\\")
    write("v2_truncation_p3_matched", "\n".join([
        head, r"% P3 recall on the matched subset: sources whose span is fully visible at all three positions",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{Rows kept} & \textbf{start} & \textbf{middle} & \textbf{end} \\",
        r"\midrule", *rec, r"\bottomrule", r"\end{tabular}", ""]))
    write("v2_truncation_contrasts", "\n".join([
        head, r"% Paired within-doc contrasts, full probe sets vs matched fully-visible subset",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"\textbf{Model} & \textbf{P3 end truncated} & \textbf{P3 end $-$ start (full)} & \textbf{P3 end $-$ start (matched)} & "
        r"\textbf{P2 start $-$ middle (matched)} & \textbf{P1 start $-$ middle (matched)} \\",
        r"\midrule", *con, r"\bottomrule", r"\end{tabular}", ""]))


# ─── TPR@FPR threshold protocols (results/v2_metrics/tpr_protocols/, latest run) ───

TPR_SETS = [("v2_test", "V2 test"), ("injecagent", "InjecAgent"), ("deepset", "deepset")]
TPR_MODELS = ["tfidf_c1", "tfidf_c100_tuned", "deberta_s42", "deberta_s123", "deberta_s999", "protectai", "piguard",
              "judge_qwen2.5_7b", "judge_qwen2.5_7b_vsum"]


def na(status) -> str:
    return status.replace("_", r"\_")


def tpr_cell(e, *parts):
    if e["status"] != "ok":
        return put(macro_name(*parts), na(e["status"]))
    return f"{num(e['tpr']['point'], *parts)} {{\\scriptsize{ci(e['tpr']['ci'], *parts, 'ci')}}}"


def tables_tpr_protocols():
    """Threshold-free metrics, FPR at the default threshold and oracle TPR@FPR (table v2_tpr_protocols); the
    calibrated protocol and its gap to oracle go to a separate table (v2_calibration_gap, macros VTwoCalgap*)."""
    m, src = latest_metrics("tpr_protocols", version="v2_metrics")
    if m is None:
        return
    rows = {(r["model"], r["eval_set"]): r for r in m["rows"]}
    main_rows, gap_rows = [], []
    for es, es_label in TPR_SETS:
        for model in TPR_MODELS:
            r = rows.get((model, es))
            if r is None:
                continue
            p, b, t = ("v2", "tpr", model, es), r["bootstrap"], r["tpr"]
            put(macro_name(*p, "nclean"), str(r["n_clean"]))
            put(macro_name(*p, "nclusters"), str(r["n_clusters"]))
            main_rows.append(
                f"{es_label} & {tex(model)} & {r['n_clean']} & {cell(b['roc_auc'], *p, 'auroc', digits=4)} & "
                f"{cell(b['fpr_default_clean'], *p, 'fprdefault', digits=4)} & "
                f"{tpr_cell(t['oracle@1pct_fpr'], *p, 'oracle1')} & {tpr_cell(t['oracle@5pct_fpr'], *p, 'oracle5')} & "
                f"{cell(b['f1_macro'], *p, 'f1macro', digits=4)} \\\\")
            if es == "deepset":
                continue
            g, c = ("v2", "calgap", model, es), t["calibrated@5pct_fpr"]
            fpr = cell(c["realized_fpr"], *g, "realizedfpr", digits=4) if c["status"] == "ok" else "--"
            thr = num(c["threshold"], *g, "threshold", digits=4) if c["status"] == "ok" else "--"
            ncal = put(macro_name(*g, "ncalib"), str(c["n_clean_calib"])) if "n_clean_calib" in c else "--"
            gap_rows.append(f"{es_label} & {tex(model)} & {ncal} & "
                            f"{tpr_cell(t['oracle@5pct_fpr'], *g, 'oracle5')} & {tpr_cell(c, *g, 'calibrated5')} & "
                            f"{thr} & {fpr} \\\\")
        for metric in ("roc_auc", "fpr_default_clean", "f1_macro", "tpr_oracle@5pct_fpr", "tpr_calibrated@5pct_fpr"):
            s = m["seed_summary"].get(f"deberta/{es}", {}).get(metric)
            if s:
                key = "calgap" if "calibrated" in metric else "tpr"
                num(s["mean"], "v2", key, "deberta_mean", es, metric, "mean")
                num(s["sd"], "v2", key, "deberta_mean", es, metric, "sd")
    head = f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json"
    write("v2_tpr_protocols", "\n".join([
        head, r"% Oracle protocol: threshold chosen on the clean rows of the evaluated set. TPR@1%FPR needs >= 500",
        r"% clean rows, TPR@5%FPR >= 100. [doc-cluster bootstrap 95% CI; InjecAgent: user case; deepset: rows]",
        r"\begin{tabular}{llrrrrrr}", r"\toprule",
        r"\textbf{Eval set} & \textbf{Model} & \textbf{$n_\text{clean}$} & \textbf{AUROC} & \textbf{FPR@default} & "
        r"\textbf{TPR@1\%FPR} & \textbf{TPR@5\%FPR} & \textbf{Macro-F1} \\",
        r"\midrule", *main_rows, r"\bottomrule", r"\end{tabular}", ""]))
    write("v2_calibration_gap", "\n".join([
        head, r"% PRELIMINARY: V2 val (the calibration set) shares the training attack pool. 5% FPR target.",
        r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"\textbf{Eval set} & \textbf{Model} & \textbf{$n_\text{clean}$ calib} & \textbf{TPR oracle} & "
        r"\textbf{TPR calibrated} & \textbf{Threshold} & \textbf{Realized FPR} \\",
        r"\midrule", *gap_rows, r"\bottomrule", r"\end{tabular}", ""]))


# ─── Provenance gate (docs/findings.src.md) ───────────────────────────────────

def _git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def run_config(run_dir):
    path = ROOT / run_dir / "config.json"
    return json.load(open(path)) if path.exists() else None


def run_closure(run_dirs):
    """Every run directory reachable from run_dirs through the run paths named in their config.json args."""
    seen, todo = set(), list(run_dirs)
    while todo:
        r = todo.pop()
        if r in seen:
            continue
        seen.add(r)
        cfg = run_config(r)
        if cfg:
            todo += RUN_RE.findall(json.dumps(cfg["args"]))
    return seen


def as_list(v):
    return v if isinstance(v, list) else [v]


def validate_entries():
    """Errors (list of str) and, per macro, the status of the entry that owns it."""
    errors, owner = [], {}
    _, entries = parse_entries()
    ids = [e["id"] for e, _, _ in entries]
    if len(ids) != len(set(ids)):
        errors.append(f"duplicate entry ids: {sorted(i for i in ids if ids.count(i) > 1)}")
    for e, _, _ in entries:
        eid = e["id"]
        miss = [k for k in REQUIRED if e.get(k) in (None, "", [])]
        if miss:
            errors.append(f"{eid}: missing fields {miss}")
            continue
        if e["status"] not in STATUSES:
            errors.append(f"{eid}: unknown status {e['status']!r}")
        if e["status"] not in NUMBERS_OK and not e.get("reason"):
            errors.append(f"{eid}: status {e['status']} needs a reason")
        if e["status"] == "pending" and not e.get("awaiting"):
            errors.append(f"{eid}: pending needs `awaiting`")
        sources = as_list(e["source"])
        runs = [s for s in sources if RUN_RE.fullmatch(s)]
        for r in runs:
            cfg = run_config(r)
            if cfg is None:
                errors.append(f"{eid}: run {r} has no config.json")
            elif cfg["data_version"] not in str(e["data_version"]):
                errors.append(f"{eid}: data_version {e['data_version']!r} does not name {r}'s {cfg['data_version']!r}")
        if runs:
            cfg = run_config(runs[0]) or {}
            if not cfg.get("git_commit", "").startswith(str(e["commit"])):
                errors.append(f"{eid}: commit {e['commit']} is not {runs[0]}'s {cfg.get('git_commit', '')[:7]}")
            dirty = sorted(r for r in run_closure(runs + as_list(e.get("upstream", [])))
                           if (run_config(r) or {}).get("git_dirty", True))
            if dirty and e["status"] in NUMBERS_OK:
                errors.append(f"{eid}: status {e['status']} but these runs are dirty or unknown: {dirty}")
            if not dirty and e["status"] == "linked-dirty":
                errors.append(f"{eid}: linked-dirty but every linked run is clean; upgrade it")
        mine = [k for k in macros if owns(e, k)]
        if not mine:
            errors.append(f"{eid}: its macros {e['macros']} match no generated macro")
        for k in mine:
            if k in owner:
                errors.append(f"macro {k} owned by both {owner[k][0]} and {eid}")
            owner[k] = (eid, e["status"])
            if not any(macro_src[k] == s or macro_src[k].startswith(s.rstrip("/") + "/") for s in sources):
                errors.append(f"{eid}: macro {k} comes from {macro_src[k]}, which is not in its source list")
    errors += [f"macro {k} (from {macro_src[k]}) is owned by no findings entry" for k in macros if k not in owner]
    return errors, owner


def write_macros(path, names, head):
    lines = [head] + [f"\\newcommand{{\\{k}}}{{{macros[k]}}}" for k in sorted(names)]
    path.write_text("\n".join(lines) + "\n")


def main():
    tf = json.load(open(RES / "tfidf_lr_results.json"))
    loto = json.load(open(RES / "leave_one_task_out.json"))
    abl = json.load(open(RES / "ablations.json"))
    table_main(tf)
    table_ablations(abl)
    table_per_task(tf)
    table_loto(loto)
    table_artifact_baselines()
    tables_probe_report()
    figure_probe_report()
    table_seed_shortcut_variance()
    tables_truncation_control()
    table_v2_test_groups()
    table_summary_json_bug()
    tables_tpr_protocols()
    table_v3_gates("prefreeze")
    table_v3_gates("frozen")
    macros_v3_sanity()
    macros_transfer_gate1()
    macros_transfer_gate2()
    table_transfer_detectors()
    macros_agent("agent_v1")
    macros_agent("agent_v2")
    errors, owner = validate_entries()
    if errors:
        print(f"{len(errors)} PROVENANCE ERRORS (docs/findings.src.md):")
        for e in errors:
            print("  " + e)
        sys.exit(1)
    PAPER.mkdir(exist_ok=True)
    commit = _git("rev-parse", "--short", "HEAD") + ("-dirty" if _git("status", "--porcelain", "--", "scripts", "src",
                                                                        "docs/findings.src.md") else "")
    full = [k for k, (_, st) in owner.items() if st in NUMBERS_OK]
    rest = [k for k in owner if k not in full]
    write_macros(NUMBERS, full, f"% Generated by scripts/make_tables.py at {commit} -- do not edit by hand. "
                                "Only findings entries with status full or recomputed-match.")
    write_macros(LEGACY, rest, f"% Generated by scripts/make_tables.py at {commit}. NOT for the paper: macros of "
                               "findings entries without full provenance (docs/findings.md says why).")
    print(f"wrote paper/numbers.tex ({len(full)} macros) and paper/legacy_numbers.tex ({len(rest)} macros)")
    print(f"\nVerification: {n_checked} values checked against frozen V2 results, tolerance {TOL}")
    if mismatches:
        print(f"{len(mismatches)} MISMATCHES:")
        for m in mismatches:
            print("  " + m)
        sys.exit(1)
    print("All values match.")


if __name__ == "__main__":
    main()
