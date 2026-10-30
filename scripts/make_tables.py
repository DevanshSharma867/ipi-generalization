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


def table_v3_prefreeze_gates():
    """Content-free gates (v3_design §8, R2) and integrity checks on the 5 pre-freeze split realizations."""
    m, src = latest_metrics("artifact_baselines_prefreeze", version="v3")
    if m is None:
        return
    g = m["gate_summary"]
    rows = []
    for name, e in g["probes"].items():
        p = ("v3", "prefreeze", name)
        pooled = list(e["pooled_by_seed"].values())
        lo, hi = num(min(pooled), *p, "pooled_min"), num(max(pooled), *p, "pooled_max")
        tasks = [num(e["per_task_mean"][t], *p, "task_mean", t) for t, _ in TASKS]
        status = put(macro_name(*p, "status"), ("PASS" if e["pass"] else "FAIL") if e["gate"] else "diag")
        rows.append(f"{tex(name)} & {lo}--{hi} & {' & '.join(tasks)} & {status} \\\\")
    put(macro_name("v3", "prefreeze", "gate", "threshold"), f"{g['threshold']:.2f}")
    put(macro_name("v3", "prefreeze", "gates", "status"), "PASS" if g["pass"] else "FAIL")
    write("v3_prefreeze_gates", "\n".join([
        f"% Generated by scripts/make_tables.py from {src.relative_to(ROOT)}/metrics.json",
        r"\begin{tabular}{lrrrrl}", r"\toprule",
        r"\textbf{Probe} & \textbf{Pooled macro-F1 (seeds 0--4)} & \textbf{EmailQA} & \textbf{TableQA} & "
        r"\textbf{CodeQA} & \textbf{Gate} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""]))

    m, src = latest_metrics("integrity_prefreeze", version="v3")
    if m is None:
        return
    put(macro_name("v3", "prefreeze", "integrity", "status"), "PASS" if m["pass"] else "FAIL")
    for s, r in m["seeds"].items():
        p = ("v3", "prefreeze", "integrity", "seed", s)
        put(macro_name(*p, "status"), "PASS" if r["pass"] else "FAIL")
        for k, n in r["rows"].items():
            put(macro_name(*p, "rows", k), str(n))
        for k, n in r["main_by_task_split"].items():
            put(macro_name(*p, "main", k), str(n))


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
    table_v3_prefreeze_gates()
    macros_transfer_gate1()
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
