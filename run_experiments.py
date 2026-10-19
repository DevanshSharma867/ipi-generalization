"""
run_experiments.py
==================
Comprehensive evaluation script for the leakage-free dataset:
- Deterministic TF-IDF + LR
- Hyperparameter tuning on VAL only: C in {1, 10, 100} and decision threshold
- Test evaluation at default (C=1, thr=0.5) and tuned settings
- Recall at 5% and 10% FPR on test set
- Cluster bootstrap at DOC LEVEL (resample doc_id, 1,000 iters) for ALL 95% CIs
- Per-task metrics with doc-level cluster bootstrap CIs
- Clean Leave-One-Task-Out (LOTO): train ONLY on train-split docs of other two tasks,
  test ONLY on test-split docs of held-out task (held-out test attacks)
- Code-only attack-pool shift reported separately
- Saves predictions CSVs for standard test and all LOTO folds
- Ablations and qualitative error analysis
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import sys
sys.path.insert(0, "src")
from preprocess import preprocess_for_tfidf
from metrics import point_metrics

DATA_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "confusion_matrices"), exist_ok=True)

# Hyperparameter search space
TFIDF_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 3),
    "max_features": 50_000,
    "sublinear_tf": True,
    "min_df": 2,
    "strip_accents": "unicode",
    "token_pattern": r"\b\w+\b",
}

LR_BASE_PARAMS = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
}


def build_fused_texts(df: pd.DataFrame, mode: str = "fused") -> list:
    """
    mode: 'fused' (Intent: <intent> | Context: <text>),
          'context_only' (<text>),
          'intent_only' (<intent>)
    """
    if mode == "context_only":
        return df["text"].astype(str).tolist()
    elif mode == "intent_only":
        return df["user_intent"].astype(str).tolist()
    else:
        res = []
        for _, r in df.iterrows():
            intent = str(r.get("user_intent", "")).strip()
            ctx = str(r.get("text", "")).strip()
            if intent and intent != "nan":
                res.append(f"Intent: {intent} | Context: {ctx}")
            else:
                res.append(ctx)
        return res


def compute_metrics_at_threshold(y_true, y_prob, threshold=0.5):
    # Delegates to src/metrics.py; single-class ROC-AUC stays 0.0 as before
    m = point_metrics(y_true, y_prob, threshold=threshold)
    out = {"threshold": float(threshold)}
    out.update({k: m[k] for k in ("accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc")})
    if np.isnan(out["roc_auc"]):
        out["roc_auc"] = 0.0
    return out


def compute_recall_at_fpr(y_true, y_prob, target_fpr):
    """
    Finds the threshold where empirical FPR <= target_fpr,
    and returns the recall (TPR) at that threshold.
    """
    clean_mask = (y_true == 0)
    inj_mask = (y_true == 1)
    n_clean = np.sum(clean_mask)
    n_inj = np.sum(inj_mask)

    if n_clean == 0 or n_inj == 0:
        return 0.0, 1.0

    # Sort thresholds descending
    thresholds = np.sort(np.unique(y_prob))[::-1]
    best_recall = 0.0
    operating_threshold = 1.0

    for thr in thresholds:
        fpr = np.sum(y_prob[clean_mask] >= thr) / n_clean
        if fpr <= target_fpr:
            rec = np.sum(y_prob[inj_mask] >= thr) / n_inj
            if rec >= best_recall:
                best_recall = rec
                operating_threshold = thr
        else:
            break

    return float(best_recall), float(operating_threshold)


def doc_cluster_bootstrap(df: pd.DataFrame, y_prob: np.ndarray, threshold: float = 0.5,
                           n_bootstraps: int = 1000, ci: float = 95.0, seed: int = 42):
    """
    Doc-level cluster bootstrap: resamples unique doc_id with replacement.
    All rows belonging to a sampled doc_id are selected.
    Computes 95% CI for: accuracy, f1_macro, f1_injected, precision, recall, roc_auc,
    recall_at_5pct_fpr, recall_at_10pct_fpr.
    """
    rng = np.random.RandomState(seed)
    unique_docs = df["doc_id"].unique()
    n_docs = len(unique_docs)

    # Pre-group indices by doc_id for fast retrieval
    doc_to_indices = {doc: np.where(df["doc_id"].values == doc)[0] for doc in unique_docs}
    y_true_all = df["label"].values

    boot_metrics = {
        "accuracy": [],
        "f1_macro": [],
        "f1_injected": [],
        "precision": [],
        "recall": [],
        "roc_auc": [],
        "recall_at_5pct_fpr": [],
        "recall_at_10pct_fpr": [],
    }

    for _ in range(n_bootstraps):
        sampled_docs = rng.choice(unique_docs, size=n_docs, replace=True)
        sampled_indices = np.concatenate([doc_to_indices[doc] for doc in sampled_docs])

        b_y_true = y_true_all[sampled_indices]
        b_y_prob = y_prob[sampled_indices]

        # Compute metrics
        m = compute_metrics_at_threshold(b_y_true, b_y_prob, threshold=threshold)
        for k in ["accuracy", "f1_macro", "f1_injected", "precision", "recall", "roc_auc"]:
            boot_metrics[k].append(m[k])

        rec_5, _ = compute_recall_at_fpr(b_y_true, b_y_prob, 0.05)
        rec_10, _ = compute_recall_at_fpr(b_y_true, b_y_prob, 0.10)
        boot_metrics["recall_at_5pct_fpr"].append(rec_5)
        boot_metrics["recall_at_10pct_fpr"].append(rec_10)

    alpha = (100.0 - ci) / 2.0
    cis = {}
    for k, vals in boot_metrics.items():
        low = np.percentile(vals, alpha)
        high = np.percentile(vals, 100.0 - alpha)
        cis[k] = [float(low), float(high)]

    return cis


def main():
    print("=" * 60)
    print("RUNNING BENCHMARKS (DETERMINISTIC, TUNED ON VAL, DOC BOOTSTRAP)")
    print("=" * 60)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    tr_raw = build_fused_texts(train_df, "fused")
    val_raw = build_fused_texts(val_df, "fused")
    te_raw = build_fused_texts(test_df, "fused")

    tr_clean = preprocess_for_tfidf(tr_raw)
    val_clean = preprocess_for_tfidf(val_raw)
    te_clean = preprocess_for_tfidf(te_raw)

    vec = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr = vec.fit_transform(tr_clean)
    X_val = vec.transform(val_clean)
    X_te = vec.transform(te_clean)

    # -------------------------------------------------------------
    # 1. Tuning C in {1, 10, 100} and Decision Threshold on VAL only
    # -------------------------------------------------------------
    print("\n--- 1. Validation Tuning (VAL Split Only, Train-File Attacks) ---")
    c_candidates = [1.0, 10.0, 100.0]
    threshold_grid = np.linspace(0.05, 0.95, 91)  # step 0.01

    tuning_results = []
    best_config = None
    best_val_f1 = -1.0

    fitted_models = {}

    for c in c_candidates:
        lr = LogisticRegression(**LR_BASE_PARAMS, C=c)
        lr.fit(X_tr, y_train)
        fitted_models[c] = lr

        val_probs = lr.predict_proba(X_val)[:, 1]

        best_c_thr = 0.5
        best_c_f1 = -1.0
        for thr in threshold_grid:
            m = compute_metrics_at_threshold(y_val, val_probs, threshold=thr)
            if m["f1_macro"] > best_c_f1:
                best_c_f1 = m["f1_macro"]
                best_c_thr = thr

        default_val_m = compute_metrics_at_threshold(y_val, val_probs, threshold=0.5)
        tuned_val_m = compute_metrics_at_threshold(y_val, val_probs, threshold=best_c_thr)

        entry = {
            "C": c,
            "best_threshold": float(best_c_thr),
            "val_f1_macro_default_thr0.5": default_val_m["f1_macro"],
            "val_f1_macro_tuned_thr": tuned_val_m["f1_macro"],
            "val_accuracy": tuned_val_m["accuracy"],
            "val_precision": tuned_val_m["precision"],
            "val_recall": tuned_val_m["recall"],
            "val_roc_auc": tuned_val_m["roc_auc"],
        }
        tuning_results.append(entry)
        print(f"  C={c:5.1f} | Default Thr (0.5) Val Macro-F1: {default_val_m['f1_macro']:.4f} | "
              f"Best Thr ({best_c_thr:.2f}) Val Macro-F1: {best_c_f1:.4f}")

        if best_c_f1 > best_val_f1:
            best_val_f1 = best_c_f1
            best_config = {"C": c, "threshold": float(best_c_thr)}

    print(f"\nOptimal Val Selection: C={best_config['C']}, threshold={best_config['threshold']:.2f} (Val Macro-F1: {best_val_f1:.4f})")

    # -------------------------------------------------------------
    # 2. Test Set Evaluation: Default vs Tuned & Operating Points
    # -------------------------------------------------------------
    print("\n--- 2. Test Set Evaluation (Held-Out Test Attacks) ---")
    # Default model: C=1.0, threshold=0.5
    default_lr = fitted_models[1.0]
    default_test_probs = default_lr.predict_proba(X_te)[:, 1]
    default_test_metrics = compute_metrics_at_threshold(y_test, default_test_probs, threshold=0.5)
    default_rec_5, default_thr_5 = compute_recall_at_fpr(y_test, default_test_probs, 0.05)
    default_rec_10, default_thr_10 = compute_recall_at_fpr(y_test, default_test_probs, 0.10)
    default_test_metrics["recall_at_5pct_fpr"] = default_rec_5
    default_test_metrics["operating_threshold_at_5pct_fpr"] = default_thr_5
    default_test_metrics["recall_at_10pct_fpr"] = default_rec_10
    default_test_metrics["operating_threshold_at_10pct_fpr"] = default_thr_10

    # Doc-level cluster bootstrap for Default settings
    default_cis = doc_cluster_bootstrap(test_df, default_test_probs, threshold=0.5, n_bootstraps=1000, seed=42)

    print(f"\n[DEFAULT SETTINGS: C=1.0, Threshold=0.50]")
    print(f"  Test Accuracy:        {default_test_metrics['accuracy']:.4f}  (95% CI: [{default_cis['accuracy'][0]:.4f}, {default_cis['accuracy'][1]:.4f}])")
    print(f"  Test Macro-F1:        {default_test_metrics['f1_macro']:.4f}  (95% CI: [{default_cis['f1_macro'][0]:.4f}, {default_cis['f1_macro'][1]:.4f}])")
    print(f"  Injected F1:          {default_test_metrics['f1_injected']:.4f}  (95% CI: [{default_cis['f1_injected'][0]:.4f}, {default_cis['f1_injected'][1]:.4f}])")
    print(f"  Injected Precision:   {default_test_metrics['precision']:.4f}  (95% CI: [{default_cis['precision'][0]:.4f}, {default_cis['precision'][1]:.4f}])")
    print(f"  Injected Recall:      {default_test_metrics['recall']:.4f}  (95% CI: [{default_cis['recall'][0]:.4f}, {default_cis['recall'][1]:.4f}])")
    print(f"  Test ROC-AUC:         {default_test_metrics['roc_auc']:.4f}  (95% CI: [{default_cis['roc_auc'][0]:.4f}, {default_cis['roc_auc'][1]:.4f}])")
    print(f"  Recall @ 5% FPR:      {default_rec_5:.4f} (thr={default_thr_5:.4f})  (95% CI: [{default_cis['recall_at_5pct_fpr'][0]:.4f}, {default_cis['recall_at_5pct_fpr'][1]:.4f}])")
    print(f"  Recall @ 10% FPR:     {default_rec_10:.4f} (thr={default_thr_10:.4f})  (95% CI: [{default_cis['recall_at_10pct_fpr'][0]:.4f}, {default_cis['recall_at_10pct_fpr'][1]:.4f}])")

    # Tuned model: C=best_config['C'], threshold=best_config['threshold']
    tuned_lr = fitted_models[best_config["C"]]
    tuned_test_probs = tuned_lr.predict_proba(X_te)[:, 1]
    tuned_test_metrics = compute_metrics_at_threshold(y_test, tuned_test_probs, threshold=best_config["threshold"])
    tuned_rec_5, tuned_thr_5 = compute_recall_at_fpr(y_test, tuned_test_probs, 0.05)
    tuned_rec_10, tuned_thr_10 = compute_recall_at_fpr(y_test, tuned_test_probs, 0.10)
    tuned_test_metrics["recall_at_5pct_fpr"] = tuned_rec_5
    tuned_test_metrics["operating_threshold_at_5pct_fpr"] = tuned_thr_5
    tuned_test_metrics["recall_at_10pct_fpr"] = tuned_rec_10
    tuned_test_metrics["operating_threshold_at_10pct_fpr"] = tuned_thr_10

    # Doc-level cluster bootstrap for Tuned settings
    tuned_cis = doc_cluster_bootstrap(test_df, tuned_test_probs, threshold=best_config["threshold"], n_bootstraps=1000, seed=42)

    print(f"\n[TUNED SETTINGS: C={best_config['C']}, Threshold={best_config['threshold']:.2f}]")
    print(f"  Test Accuracy:        {tuned_test_metrics['accuracy']:.4f}  (95% CI: [{tuned_cis['accuracy'][0]:.4f}, {tuned_cis['accuracy'][1]:.4f}])")
    print(f"  Test Macro-F1:        {tuned_test_metrics['f1_macro']:.4f}  (95% CI: [{tuned_cis['f1_macro'][0]:.4f}, {tuned_cis['f1_macro'][1]:.4f}])")
    print(f"  Injected F1:          {tuned_test_metrics['f1_injected']:.4f}  (95% CI: [{tuned_cis['f1_injected'][0]:.4f}, {tuned_cis['f1_injected'][1]:.4f}])")
    print(f"  Injected Precision:   {tuned_test_metrics['precision']:.4f}  (95% CI: [{tuned_cis['precision'][0]:.4f}, {tuned_cis['precision'][1]:.4f}])")
    print(f"  Injected Recall:      {tuned_test_metrics['recall']:.4f}  (95% CI: [{tuned_cis['recall'][0]:.4f}, {tuned_cis['recall'][1]:.4f}])")
    print(f"  Test ROC-AUC:         {tuned_test_metrics['roc_auc']:.4f}  (95% CI: [{tuned_cis['roc_auc'][0]:.4f}, {tuned_cis['roc_auc'][1]:.4f}])")
    print(f"  Recall @ 5% FPR:      {tuned_rec_5:.4f} (thr={tuned_thr_5:.4f})  (95% CI: [{tuned_cis['recall_at_5pct_fpr'][0]:.4f}, {tuned_cis['recall_at_5pct_fpr'][1]:.4f}])")
    print(f"  Recall @ 10% FPR:     {tuned_rec_10:.4f} (thr={tuned_thr_10:.4f})  (95% CI: [{tuned_cis['recall_at_10pct_fpr'][0]:.4f}, {tuned_cis['recall_at_10pct_fpr'][1]:.4f}])")

    # Save predictions CSV (both default and tuned predictions)
    pred_df = test_df[["id", "doc_id", "task_type", "attack_type", "attack_source", "label"]].copy()
    pred_df["prob_injected"] = default_test_probs
    pred_df["pred_default"] = (default_test_probs >= 0.5).astype(int)
    pred_df["prob_injected_tuned"] = tuned_test_probs
    pred_df["pred_tuned"] = (tuned_test_probs >= best_config["threshold"]).astype(int)
    pred_df["pred"] = pred_df["pred_default"]  # standard compatibility
    pred_csv_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"\nSaved test predictions to: {pred_csv_path}")

    # Per-task breakdown with doc-level cluster bootstrap
    print("\n--- Per-Task Breakdown (Default Settings, Doc-Level Bootstrap) ---")
    task_breakdown = {}
    for task in ["email", "table", "code"]:
        task_mask = (test_df["task_type"] == task).values
        task_df = test_df[task_mask].reset_index(drop=True)
        task_probs = default_test_probs[task_mask]
        m_task = compute_metrics_at_threshold(task_df["label"].values, task_probs, threshold=0.5)
        task_cis = doc_cluster_bootstrap(task_df, task_probs, threshold=0.5, n_bootstraps=1000, seed=42)
        task_breakdown[task] = {
            "metrics": m_task,
            "cluster_bootstrap_95ci": task_cis,
        }
        print(f"  Task {task:5s} | Acc: {m_task['accuracy']:.4f} [{task_cis['accuracy'][0]:.4f}, {task_cis['accuracy'][1]:.4f}] | "
              f"Macro-F1: {m_task['f1_macro']:.4f} [{task_cis['f1_macro'][0]:.4f}, {task_cis['f1_macro'][1]:.4f}] | "
              f"Prec: {m_task['precision']:.4f} [{task_cis['precision'][0]:.4f}, {task_cis['precision'][1]:.4f}] | "
              f"Rec: {m_task['recall']:.4f} [{task_cis['recall'][0]:.4f}, {task_cis['recall'][1]:.4f}] | "
              f"AUC: {m_task['roc_auc']:.4f} [{task_cis['roc_auc'][0]:.4f}, {task_cis['roc_auc'][1]:.4f}]")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred_df["pred_default"]).tolist()

    # FP and FN qualitative examples
    fps = pred_df[(pred_df["label"] == 0) & (pred_df["pred"] == 1)].sort_values("prob_injected", ascending=False).head(20).to_dict("records")
    fns = pred_df[(pred_df["label"] == 1) & (pred_df["pred"] == 0)].sort_values("prob_injected", ascending=True).head(20).to_dict("records")

    # Top coefficients
    feat_names = np.array(vec.get_feature_names_out())
    coefs = default_lr.coef_[0]
    top_inj_idx = np.argsort(coefs)[::-1][:20]
    top_clean_idx = np.argsort(coefs)[:20]

    top_features = {
        "top_injected_features": [{"feature": str(feat_names[i]), "coef": float(coefs[i])} for i in top_inj_idx],
        "top_clean_features": [{"feature": str(feat_names[i]), "coef": float(coefs[i])} for i in top_clean_idx],
    }

    test_summary = {
        "deterministic": True,
        "solver": "lbfgs",
        "val_tuning": {
            "grid_evaluated": tuning_results,
            "optimal_config": best_config,
        },
        "default_evaluation": {
            "metrics": default_test_metrics,
            "doc_cluster_bootstrap_95ci": default_cis,
        },
        "tuned_evaluation": {
            "metrics": tuned_test_metrics,
            "doc_cluster_bootstrap_95ci": tuned_cis,
        },
        "per_task_breakdown": task_breakdown,
        "confusion_matrix": cm,
        "top_features": top_features,
    }

    with open(os.path.join(RESULTS_DIR, "tfidf_lr_results.json"), "w") as f:
        json.dump(test_summary, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "error_examples_20fp_20fn.json"), "w") as f:
        json.dump({"false_positives_top20": fps, "false_negatives_top20": fns}, f, indent=2)

    # -------------------------------------------------------------
    # 3. Clean Leave-One-Task-Out (LOTO) Cross-Validation
    # -------------------------------------------------------------
    print("\n--- 3. Clean Leave-One-Task-Out (LOTO) Evaluation ---")
    print("Protocol: Train on TRAIN-split docs of other 2 tasks only; Test on TEST-split docs of held-out task.")
    tasks = ["email", "table", "code"]
    loto_results = {}

    for test_task in tasks:
        train_tasks = [t for t in tasks if t != test_task]
        df_loto_tr = train_df[train_df["task_type"].isin(train_tasks)].copy().reset_index(drop=True)
        df_loto_te = test_df[test_df["task_type"] == test_task].copy().reset_index(drop=True)

        y_loto_tr = df_loto_tr["label"].values
        y_loto_te = df_loto_te["label"].values

        vec_loto = TfidfVectorizer(**TFIDF_PARAMS)
        X_loto_tr = vec_loto.fit_transform(preprocess_for_tfidf(build_fused_texts(df_loto_tr, "fused")))
        X_loto_te = vec_loto.transform(preprocess_for_tfidf(build_fused_texts(df_loto_te, "fused")))

        lr_loto = LogisticRegression(**LR_BASE_PARAMS, C=best_config["C"])
        lr_loto.fit(X_loto_tr, y_loto_tr)
        prob_loto = lr_loto.predict_proba(X_loto_te)[:, 1]
        p_loto = (prob_loto >= 0.5).astype(int)

        m_loto = compute_metrics_at_threshold(y_loto_te, prob_loto, threshold=0.5)
        loto_cis = doc_cluster_bootstrap(df_loto_te, prob_loto, threshold=0.5, n_bootstraps=1000, seed=42)

        # Save fold predictions CSV
        df_loto_te_preds = df_loto_te[["id", "doc_id", "task_type", "attack_type", "attack_source", "label"]].copy()
        df_loto_te_preds["prob_injected"] = prob_loto
        df_loto_te_preds["pred"] = p_loto
        fold_csv_path = os.path.join(RESULTS_DIR, f"loto_predictions_{test_task}.csv")
        df_loto_te_preds.to_csv(fold_csv_path, index=False)

        loto_results[f"held_out_{test_task}"] = {
            "train_tasks": train_tasks,
            "test_task": test_task,
            "n_train_samples": len(df_loto_tr),
            "n_train_docs": int(df_loto_tr["doc_id"].nunique()),
            "n_test_samples": len(df_loto_te),
            "n_test_docs": int(df_loto_te["doc_id"].nunique()),
            "metrics": m_loto,
            "doc_cluster_bootstrap_95ci": loto_cis,
            "predictions_csv": fold_csv_path,
        }

        print(f"  Held-out '{test_task:5s}' (Train: {train_tasks}) | N_tr={len(df_loto_tr)}, N_te={len(df_loto_te)} | "
              f"Acc: {m_loto['accuracy']:.4f} [{loto_cis['accuracy'][0]:.4f}, {loto_cis['accuracy'][1]:.4f}] | "
              f"Macro-F1: {m_loto['f1_macro']:.4f} [{loto_cis['f1_macro'][0]:.4f}, {loto_cis['f1_macro'][1]:.4f}] | "
              f"Inj-F1: {m_loto['f1_injected']:.4f} | Prec: {m_loto['precision']:.4f} | Rec: {m_loto['recall']:.4f} | "
              f"AUC: {m_loto['roc_auc']:.4f} [{loto_cis['roc_auc'][0]:.4f}, {loto_cis['roc_auc'][1]:.4f}]")

    # Separate Code-Only Attack-Pool Shift Analysis
    print("\n--- 4. Code-Only Attack-Pool Shift Analysis ---")
    # In LOTO on code, the training set (email + table) only contains text attacks (text_attack_train.json).
    # The test set on code contains code attacks (code_attack_test.json).
    code_fold = loto_results["held_out_code"]
    code_preds_df = pd.read_csv(code_fold["predictions_csv"])
    clean_code = code_preds_df[code_preds_df["label"] == 0]
    inj_code = code_preds_df[code_preds_df["label"] == 1]
    code_fp_rate = float((clean_code["pred"] == 1).mean())
    code_fn_rate = float((inj_code["pred"] == 0).mean())
    code_recall = float((inj_code["pred"] == 1).mean())

    # In-distribution comparison from standard model:
    indist_code_m = task_breakdown["code"]["metrics"]

    code_shift_analysis = {
        "finding": "Holding out CodeQA exposes the detector to a compound shift: (1) Domain/Task context shift (Text/Table QA -> Code QA) AND (2) Cross-modality Attack-Pool shift (training saw only text_attack_train.json; testing evaluates code_attack_test.json). In contrast, Email and Table folds had code attacks present in their training data.",
        "in_distribution_code_test": {
            "macro_f1": indist_code_m["f1_macro"],
            "recall": indist_code_m["recall"],
            "precision": indist_code_m["precision"],
            "roc_auc": indist_code_m["roc_auc"],
        },
        "loto_held_out_code_test": {
            "macro_f1": code_fold["metrics"]["f1_macro"],
            "recall": code_fold["metrics"]["recall"],
            "precision": code_fold["metrics"]["precision"],
            "roc_auc": code_fold["metrics"]["roc_auc"],
            "false_positive_rate": code_fp_rate,
            "false_negative_rate": code_fn_rate,
        },
        "macro_f1_delta": float(code_fold["metrics"]["f1_macro"] - indist_code_m["f1_macro"]),
        "recall_delta": float(code_fold["metrics"]["recall"] - indist_code_m["recall"]),
    }
    loto_results["code_only_attack_pool_shift_analysis"] = code_shift_analysis

    print(f"  In-Distribution CodeQA Test: Macro-F1 = {indist_code_m['f1_macro']:.4f}, Recall = {indist_code_m['recall']:.4f}, AUC = {indist_code_m['roc_auc']:.4f}")
    print(f"  Held-Out LOTO CodeQA Test:   Macro-F1 = {code_fold['metrics']['f1_macro']:.4f}, Recall = {code_fold['metrics']['recall']:.4f}, AUC = {code_fold['metrics']['roc_auc']:.4f}")
    print(f"  Shift Delta (LOTO - InDist): Macro-F1 Delta = {code_shift_analysis['macro_f1_delta']:+.4f}, Recall Delta = {code_shift_analysis['recall_delta']:+.4f}")
    print(f"  Code Test Breakdown: FP Rate on Clean Code = {code_fp_rate*100:.1f}%, FN Rate on Code Attacks = {code_fn_rate*100:.1f}%")

    with open(os.path.join(RESULTS_DIR, "leave_one_task_out.json"), "w") as f:
        json.dump(loto_results, f, indent=2)

    # -------------------------------------------------------------
    # 5. Ablations (Majority, Length, Context-Only, Intent-Only)
    # -------------------------------------------------------------
    print("\n--- 5. Feature Ablations ---")
    ablations = {}

    # (a) Majority Class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df[["text"]], y_train)
    p_maj = dummy.predict(test_df[["text"]])
    ablations["majority_class"] = {
        "macro_f1": float(f1_score(y_test, p_maj, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_test, p_maj)),
    }
    print(f"  Majority Class Baseline: Macro-F1 = {ablations['majority_class']['macro_f1']:.4f}")

    # (b) Text Length Only
    def len_feats(df):
        c = df["text"].astype(str).str.len().values.reshape(-1, 1)
        w = df["text"].astype(str).str.split().str.len().values.reshape(-1, 1)
        return np.hstack([c, w])

    lr_len = LogisticRegression(**LR_BASE_PARAMS, C=1.0)
    lr_len.fit(len_feats(train_df), y_train)
    prob_len = lr_len.predict_proba(len_feats(test_df))[:, 1]
    p_len = (prob_len >= 0.5).astype(int)
    ablations["length_only"] = compute_metrics_at_threshold(y_test, prob_len, threshold=0.5)
    print(f"  Length-Only Baseline:   Macro-F1 = {ablations['length_only']['f1_macro']:.4f}")

    # (c) Context Only (no "Intent:" prefix)
    vec_ctx = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr_ctx = vec_ctx.fit_transform(preprocess_for_tfidf(build_fused_texts(train_df, "context_only")))
    X_te_ctx = vec_ctx.transform(preprocess_for_tfidf(build_fused_texts(test_df, "context_only")))
    lr_ctx = LogisticRegression(**LR_BASE_PARAMS, C=1.0)
    lr_ctx.fit(X_tr_ctx, y_train)
    prob_ctx = lr_ctx.predict_proba(X_te_ctx)[:, 1]
    ablations["context_only"] = compute_metrics_at_threshold(y_test, prob_ctx, threshold=0.5)
    print(f"  Context-Only Baseline:  Macro-F1 = {ablations['context_only']['f1_macro']:.4f}")

    # (d) Intent Only
    vec_int = TfidfVectorizer(**TFIDF_PARAMS)
    X_tr_int = vec_int.fit_transform(preprocess_for_tfidf(build_fused_texts(train_df, "intent_only")))
    X_te_int = vec_int.transform(preprocess_for_tfidf(build_fused_texts(test_df, "intent_only")))
    lr_int = LogisticRegression(**LR_BASE_PARAMS, C=1.0)
    lr_int.fit(X_tr_int, y_train)
    prob_int = lr_int.predict_proba(X_te_int)[:, 1]
    ablations["intent_only"] = compute_metrics_at_threshold(y_test, prob_int, threshold=0.5)
    print(f"  Intent-Only Baseline:   Macro-F1 = {ablations['intent_only']['f1_macro']:.4f}")

    with open(os.path.join(RESULTS_DIR, "ablations.json"), "w") as f:
        json.dump(ablations, f, indent=2)

    print("\nAll experiments complete. Results saved to results/")


if __name__ == "__main__":
    main()
