"""
evaluate.py
===========
Evaluation and Error Analysis Module

Produces:
  1. Comprehensive metrics table (all models side by side)
  2. Confusion matrices (saved as PNG)
  3. Quantitative error analysis (per task_type, per attack_type)
  4. Qualitative error examples (False Positives + False Negatives)
  5. Cross-task generalization preliminary table
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import get_tfidf_data, get_deberta_data


# ─── Plotting style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})


def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    model_name: str,
    output_dir: str = "results/confusion_matrices",
) -> str:
    """Plot and save confusion matrix. Returns path to saved PNG."""
    os.makedirs(output_dir, exist_ok=True)
    
    cm = confusion_matrix(labels, preds)
    cm_norm = confusion_matrix(labels, preds, normalize="true")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Clean/Benign", "Injected"],
                yticklabels=["Clean/Benign", "Injected"],
                ax=axes[0])
    axes[0].set_title(f"{model_name}\nConfusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")
    
    # Normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=["Clean/Benign", "Injected"],
                yticklabels=["Clean/Benign", "Injected"],
                ax=axes[1])
    axes[1].set_title(f"{model_name}\nConfusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_cm.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    return out_path


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    output_dir: str = "results",
) -> str:
    """Plot ROC curves for all models on the same plot."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
    
    for (model_name, (labels, probs)), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{model_name} (AUC = {roc_auc:.3f})")
    
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — IPI Detection")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    return out_path


def error_analysis_by_group(
    df: pd.DataFrame,
    preds: np.ndarray,
    group_col: str = "task_type",
) -> pd.DataFrame:
    """
    Compute per-group accuracy and error rates.
    
    Shows:
    - Whether errors cluster in specific task types (EmailQA vs TableQA vs CodeQA)
    - Whether certain attack types are harder to detect
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    rows = []
    df = df.copy()
    df["pred"] = preds
    
    for group in df[group_col].unique():
        subset = df[df[group_col] == group]
        n = len(subset)
        labels = subset["label"].values
        group_preds = subset["pred"].values
        
        rows.append({
            "group": group,
            "n_samples": n,
            "n_injected": int(labels.sum()),
            "n_clean": int((labels == 0).sum()),
            "accuracy": accuracy_score(labels, group_preds),
            "f1_macro": f1_score(labels, group_preds, average="macro", zero_division=0),
            "f1_injected": f1_score(labels, group_preds, pos_label=1, average="binary", zero_division=0),
            "false_positives": int(((labels == 0) & (group_preds == 1)).sum()),
            "false_negatives": int(((labels == 1) & (group_preds == 0)).sum()),
        })
    
    return pd.DataFrame(rows).sort_values("f1_macro", ascending=True)


def get_failure_examples(
    df: pd.DataFrame,
    preds: np.ndarray,
    n_fp: int = 5,
    n_fn: int = 5,
) -> Dict[str, List[Dict]]:
    """
    Extract False Positive and False Negative examples for qualitative analysis.
    
    False Positives: benign text misclassified as injection
      → Reveals what surface patterns fool the model (over-sensitivity)
    
    False Negatives: injection missed by classifier
      → Reveals evasion patterns (under-sensitivity)
    """
    df = df.copy()
    df["pred"] = preds
    
    # False Positives: true label=0, predicted=1
    fp_mask = (df["label"] == 0) & (df["pred"] == 1)
    fp_examples = df[fp_mask].head(n_fp)[
        ["text", "label", "pred", "task_type", "attack_type"]
    ].to_dict("records")
    
    # False Negatives: true label=1, predicted=0
    fn_mask = (df["label"] == 1) & (df["pred"] == 0)
    fn_examples = df[fn_mask].head(n_fn)[
        ["text", "label", "pred", "task_type", "attack_type"]
    ].to_dict("records")
    
    return {
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
    }


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted side-by-side comparison table."""
    print("\n" + "=" * 75)
    print("MODEL COMPARISON TABLE")
    print("=" * 75)
    
    header = f"{'Model':<25} {'Acc':>8} {'F1-Mac':>8} {'F1-Inj':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}"
    print(header)
    print("-" * 75)
    
    for model_name, metrics in results.items():
        prefix = list(metrics.keys())[0].split("_")[0]  # e.g., 'test'
        acc = metrics.get(f"{prefix}_accuracy", 0)
        f1m = metrics.get(f"{prefix}_f1_macro", 0)
        f1i = metrics.get(f"{prefix}_f1_injected", 0)
        pre = metrics.get(f"{prefix}_precision", 0)
        rec = metrics.get(f"{prefix}_recall", 0)
        auc_score = metrics.get(f"{prefix}_roc_auc", 0)
        print(f"{model_name:<25} {acc:>8.4f} {f1m:>8.4f} {f1i:>8.4f} {pre:>8.4f} {rec:>8.4f} {auc_score:>8.4f}")
    
    print("=" * 75)


def run_full_evaluation(
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete evaluation for both models and generate all analysis artifacts.
    Returns combined results dict.
    """
    os.makedirs(results_dir, exist_ok=True)
    all_results = {}
    
    # ── 1. TF-IDF + LR ───────────────────────────────────────────────────────
    tfidf_ckpt = os.path.join(checkpoint_dir, "tfidf_lr_model.pkl")
    if os.path.exists(tfidf_ckpt):
        if verbose:
            print("\nEvaluating TF-IDF + LR...")
        
        with open(tfidf_ckpt, "rb") as f:
            pipeline = pickle.load(f)
        
        from preprocess import get_tfidf_data, preprocess_for_tfidf
        _, _, test_texts, _, _, test_labels = get_tfidf_data(data_dir)
        
        tfidf_preds = pipeline.predict(test_texts)
        tfidf_probs = pipeline.predict_proba(test_texts)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        tfidf_metrics = {
            "test_accuracy": accuracy_score(test_labels, tfidf_preds),
            "test_f1_macro": f1_score(test_labels, tfidf_preds, average="macro"),
            "test_f1_injected": f1_score(test_labels, tfidf_preds, pos_label=1, average="binary"),
            "test_precision": precision_score(test_labels, tfidf_preds, pos_label=1, average="binary"),
            "test_recall": recall_score(test_labels, tfidf_preds, pos_label=1, average="binary"),
            "test_roc_auc": roc_auc_score(test_labels, tfidf_probs),
        }
        
        all_results["TF-IDF + LR"] = {
            "metrics": tfidf_metrics,
            "preds": tfidf_preds,
            "probs": tfidf_probs,
            "labels": test_labels,
        }
        
        cm_path = plot_confusion_matrix(test_labels, tfidf_preds, "TF-IDF LR", results_dir + "/confusion_matrices")
        if verbose:
            print(f"  Confusion matrix saved: {cm_path}")
    else:
        if verbose:
            print("  TF-IDF checkpoint not found. Run training first.")
    
    # ── 2. DeBERTa ───────────────────────────────────────────────────────────
    deberta_ckpt = os.path.join(checkpoint_dir, "deberta_best")
    if os.path.exists(deberta_ckpt):
        if verbose:
            print("\nEvaluating DeBERTa...")
        try:
            sys.path.insert(0, str(Path(__file__).parent / "models"))
            from models.deberta import evaluate_from_checkpoint
            deberta_metrics, deberta_preds, deberta_probs, deberta_labels = \
                evaluate_from_checkpoint(deberta_ckpt, data_dir, "test", verbose=verbose)
            
            all_results["DeBERTa-v3"] = {
                "metrics": deberta_metrics,
                "preds": deberta_preds,
                "probs": deberta_probs,
                "labels": deberta_labels,
            }
            
            cm_path = plot_confusion_matrix(deberta_labels, deberta_preds, "DeBERTa-v3", results_dir + "/confusion_matrices")
            if verbose:
                print(f"  Confusion matrix saved: {cm_path}")
        except Exception as e:
            if verbose:
                print(f"  DeBERTa evaluation error: {e}")
    else:
        if verbose:
            print("  DeBERTa checkpoint not found. Run training first.")
    
    # ── 3. Comparison table ───────────────────────────────────────────────────
    if all_results:
        metrics_only = {k: v["metrics"] for k, v in all_results.items()}
        print_comparison_table(metrics_only)
        
        # ROC curves
        roc_data = {k: (v["labels"], v["probs"]) for k, v in all_results.items()}
        roc_path = plot_roc_curves(roc_data, results_dir)
        if verbose:
            print(f"\nROC curves saved: {roc_path}")
    
    # ── 4. Error analysis ─────────────────────────────────────────────────────
    if "TF-IDF + LR" in all_results:
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        if "combined_text" not in test_df.columns:
            from preprocess import build_combined_text
            test_df["combined_text"] = test_df.apply(build_combined_text, axis=1)
        
        tfidf_preds = all_results["TF-IDF + LR"]["preds"]
        
        # Per task_type analysis
        task_analysis = error_analysis_by_group(test_df, tfidf_preds, "task_type")
        if verbose:
            print("\n--- Error Analysis by Task Type (TF-IDF + LR) ---")
            print(task_analysis.to_string(index=False))
        
        task_analysis.to_csv(os.path.join(results_dir, "error_by_task.csv"), index=False)
        
        # Qualitative examples
        failure_examples = get_failure_examples(test_df, tfidf_preds)
        if verbose:
            print(f"\n--- Qualitative Error Examples ---")
            print(f"False Positives (clean text misclassified as injection):")
            for i, ex in enumerate(failure_examples["false_positives"][:3], 1):
                print(f"  {i}. [{ex.get('task_type', 'N/A')}] {ex.get('text', '')[:100]}...")
            
            print(f"\nFalse Negatives (injection missed by classifier):")
            for i, ex in enumerate(failure_examples["false_negatives"][:3], 1):
                print(f"  {i}. [{ex.get('task_type', 'N/A')}] {ex.get('text', '')[:100]}...")
        
        with open(os.path.join(results_dir, "failure_examples.json"), "w") as f:
            json.dump(failure_examples, f, indent=2)
    
    # Save all serializable results
    serializable = {}
    for model, data in all_results.items():
        serializable[model] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in data.items()}
    
    with open(os.path.join(results_dir, "all_results.json"), "w") as f:
        json.dump({k: v["metrics"] for k, v in all_results.items()}, f, indent=2)
    
    if verbose:
        print(f"\nAll results saved to: {results_dir}/")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluation & Error Analysis")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()
    
    run_full_evaluation(args.data_dir, args.checkpoint_dir, args.results_dir)


if __name__ == "__main__":
    main()
