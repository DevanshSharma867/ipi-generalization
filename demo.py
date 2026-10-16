"""
demo.py
=======
Phase 2 CLI Demo Interface
===================================================
Features:
  1. Dataset statistics (from data/processed)
  2. Classical Baseline evaluation (TF-IDF + LR)
  3. DeBERTa-v3 fine-tuning / evaluation
  4. Model comparison table (dynamic metric deltas, no hardcoded claims)
  5. Error analysis (reads results/test_predictions.csv and error examples)
  6. Interactive single-sample prediction

Usage:
  python demo.py --mode stats                          # Show dataset stats
  python demo.py --mode train --model tfidf_lr         # Train/run TF-IDF baseline
  python demo.py --mode train --model deberta --quick  # Train DeBERTa demo
  python demo.py --mode compare                        # Side-by-side comparison
  python demo.py --mode errors                         # Error analysis
  python demo.py --mode predict --text "..."           # Single prediction
"""

import os
import sys
import json
import time
import pickle
import argparse
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DEFAULT_DATA_DIR = "data/processed"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_CHECKPOINT_DIR = "checkpoints"


def banner():
    msg = """
╔══════════════════════════════════════════════════════════════╗
║   CS F429 NLP — Phase 2 Demo                                 ║
║   Indirect Prompt Injection Detection on BIPIA               ║
║   TF-IDF + Logistic Regression  vs  DeBERTa-v3-base           ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(msg)


def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def show_stats(data_dir: str = DEFAULT_DATA_DIR):
    """Mode: stats — Show dataset statistics from V2 data without calling V1 pipeline."""
    section("DATASET STATISTICS (V2 PROCESSED)")

    full_path = os.path.join(data_dir, "dataset.csv")
    if not os.path.exists(full_path):
        print(f"[!] Dataset not found at {full_path}")
        print("    Rebuilding with V2 pipeline: python src/data_pipeline.py")
        ret = os.system("python3 src/data_pipeline.py")
        if ret != 0:
            print("Failed to run data_pipeline.py")
            return

    df = pd.read_csv(full_path)

    print(f"\n  Dataset Location:   {full_path}")
    print(f"  Total Samples:      {len(df):,}")
    print(f"  Total Documents:    {df['doc_id'].nunique() if 'doc_id' in df.columns else 'N/A'}")
    print(f"  Injected (1):       {df['label'].sum():,}  ({df['label'].mean()*100:.1f}%)")
    print(f"  Clean/Benign (0):   {(df['label']==0).sum():,}  ({(1-df['label'].mean())*100:.1f}%)")

    print(f"\n  Split Distribution:")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        docs = sub["doc_id"].nunique() if "doc_id" in sub.columns else len(sub) // 4
        inj = sub["label"].sum()
        cln = (sub["label"] == 0).sum()
        print(f"    {split:6s}: {len(sub):5,} samples | {docs:4d} unique docs (Clean={cln}, Inj={inj})")

    print(f"\n  Task Distribution:")
    for task in sorted(df["task_type"].unique()):
        sub = df[df["task_type"] == task]
        docs = sub["doc_id"].nunique() if "doc_id" in sub.columns else len(sub) // 4
        print(f"    {task:10s}: {len(sub):5,} samples | {docs:4d} base docs")

    print(f"\n  Sample Preview:")
    inj_sample = df[df["label"] == 1].iloc[0]
    clean_sample = df[df["label"] == 0].iloc[0]
    print(f"    [INJECTED] Task={inj_sample['task_type']} | Atk={inj_sample.get('attack_type', 'N/A')}")
    print(f"               {str(inj_sample.get('text', ''))[:110]}...")
    print(f"    [CLEAN]    Task={clean_sample['task_type']}")
    print(f"               {str(clean_sample.get('text', ''))[:110]}...")


def train_tfidf(data_dir: str = DEFAULT_DATA_DIR, results_dir: str = DEFAULT_RESULTS_DIR):
    """Run TF-IDF experiments."""
    section("TF-IDF + LOGISTIC REGRESSION (DETERMINISTIC EVALUATION)")
    ret = os.system(f"python3 run_experiments.py")
    if ret != 0:
        print("Error executing run_experiments.py")


def train_deberta(
    data_dir: str = DEFAULT_DATA_DIR,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    quick: bool = False,
):
    """Run DeBERTa training."""
    section("DeBERTa-v3-base FINE-TUNING")
    out_dir = "checkpoints/deberta_demo" if quick else os.path.join(checkpoint_dir, "deberta_best")
    flag_quick = "--quick" if quick else ""
    cmd = f"python3 src/models/deberta.py --mode train --data_dir {data_dir} --output_dir {out_dir} {flag_quick}"
    print(f"Executing: {cmd}")
    os.system(cmd)


def show_comparison(results_dir: str = DEFAULT_RESULTS_DIR, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Mode: compare — Read saved metrics from results and compute deltas dynamically."""
    section("MODEL COMPARISON (SAVED EVALUATION METRICS)")

    tfidf_path = os.path.join(results_dir, "tfidf_lr_results.json")
    deberta_json_candidates = [
        os.path.join(results_dir, "deberta_results.json"),
        os.path.join(results_dir, "deberta_seed42_results.json"),
        os.path.join(checkpoint_dir, "deberta_best", "metrics.json"),
        os.path.join("checkpoints/deberta_demo", "metrics.json"),
    ]

    loaded_models = {}

    # Load TF-IDF metrics
    if os.path.exists(tfidf_path):
        with open(tfidf_path) as f:
            t_data = json.load(f)
        def_m = t_data.get("default_evaluation", {}).get("metrics", {})
        tun_m = t_data.get("tuned_evaluation", {}).get("metrics", {})
        if def_m:
            loaded_models["TF-IDF+LR (Default C=1)"] = {
                "Acc": def_m.get("accuracy", 0.0),
                "F1-Mac": def_m.get("f1_macro", 0.0),
                "F1-Inj": def_m.get("f1_injected", 0.0),
                "Prec": def_m.get("precision", 0.0),
                "Rec": def_m.get("recall", 0.0),
                "AUC": def_m.get("roc_auc", 0.0),
            }
        if tun_m:
            loaded_models["TF-IDF+LR (Tuned)"] = {
                "Acc": tun_m.get("accuracy", 0.0),
                "F1-Mac": tun_m.get("f1_macro", 0.0),
                "F1-Inj": tun_m.get("f1_injected", 0.0),
                "Prec": tun_m.get("precision", 0.0),
                "Rec": tun_m.get("recall", 0.0),
                "AUC": tun_m.get("roc_auc", 0.0),
            }

    # Load DeBERTa metrics
    deb_m = None
    for cand in deberta_json_candidates:
        if os.path.exists(cand):
            with open(cand) as f:
                d_cand = json.load(f)
            deb_m = d_cand
            break

    if deb_m:
        loaded_models["DeBERTa-v3-base"] = {
            "Acc": deb_m.get("test_accuracy", 0.0),
            "F1-Mac": deb_m.get("test_f1_macro", 0.0),
            "F1-Inj": deb_m.get("test_f1_injected", 0.0),
            "Prec": deb_m.get("test_precision", 0.0),
            "Rec": deb_m.get("test_recall", 0.0),
            "AUC": deb_m.get("test_roc_auc", 0.0),
        }

    if not loaded_models:
        print("  No evaluation results found in results/. Run experiments first.")
        return

    print(f"\n  {'Model':<26} {'Acc':>8} {'F1-Mac':>8} {'F1-Inj':>8} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print(f"  {'-'*76}")

    for m_name, m in loaded_models.items():
        print(f"  {m_name:<26} {m['Acc']:>8.4f} {m['F1-Mac']:>8.4f} {m['F1-Inj']:>8.4f} {m['Prec']:>8.4f} {m['Rec']:>8.4f} {m['AUC']:>8.4f}")

    print(f"  {'-'*76}")

    # Compute delta dynamically if both DeBERTa and TF-IDF are present
    if "DeBERTa-v3-base" in loaded_models and "TF-IDF+LR (Default C=1)" in loaded_models:
        d = loaded_models["DeBERTa-v3-base"]
        t = loaded_models["TF-IDF+LR (Default C=1)"]
        delta_acc = d["Acc"] - t["Acc"]
        delta_f1m = d["F1-Mac"] - t["F1-Mac"]
        delta_f1i = d["F1-Inj"] - t["F1-Inj"]
        delta_pre = d["Prec"] - t["Prec"]
        delta_rec = d["Rec"] - t["Rec"]
        delta_auc = d["AUC"] - t["AUC"]
        print(f"  {'Delta (DeBERTa - TF-IDF)':<26} {delta_acc:>+8.4f} {delta_f1m:>+8.4f} {delta_f1i:>+8.4f} {delta_pre:>+8.4f} {delta_rec:>+8.4f} {delta_auc:>+8.4f}")
        print(f"  {'-'*76}")


def show_error_analysis(results_dir: str = DEFAULT_RESULTS_DIR):
    """Mode: errors — Read real predictions and errors from results."""
    section("ERROR ANALYSIS (RESULTS/V2 PREDICTIONS)")

    pred_path = os.path.join(results_dir, "test_predictions.csv")
    examples_path = os.path.join(results_dir, "error_examples_20fp_20fn.json")

    if not os.path.exists(pred_path):
        print(f"  Prediction file not found at {pred_path}. Run run_experiments.py first.")
        return

    df = pd.read_csv(pred_path)
    pred_col = "pred" if "pred" in df.columns else "pred_default"

    print("\n  Per-Task Error Breakdown (Test Set Predictions):")
    print(f"  {'Task':<12} {'N':>6} {'Accuracy':>10} {'Macro-F1':>10} {'FP':>6} {'FN':>6}")
    print(f"  {'-'*56}")

    for task in sorted(df["task_type"].unique()):
        sub = df[df["task_type"] == task]
        y_true = sub["label"].values
        y_pred = sub[pred_col].values
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        print(f"  {task:<12} {len(sub):>6d} {acc:>10.4f} {f1m:>10.4f} {fp:>6d} {fn:>6d}")

    total_fp = int(((df["label"] == 0) & (df[pred_col] == 1)).sum())
    total_fn = int(((df["label"] == 1) & (df[pred_col] == 0)).sum())
    print(f"  {'-'*56}")
    print(f"  {'Overall':<12} {len(df):>6d} {accuracy_score(df['label'], df[pred_col]):>10.4f} "
          f"{f1_score(df['label'], df[pred_col], average='macro', zero_division=0):>10.4f} {total_fp:>6d} {total_fn:>6d}")

    if os.path.exists(examples_path):
        with open(examples_path) as f:
            ex_data = json.load(f)
        fps = ex_data.get("false_positives_top20", [])
        fns = ex_data.get("false_negatives_top20", [])
        print(f"\n  Top False Positives ({len(fps)} logged):")
        for i, ex in enumerate(fps[:3], 1):
            print(f"    {i}. Doc: {ex.get('doc_id')} | Task: {ex.get('task_type')} | Prob(Inj): {ex.get('prob_injected', 0):.4f}")
        print(f"\n  Top False Negatives ({len(fns)} logged):")
        for i, ex in enumerate(fns[:3], 1):
            print(f"    {i}. Doc: {ex.get('doc_id')} | Task: {ex.get('task_type')} | Attack: {ex.get('attack_type')} | Prob(Inj): {ex.get('prob_injected', 0):.4f}")


def predict_interactive(text: str = "", intent: str = "", checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Mode: predict — Single text prediction."""
    section("INTERACTIVE PREDICTION")

    if not text:
        text = input("Enter text sample: ").strip()
    if not intent:
        intent = input("Enter user intent (optional): ").strip()

    combined = f"Intent: {intent} | Context: {text}" if intent else text

    # Check classical baseline
    ckpt_tfidf = "checkpoints/tfidf_lr_model.pkl"
    if os.path.exists(ckpt_tfidf):
        with open(ckpt_tfidf, "rb") as f:
            pipeline = pickle.load(f)
        from src.preprocess import preprocess_for_tfidf
        clean = preprocess_for_tfidf([combined])[0]
        prob = pipeline.predict_proba([clean])[0]
        pred = int(prob[1] >= 0.5)
        print(f"\n  [TF-IDF + LR]  Pred: {'INJECTED ⚠️' if pred == 1 else 'CLEAN ✓'} | P(clean): {prob[0]:.4f} | P(inj): {prob[1]:.4f}")

    # Check DeBERTa
    deb_best = os.path.join(checkpoint_dir, "deberta_best")
    if os.path.exists(deb_best):
        try:
            from src.models.deberta import predict_single
            res = predict_single(text, intent, checkpoint_dir=deb_best, verbose=False)
            print(f"  [DeBERTa-v3]   Pred: {'INJECTED ⚠️' if res['label'] == 1 else 'CLEAN ✓'} | P(clean): {res['prob_clean']:.4f} | P(inj): {res['prob_injected']:.4f}")
        except Exception as e:
            print(f"  DeBERTa inference error: {e}")


def show_per_task(results_dir: str = DEFAULT_RESULTS_DIR):
    """Mode: tasks — Show per-task breakdown with confidence intervals."""
    section("PER-TASK BREAKDOWN (DEFAULT SETTINGS, HELD-OUT ATTACKS)")
    res_path = os.path.join(results_dir, "tfidf_lr_results.json")
    if not os.path.exists(res_path):
        print("  Results not found. Run experiments first.")
        return

    with open(res_path) as f:
        data = json.load(f)
    pt = data.get("per_task_breakdown", {})

    print(f"\n  {'Task Type':<12} {'N':>6} {'Accuracy [95% CI]':>22} {'Macro-F1 [95% CI]':>22} {'Precision':>10} {'Recall':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*96}")
    task_counts = {"email": 48, "table": 180, "code": 60}
    for task, t_data in pt.items():
        m = t_data.get("metrics", {})
        ci = t_data.get("cluster_bootstrap_95ci", {})
        acc_ci = f"{m.get('accuracy',0):.4f} [{ci.get('accuracy',[0,0])[0]:.2f}, {ci.get('accuracy',[0,0])[1]:.2f}]"
        f1_ci = f"{m.get('f1_macro',0):.4f} [{ci.get('f1_macro',[0,0])[0]:.2f}, {ci.get('f1_macro',[0,0])[1]:.2f}]"
        print(f"  {task.upper():<12} {task_counts.get(task, 0):>6d} {acc_ci:>22} {f1_ci:>22} {m.get('precision',0):>10.4f} {m.get('recall',0):>10.4f} {m.get('roc_auc',0):>10.4f}")
    print(f"  {'-'*96}")


def show_preset_samples(data_dir: str = DEFAULT_DATA_DIR, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Run live inference on curated benchmark test samples."""
    section("LIVE INFERENCE ON CURATED TEST SET EXAMPLES")
    test_path = os.path.join(data_dir, "test.csv")
    ckpt_tfidf = "checkpoints/tfidf_lr_model.pkl"
    if not os.path.exists(test_path) or not os.path.exists(ckpt_tfidf):
        print("  Data or model checkpoint missing.")
        return

    df = pd.read_csv(test_path)
    with open(ckpt_tfidf, "rb") as f:
        pipe = pickle.load(f)
    from src.preprocess import preprocess_for_tfidf, build_combined_text

    cases = [
        ("Clean Email Notification", "sample_01624"),
        ("Injected Email (iPhone Prompt Hijack)", "sample_01626"),
        ("Clean Code Snippet", "sample_01672"),
        ("Injected Code (Malicious Shell Payload)", "sample_01674"),
    ]

    for title, s_id in cases:
        match = df[df["id"] == s_id]
        if match.empty:
            continue
        row = match.iloc[0]
        comb = build_combined_text(row)
        cl = preprocess_for_tfidf([comb])[0]
        prob = pipe.predict_proba([cl])[0]
        pred = int(prob[1] >= 0.5)

        print(f"\n  ▶ Case: {title} (ID: {s_id})")
        print(f"    Intent:  {row['user_intent'][:85]}...")
        snippet = row['text'].replace('\n', ' ')[:100]
        print(f"    Context: {snippet}...")
        print(f"    Ground Truth: {'INJECTED (1)' if row['label'] == 1 else 'CLEAN (0)'}")
        tag = "INJECTED ⚠️" if pred == 1 else "CLEAN ✓"
        print(f"    Model Pred:   {tag} | P(clean): {prob[0]:.4f} | P(inj): {prob[1]:.4f}")


def run_interactive_menu(args):
    """Interactive CLI menu for Part 2 evaluation & viva voce."""
    while True:
        banner()
        print("  [1] Dataset Statistics & Splitting Verification")
        print("  [2] Main Model Benchmark (Default vs Tuned Settings)")
        print("  [3] Per-Task Performance Breakdown (Email, Table, Code)")
        print("  [4] Live Inference: Test Curated Benchmark Samples")
        print("  [5] Live Inference: Custom User Input (Type Live)")
        print("  [6] Qualitative Error Analysis (Top FPs & FNs)")
        print("  [7] Re-Run Evaluation Pipeline (Live Deterministic Benchmark)")
        print("  [0] Exit")
        print()
        choice = input("  Select an option [0-7]: ").strip()

        if choice == "1":
            show_stats(args.data_dir)
        elif choice == "2":
            show_comparison(args.results_dir, args.checkpoint_dir)
        elif choice == "3":
            show_per_task(args.results_dir)
        elif choice == "4":
            show_preset_samples(args.data_dir, args.checkpoint_dir)
        elif choice == "5":
            predict_interactive("", "", args.checkpoint_dir)
        elif choice == "6":
            show_error_analysis(args.results_dir)
        elif choice == "7":
            train_tfidf(args.data_dir, args.results_dir)
        elif choice in ("0", "q", "exit"):
            print("\nExiting demo. Good luck with the viva!\n")
            break
        else:
            print("\nInvalid choice. Please select 0 to 7.")
        input("\nPress Enter to return to menu...")


def main():
    parser = argparse.ArgumentParser(description="CS F429 NLP Phase 2 Demo")
    parser.add_argument(
        "--mode",
        choices=["menu", "stats", "train", "compare", "tasks", "samples", "errors", "predict", "full"],
        default="menu",
        help="Demo mode to execute (default: interactive menu)",
    )
    parser.add_argument("--model", choices=["tfidf_lr", "deberta"], default="tfidf_lr")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Results directory")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoints directory")
    parser.add_argument("--text", type=str, default="", help="Text for --mode predict")
    parser.add_argument("--intent", type=str, default="", help="Intent for --mode predict")
    args = parser.parse_args()

    if args.mode == "menu":
        run_interactive_menu(args)
    else:
        banner()
        if args.mode == "stats":
            show_stats(args.data_dir)
        elif args.mode == "train":
            if args.model == "tfidf_lr":
                train_tfidf(args.data_dir, args.results_dir)
            else:
                train_deberta(args.data_dir, args.checkpoint_dir, quick=args.quick)
        elif args.mode == "compare":
            show_comparison(args.results_dir, args.checkpoint_dir)
        elif args.mode == "tasks":
            show_per_task(args.results_dir)
        elif args.mode == "samples":
            show_preset_samples(args.data_dir, args.checkpoint_dir)
        elif args.mode == "errors":
            show_error_analysis(args.results_dir)
        elif args.mode == "predict":
            predict_interactive(args.text, args.intent, args.checkpoint_dir)
        elif args.mode == "full":
            show_stats(args.data_dir)
            show_comparison(args.results_dir, args.checkpoint_dir)
            show_per_task(args.results_dir)
            show_preset_samples(args.data_dir, args.checkpoint_dir)
            show_error_analysis(args.results_dir)


if __name__ == "__main__":
    main()
