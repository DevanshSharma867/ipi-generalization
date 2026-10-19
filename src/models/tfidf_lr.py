"""
tfidf_lr.py
===========
Classical Baseline: TF-IDF + Logistic Regression

This is Baseline 1 — a pre-transformer approach that:
1. Vectorizes text using character + word n-grams (TF-IDF)
2. Trains a Logistic Regression classifier

Key design choices for IPI detection:
- Sublinear TF scaling (prevents common injection phrases from dominating)
- Character n-grams (1,3) to capture obfuscated injections
- Word n-grams (1,3) for semantic patterns
- class_weight='balanced' for potential class imbalance
"""

import os
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import get_tfidf_data
from metrics import legacy_metrics
from runlog import assert_not_frozen


# ─── Hyperparameters ──────────────────────────────────────────────────────────
TFIDF_PARAMS = {
    "analyzer": "word",
    "ngram_range": (1, 3),
    "max_features": 50_000,
    "sublinear_tf": True,
    "min_df": 2,
    "strip_accents": "unicode",
    "token_pattern": r"\b\w+\b",
}

LR_PARAMS = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


def build_pipeline() -> Pipeline:
    """Build TF-IDF + LR pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
        ("lr", LogisticRegression(**LR_PARAMS)),
    ])


def train(
    data_dir: str = "data/processed",
    checkpoint_dir: str = "checkpoints",
    verbose: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Train the TF-IDF + LR baseline.
    
    Returns:
        pipeline: trained sklearn Pipeline
        val_metrics: dict of validation metrics
    """
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE 1: TF-IDF + Logistic Regression")
        print("=" * 60)
    
    # Load data
    if verbose:
        print("Loading data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        get_tfidf_data(data_dir)
    
    if verbose:
        print(f"  Train: {len(train_texts):,} samples")
        print(f"  Val:   {len(val_texts):,} samples")
        print(f"  Test:  {len(test_texts):,} samples")
    
    # Build and train pipeline
    pipeline = build_pipeline()
    
    if verbose:
        print(f"\nTF-IDF parameters:")
        for k, v in TFIDF_PARAMS.items():
            print(f"  {k}: {v}")
        print(f"\nLogistic Regression parameters:")
        for k, v in LR_PARAMS.items():
            print(f"  {k}: {v}")
        print("\nTraining...", end="", flush=True)
    
    start = time.time()
    pipeline.fit(train_texts, train_labels)
    train_time = time.time() - start
    
    if verbose:
        print(f" Done in {train_time:.1f}s")
    
    # Validate
    val_preds = pipeline.predict(val_texts)
    val_probs = pipeline.predict_proba(val_texts)[:, 1]
    
    val_metrics = compute_metrics(val_labels, val_preds, val_probs, prefix="val")
    
    if verbose:
        print("\n--- Validation Results ---")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "tfidf_lr_model.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(pipeline, f)
    if verbose:
        print(f"\nSaved model checkpoint to: {ckpt_path}")
    
    return pipeline, val_metrics


def evaluate(
    pipeline: Optional[Pipeline] = None,
    checkpoint_dir: str = "checkpoints",
    data_dir: str = "data/processed",
    split: str = "test",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained TF-IDF + LR pipeline on val or test set.
    Loads from checkpoint if pipeline is None.
    """
    if pipeline is None:
        ckpt_path = os.path.join(checkpoint_dir, "tfidf_lr_model.pkl")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train first.")
        with open(ckpt_path, "rb") as f:
            pipeline = pickle.load(f)
    
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = \
        get_tfidf_data(data_dir)
    
    texts = test_texts if split == "test" else val_texts
    labels = test_labels if split == "test" else val_labels
    
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)[:, 1]
    
    metrics = compute_metrics(labels, preds, probs, prefix=split)
    
    if verbose:
        print(f"\n--- TF-IDF + LR Results ({split.upper()}) ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\n{classification_report(labels, preds, target_names=['Clean/Benign', 'Injected'])}")
    
    return metrics, preds, probs, labels


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    prefix: str = "test",
) -> Dict[str, float]:
    """Compute all evaluation metrics (delegates to src/metrics.py)."""
    return legacy_metrics(labels, preds, probs, prefix)


def get_top_features(pipeline: Pipeline, n: int = 20) -> Dict[str, list]:
    """Extract top TF-IDF features for each class (for interpretability / error analysis)."""
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["lr"]
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]
    
    top_positive = [(feature_names[i], coefs[i]) 
                    for i in np.argsort(coefs)[-n:]][::-1]
    top_negative = [(feature_names[i], coefs[i]) 
                    for i in np.argsort(coefs)[:n]]
    
    return {
        "top_injected_features": top_positive,    # → injection class
        "top_benign_features": top_negative,       # → clean/benign class
    }


def main():
    parser = argparse.ArgumentParser(description="TF-IDF + LR Baseline")
    parser.add_argument("--mode", choices=["train", "eval", "features"], default="train")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None,
                        help="Metrics JSON (default results/v3/tfidf_lr/<mode>_results.json); frozen V2 paths are refused")
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join("results", "v3", "tfidf_lr", f"{args.mode}_results.json")
    if args.mode in ("train", "eval"):
        assert_not_frozen(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    if args.mode == "train":
        # train() writes <checkpoint_dir>/tfidf_lr_model.pkl; the V2 checkpoint is frozen
        assert_not_frozen(os.path.join(args.checkpoint_dir, "tfidf_lr_model.pkl"))

    if args.mode == "train":
        pipeline, val_metrics = train(args.data_dir, args.checkpoint_dir)
        # Also run test evaluation
        metrics, preds, probs, labels = evaluate(
            pipeline, args.checkpoint_dir, args.data_dir, "test"
        )
        all_metrics = {**val_metrics, **metrics}
        
        # Add top features
        features = get_top_features(pipeline)
        print("\n--- Top Injected-Class Features ---")
        for feat, coef in features["top_injected_features"][:10]:
            print(f"  {feat:30s}: {coef:+.4f}")
        
        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    elif args.mode == "eval":
        metrics, preds, probs, labels = evaluate(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            split=args.split,
        )
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
    
    elif args.mode == "features":
        ckpt_path = os.path.join(args.checkpoint_dir, "tfidf_lr_model.pkl")
        with open(ckpt_path, "rb") as f:
            pipeline = pickle.load(f)
        features = get_top_features(pipeline, n=30)
        print("\nTop features → Injected class:")
        for feat, coef in features["top_injected_features"]:
            print(f"  {feat:40s}: {coef:+.4f}")
        print("\nTop features → Clean/Benign class:")
        for feat, coef in features["top_benign_features"]:
            print(f"  {feat:40s}: {coef:+.4f}")


if __name__ == "__main__":
    main()
