"""
deberta.py
==========
Modern Baseline: DeBERTa-v3-base Fine-tuned for IPI Detection

DeBERTa-v3-base uses:
  - Disentangled attention (content + position separately)
  - Enhanced mask decoder
  - ELECTRA-style pre-training on DeBERTa-v3

Patched for V3:
  - --data_dir default data/processed
  - --seed arg (import-time set_seed removed)
  - Dynamic padding (DataCollatorWithPadding) instead of fixed padding="max_length"
  - Batch size 8 with gradient accumulation steps = 2 (effective batch size 16)
  - Saves per-split predictions CSV (id, doc_id, task_type, attack_type, label, prob_injected, pred)
  - --output_dir argument; quick/demo runs route to checkpoints/deberta_demo, never deberta_best
  - --train_tasks and --test_task options for Leave-One-Task-Out (LOTO) cross-validation
"""

import os
from pathlib import Path

# Configure local HF cache and offline mode if present
local_hf_cache = Path(__file__).parent.parent.parent / "hf_cache"
if local_hf_cache.exists():
    os.environ["HF_HOME"] = str(local_hf_cache.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_OFFLINE = True
    except Exception:
        pass

import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocess import get_deberta_data, build_combined_text


# ─── Default Hyperparameters ──────────────────────────────────────────────────
DEBERTA_PARAMS = {
    "model_name": "microsoft/deberta-v3-base",
    "max_length": 512,
    "learning_rate": 2e-5,
    "batch_size_train": 8,
    "grad_accum_steps": 2,
    "batch_size_eval": 16,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "num_labels": 2,
    "dropout": 0.1,
}


def set_all_seeds(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    set_seed(seed)


class IPIDataset(Dataset):
    """PyTorch Dataset for IPI binary classification with dynamic padding."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.labels = self.df["label"].values
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Build combined texts
        if "combined_text" in self.df.columns:
            self.texts = self.df["combined_text"].tolist()
        else:
            self.texts = self.df.apply(build_combined_text, axis=1).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Truncate to max_length without padding (DataCollatorWithPadding pads dynamically)
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": int(self.labels[idx]),
        }


def get_device() -> torch.device:
    """Get best available compute device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def compute_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    prefix: str = "test",
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    has_both_classes = len(np.unique(labels)) > 1
    roc_auc = float(roc_auc_score(labels, probs)) if has_both_classes else 0.0

    return {
        f"{prefix}_accuracy": float(accuracy_score(labels, preds)),
        f"{prefix}_f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        f"{prefix}_f1_injected": float(f1_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        f"{prefix}_precision": float(precision_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, preds, pos_label=1, average="binary", zero_division=0)),
        f"{prefix}_roc_auc": roc_auc,
    }


def save_split_predictions_csv(
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    output_path: str,
):
    """
    Save per-split predictions CSV with schema:
    (id, doc_id, task_type, attack_type, label, prob_injected, pred)
    """
    out_df = pd.DataFrame()
    cols = ["id", "doc_id", "task_type", "attack_type", "label"]
    for col in cols:
        if col in df.columns:
            out_df[col] = df[col].values
        else:
            out_df[col] = "None"
    out_df["prob_injected"] = probs
    out_df["pred"] = preds
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved predictions CSV ({len(out_df)} rows) to: {output_path}")


def evaluate_model(
    model,
    dataloader: DataLoader,
    device: torch.device,
    prefix: str = "val",
    verbose: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation and return metrics + predictions."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            probs = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits.float(), dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels_batch.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_labels, all_preds, all_probs, prefix)

    if verbose:
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"\n{classification_report(all_labels, all_preds, target_names=['Clean/Benign', 'Injected'], zero_division=0)}")

    return metrics, all_preds, all_probs, all_labels


def resolve_model_name_or_path(name: str) -> str:
    if os.path.exists(name):
        return name
    candidate = Path(__file__).parent.parent.parent / "hf_cache/hub/models--microsoft--deberta-v3-base/snapshots/8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
    if name == "microsoft/deberta-v3-base" and candidate.exists():
        return str(candidate.resolve())
    return name


def train(
    data_dir: str = "data/processed",
    output_dir: str = "checkpoints/deberta_best",
    model_name: str = "microsoft/deberta-v3-base",
    num_epochs: int = 3,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
    train_tasks: Optional[List[str]] = None,
    test_task: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Any, Any, Dict[str, float]]:
    """Fine-tune DeBERTa for IPI binary classification with dynamic padding and grad accumulation."""
    set_all_seeds(seed)

    model_path = resolve_model_name_or_path(model_name)

    if verbose:
        print("\n" + "=" * 60)
        print("DeBERTa-v3-base Fine-tuning")
        print("=" * 60)
        print(f"Model: {model_name} (Resolved: {model_path}) | Seed: {seed}")
        print(f"Epochs: {num_epochs}, Micro-batch: {batch_size}, Grad Accum: {grad_accum_steps} (Effective batch: {batch_size * grad_accum_steps})")
        print(f"Learning rate: {learning_rate}, Output dir: {output_dir}")

    device = get_device()
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    if verbose:
        print(f"\nLoading tokenizer and model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = model.to(device)

    # Load data
    if verbose:
        print(f"Loading datasets from {data_dir}...")
    train_df, val_df, test_df = get_deberta_data(data_dir)

    # Apply LOTO task filtering if specified
    if test_task:
        if not train_tasks:
            all_tasks = ["email", "table", "code"]
            train_tasks = [t for t in all_tasks if t != test_task]
        if verbose:
            print(f"LOTO Mode: Training tasks = {train_tasks}, Held-out test task = '{test_task}'")
        train_df = train_df[train_df["task_type"].isin(train_tasks)].reset_index(drop=True)
        val_df = val_df[val_df["task_type"].isin(train_tasks)].reset_index(drop=True)
        test_df = test_df[test_df["task_type"] == test_task].reset_index(drop=True)

    if verbose:
        print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Datasets and dynamic padding collator
    train_dataset = IPIDataset(train_df, tokenizer, max_length)
    val_dataset = IPIDataset(val_df, tokenizer, max_length)
    test_dataset = IPIDataset(test_df, tokenizer, max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, collate_fn=collator, num_workers=0)

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    effective_updates_per_epoch = int(np.ceil(len(train_loader) / grad_accum_steps))
    total_steps = effective_updates_per_epoch * num_epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Loss function with class weights
    pos_weight = (train_df["label"] == 0).sum() / max(1, (train_df["label"] == 1).sum())
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, float(pos_weight)], dtype=torch.float32).to(device))

    best_val_f1 = -1.0
    best_metrics = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        start = time.time()

        if verbose:
            print(f"\nEpoch {epoch}/{num_epochs}")

        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits.float(), labels)
            loss = loss / grad_accum_steps
            loss.backward()

            if (step % grad_accum_steps == 0) or (step == len(train_loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum_steps

            if verbose and step % max(1, len(train_loader) // 5) == 0:
                print(f"  Step {step:4d}/{len(train_loader)} | "
                      f"Loss: {epoch_loss/step:.4f} | "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")

        epoch_time = time.time() - start
        avg_loss = epoch_loss / len(train_loader)

        if verbose:
            print(f"  Epoch {epoch} complete in {epoch_time:.0f}s | Avg Loss: {avg_loss:.4f}")
            print("  Validating...")

        val_metrics, val_preds, val_probs, _ = evaluate_model(
            model, val_loader, device, prefix="val", verbose=verbose
        )

        # Save checkpoint if better
        if val_metrics["val_f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["val_f1_macro"]
            best_metrics = val_metrics
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_split_predictions_csv(val_df, val_preds, val_probs, os.path.join(output_dir, "val_predictions.csv"))
            if verbose:
                print(f"  ✓ New best val F1 macro: {best_val_f1:.4f} → Saved to {output_dir}")

    # Evaluate best model on test set
    if verbose:
        print(f"\nLoading best checkpoint from {output_dir} for test evaluation...")
    best_model = AutoModelForSequenceClassification.from_pretrained(output_dir).to(device)
    test_metrics, test_preds, test_probs, _ = evaluate_model(
        best_model, test_loader, device, prefix="test", verbose=verbose
    )
    save_split_predictions_csv(test_df, test_preds, test_probs, os.path.join(output_dir, "test_predictions.csv"))

    all_metrics = {**best_metrics, **test_metrics}
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    return best_model, tokenizer, all_metrics


def evaluate_from_checkpoint(
    checkpoint_dir: str = "checkpoints/deberta_best",
    data_dir: str = "data/processed",
    split: str = "test",
    test_task: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Load from checkpoint and evaluate on specified split."""
    device = get_device()

    if verbose:
        print(f"\nLoading DeBERTa checkpoint from: {checkpoint_dir}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model = model.to(device)

    train_df, val_df, test_df = get_deberta_data(data_dir)
    eval_df = test_df if split == "test" else val_df
    if test_task:
        eval_df = eval_df[eval_df["task_type"] == test_task].reset_index(drop=True)

    eval_dataset = IPIDataset(eval_df, tokenizer, max_length=512)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collator, num_workers=0)

    if verbose:
        print(f"\n--- DeBERTa Results ({split.upper()}) ---")

    metrics, preds, probs, labels = evaluate_model(model, eval_loader, device, prefix=split, verbose=verbose)
    save_split_predictions_csv(eval_df, preds, probs, os.path.join(checkpoint_dir, f"{split}_predictions.csv"))
    return metrics, preds, probs, labels


def predict_single(
    text: str,
    user_intent: str = "",
    checkpoint_dir: str = "checkpoints/deberta_best",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Predict on a single text sample."""
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    model = model.to(device)
    model.eval()

    combined = f"Intent: {user_intent} | Context: {text}" if user_intent else text

    encoding = tokenizer(combined, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    result = {
        "label": pred,
        "label_str": "INJECTED" if pred == 1 else "CLEAN",
        "prob_clean": float(probs[0]),
        "prob_injected": float(probs[1]),
        "confidence": float(max(probs)),
    }

    if verbose:
        print(f"\n--- DeBERTa Prediction ---")
        print(f"Text: {combined[:100]}...")
        print(f"Prediction: {result['label_str']}")
        print(f"P(clean):    {result['prob_clean']:.4f}")
        print(f"P(injected): {result['prob_injected']:.4f}")
        print(f"Confidence:  {result['confidence']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="DeBERTa Fine-tuning for IPI Detection (V3)")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], default="train")
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base",
                        help="HuggingFace model name")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for checkpoint and predictions")
    parser.add_argument("--checkpoint_dir", default=None,
                        help="Legacy alias for --output_dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size (default 8)")
    parser.add_argument("--grad_accum_steps", type=int, default=2, help="Gradient accumulation steps (default 2)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--split", default="test")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    parser.add_argument("--train_tasks", nargs="+", default=None, help="Tasks to train on for LOTO")
    parser.add_argument("--test_task", type=str, default=None, help="Held-out task for LOTO")
    parser.add_argument("--text", type=str, default="", help="Text for --mode predict")
    parser.add_argument("--intent", type=str, default="", help="User intent for --mode predict")
    parser.add_argument("--output", default=None, help="Path to save summary JSON")
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        out_dir = args.output_dir
    elif args.checkpoint_dir is not None:
        out_dir = args.checkpoint_dir
    elif args.quick:
        out_dir = "checkpoints/deberta_demo"
    elif args.test_task:
        out_dir = f"checkpoints/deberta_loto_{args.test_task}"
    else:
        out_dir = "checkpoints/deberta_best"

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    if args.mode == "train":
        epochs = 1 if args.quick else args.epochs
        model_name = "microsoft/deberta-v3-small" if args.quick else args.model_name

        model, tokenizer, metrics = train(
            data_dir=args.data_dir,
            output_dir=out_dir,
            model_name=model_name,
            num_epochs=epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            learning_rate=args.lr,
            seed=args.seed,
            train_tasks=args.train_tasks,
            test_task=args.test_task,
        )

        out_json = args.output or os.path.join("results", f"deberta_seed{args.seed}_results.json" if not args.test_task else f"deberta_loto_{args.test_task}_results.json")
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nSaved metrics to: {out_json}")

    elif args.mode == "eval":
        metrics, _, _, _ = evaluate_from_checkpoint(
            checkpoint_dir=out_dir,
            data_dir=args.data_dir,
            split=args.split,
            test_task=args.test_task,
        )
        out_json = args.output or os.path.join(out_dir, "eval_metrics.json")
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)

    elif args.mode == "predict":
        if not args.text:
            args.text = input("Enter text to classify: ")
        predict_single(args.text, args.intent, checkpoint_dir=out_dir)


if __name__ == "__main__":
    main()
