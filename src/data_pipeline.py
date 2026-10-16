"""
data_pipeline.py
===================
Leakage-free data pipeline:
1. Document-level splitting (by doc_id) with 0 doc_id and 0 exact-text overlap across splits.
2. Held-out attacks: train/val docs use *_attack_train.json, test docs use *_attack_test.json.
3. Exact 1:1 balance (clean == injected) per task AND per split.
4. Benign insertion uses the EXACT same insert_attack() logic, modes (start/middle/end), and separators, with NO 'Note:' prefix.
5. Disjoint benign templates: benign_train for train/val, benign_test for test from data/benign/benign_instructions.json.
6. Caps docs per task via --max_docs_per_task (default 300).
7. Comprehensive assertions and statistics reporting.
"""

import os
import json
import random
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_jsonl(filepath: Path) -> List[Dict[str, Any]]:
    items = []
    if not filepath.exists():
        return items
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    items.append(json.loads(line.strip()))
                except Exception:
                    pass
    return items


def insert_sentence(context: str, sentence: str, mode: str = "end") -> str:
    """
    Unified function for inserting both attacks and benign imperatives.
    Uses identical separators and positioning modes.
    """
    sentence = sentence.strip()
    if mode == "start":
        return f"{sentence}\n\n{context}"
    elif mode == "middle":
        words = context.split()
        if len(words) > 10:
            mid = len(words) // 2
            return " ".join(words[:mid]) + f"\n\n{sentence}\n\n" + " ".join(words[mid:])
        return f"{context}\n\n{sentence}"
    else:  # end
        return f"{context}\n\n{sentence}"


def load_attacks(benchmark_dir: Path) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, List[Dict[str, str]]]]:
    """
    Loads train and test attacks separately.
    Returns:
      train_attacks: {"text": [...], "code": [...]}
      test_attacks:  {"text": [...], "code": [...]}
    Each item is {"attack": str, "type": str, "source": str}
    """
    train_attacks = {"text": [], "code": []}
    test_attacks = {"text": [], "code": []}

    # Text attacks train
    p_tat = benchmark_dir / "text_attack_train.json"
    if p_tat.exists():
        with open(p_tat, "r", encoding="utf-8") as f:
            for cat, atks in json.load(f).items():
                for a in atks:
                    train_attacks["text"].append({"attack": a, "type": cat, "source": "text_attack_train.json"})

    # Text attacks test
    p_tate = benchmark_dir / "text_attack_test.json"
    if p_tate.exists():
        with open(p_tate, "r", encoding="utf-8") as f:
            for cat, atks in json.load(f).items():
                for a in atks:
                    test_attacks["text"].append({"attack": a, "type": cat, "source": "text_attack_test.json"})

    # Code attacks train
    p_cat = benchmark_dir / "code_attack_train.json"
    if p_cat.exists():
        with open(p_cat, "r", encoding="utf-8") as f:
            for cat, atks in json.load(f).items():
                for a in atks:
                    train_attacks["code"].append({"attack": a, "type": cat, "source": "code_attack_train.json"})

    # Code attacks test
    p_cate = benchmark_dir / "code_attack_test.json"
    if p_cate.exists():
        with open(p_cate, "r", encoding="utf-8") as f:
            for cat, atks in json.load(f).items():
                for a in atks:
                    test_attacks["code"].append({"attack": a, "type": cat, "source": "code_attack_test.json"})

    return train_attacks, test_attacks


def build_dataset(
    bipia_dir: str = "data/raw/BIPIA",
    benign_json: str = "data/benign/benign_instructions.json",
    max_docs_per_task: int = 300,
    variants_per_doc: int = 4,  # generates 2 clean and 2 injected per doc => exact 1:1 balance
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    benchmark_dir = Path(bipia_dir) / "benchmark"
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"BIPIA benchmark not found at {benchmark_dir}")

    # Load benign instructions
    with open(benign_json, "r", encoding="utf-8") as f:
        benign_data = json.load(f)
    benign_train = benign_data["benign_train"]
    benign_test = benign_data["benign_test"]
    assert len(set(benign_train) & set(benign_test)) == 0, "Benign train and test sets must be disjoint!"

    # Load attack sets
    train_attacks, test_attacks = load_attacks(benchmark_dir)

    # 1. Collect unique base documents per task
    tasks = ["email", "code", "table"]
    docs_by_task = {}

    for task in tasks:
        task_dir = benchmark_dir / task
        raw_items = load_jsonl(task_dir / "train.jsonl") + load_jsonl(task_dir / "test.jsonl")
        
        # Deduplicate raw contexts within task
        seen_contexts = set()
        unique_docs = []
        for item in raw_items:
            ctx = item.get("context", "")
            if isinstance(ctx, list):
                ctx = "\n".join(str(x) for x in ctx)
            elif isinstance(ctx, dict):
                ctx = json.dumps(ctx)
            else:
                ctx = str(ctx)
            ctx = ctx.strip()
            if not ctx or ctx in seen_contexts:
                continue
            seen_contexts.add(ctx)

            intent = item.get("question", item.get("user_intent", f"Perform {task} QA"))
            if isinstance(intent, list):
                intent = " ".join(str(x) for x in intent)
            else:
                intent = str(intent)

            unique_docs.append({
                "context": ctx,
                "user_intent": intent.strip(),
                "task_type": task,
            })

        # Cap docs per task
        if max_docs_per_task and len(unique_docs) > max_docs_per_task:
            random.shuffle(unique_docs)
            unique_docs = unique_docs[:max_docs_per_task]

        docs_by_task[task] = unique_docs
        print(f"Task '{task}': {len(unique_docs)} unique base documents selected (capped at {max_docs_per_task}).")

    # 2. Split base documents 70 / 15 / 15 by task (Group splitting at document level)
    doc_splits = {"train": [], "val": [], "test": []}
    global_doc_id = 0

    for task, doc_list in docs_by_task.items():
        # Assign unique doc_id
        for doc in doc_list:
            doc["doc_id"] = f"doc_{global_doc_id:04d}"
            global_doc_id += 1

        n_docs = len(doc_list)
        # Train / val / test split: 70% / 15% / 15%
        # For e.g. 100 docs: 70 train, 15 val, 15 test
        n_test = max(1, int(round(n_docs * 0.15)))
        n_val = max(1, int(round(n_docs * 0.15)))
        n_train = n_docs - n_test - n_val

        # Deterministic shuffle
        random.Random(seed).shuffle(doc_list)
        train_slice = doc_list[:n_train]
        val_slice = doc_list[n_train:n_train + n_val]
        test_slice = doc_list[n_train + n_val:]

        for d in train_slice:
            d["split"] = "train"
            doc_splits["train"].append(d)
        for d in val_slice:
            d["split"] = "val"
            doc_splits["val"].append(d)
        for d in test_slice:
            d["split"] = "test"
            doc_splits["test"].append(d)

    # 3. Generate samples per document with exact 1:1 balance
    # For each doc, generate K clean and K injected samples (K = variants_per_doc // 2)
    # E.g., K = 2 => 2 clean (1 raw clean, 1 benign variant) and 2 injected (2 attack variants)
    K = max(1, variants_per_doc // 2)
    modes = ["start", "middle", "end"]

    all_records = []
    sample_id = 0

    for split_name in ["train", "val", "test"]:
        is_test = (split_name == "test")
        benign_pool = benign_test if is_test else benign_train
        attack_source_pool = test_attacks if is_test else train_attacks

        for doc in doc_splits[split_name]:
            task = doc["task_type"]
            ctx = doc["context"]
            intent = doc["user_intent"]
            doc_id = doc["doc_id"]
            atk_pool = attack_source_pool["code"] if task == "code" else attack_source_pool["text"]

            # --- Clean samples (exactly K) ---
            # 1. Original clean context (no insertion)
            all_records.append({
                "id": f"sample_{sample_id:05d}",
                "doc_id": doc_id,
                "text": ctx,
                "user_intent": intent,
                "label": 0,
                "task_type": task,
                "attack_type": "none",
                "attack_source": "none",
                "insertion_mode": "none",
                "split": split_name,
            })
            sample_id += 1

            # 2..K: Benign imperative variants (using SAME insert_sentence function & modes)
            for k in range(1, K):
                benign_sent = random.choice(benign_pool)
                mode = modes[k % len(modes)]
                benign_text = insert_sentence(ctx, benign_sent, mode=mode)
                all_records.append({
                    "id": f"sample_{sample_id:05d}",
                    "doc_id": doc_id,
                    "text": benign_text,
                    "user_intent": intent,
                    "label": 0,
                    "task_type": task,
                    "attack_type": "benign_imperative",
                    "attack_source": "benign_instructions.json",
                    "insertion_mode": mode,
                    "split": split_name,
                })
                sample_id += 1

            # --- Injected samples (exactly K) ---
            # Sample K distinct attacks if available
            chosen_atks = random.sample(atk_pool, min(K, len(atk_pool))) if len(atk_pool) >= K else [random.choice(atk_pool) for _ in range(K)]
            for k in range(K):
                atk_obj = chosen_atks[k]
                mode = modes[k % len(modes)]
                inj_text = insert_sentence(ctx, atk_obj["attack"], mode=mode)
                all_records.append({
                    "id": f"sample_{sample_id:05d}",
                    "doc_id": doc_id,
                    "text": inj_text,
                    "user_intent": intent,
                    "label": 1,
                    "task_type": task,
                    "attack_type": atk_obj["type"],
                    "attack_source": atk_obj["source"],
                    "insertion_mode": mode,
                    "split": split_name,
                })
                sample_id += 1

    df_all = pd.DataFrame(all_records)
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val = df_all[df_all["split"] == "val"].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    # 4. Strict sanity assertions
    print("\n" + "=" * 60)
    print("RUNNING DATASET SANITY CHECKS & ASSERTIONS")
    print("=" * 60)

    # Assert 0 doc_id overlap
    train_docs = set(df_train["doc_id"])
    val_docs = set(df_val["doc_id"])
    test_docs = set(df_test["doc_id"])
    assert len(train_docs & val_docs) == 0, f"Overlap in doc_id between train & val: {train_docs & val_docs}"
    assert len(train_docs & test_docs) == 0, f"Overlap in doc_id between train & test: {train_docs & test_docs}"
    assert len(val_docs & test_docs) == 0, f"Overlap in doc_id between val & test: {val_docs & test_docs}"
    print("✓ ASSERTION PASSED: 0 doc_id overlap across train, val, and test splits.")

    # Assert 0 exact text overlap across splits
    train_texts = set(df_train["text"])
    val_texts = set(df_val["text"])
    test_texts = set(df_test["text"])
    assert len(train_texts & val_texts) == 0, f"Exact text overlap train & val: {len(train_texts & val_texts)}"
    assert len(train_texts & test_texts) == 0, f"Exact text overlap train & test: {len(train_texts & test_texts)}"
    assert len(val_texts & test_texts) == 0, f"Exact text overlap val & test: {len(val_texts & test_texts)}"
    print("✓ ASSERTION PASSED: 0 exact-text overlap across train, val, and test splits.")

    # Assert exact 1:1 balance per task AND per split
    for split_name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        for task in tasks:
            sub = split_df[split_df["task_type"] == task]
            n_clean = (sub["label"] == 0).sum()
            n_inj = (sub["label"] == 1).sum()
            assert n_clean == n_inj, f"Imbalance in {split_name} for task {task}: {n_clean} clean vs {n_inj} injected"
    print("✓ ASSERTION PASSED: Exact 1:1 balance (clean == injected) within EVERY task and EVERY split.")

    # Assert held-out attack source separation
    assert df_test[df_test["label"] == 1]["attack_source"].str.contains("test.json").all(), "Test set contains non-test attack sources!"
    assert not df_train[df_train["label"] == 1]["attack_source"].str.contains("test.json").any(), "Train set contains test attack sources!"
    assert not df_val[df_val["label"] == 1]["attack_source"].str.contains("test.json").any(), "Val set contains test attack sources!"
    print("✓ ASSERTION PASSED: Attack sources are strictly held out (train/val use *_train.json, test uses *_test.json).")

    return df_all, df_train, df_val, df_test


def print_stats(df_all: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DATASET DETAILED STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(df_all):,}")
    print(f"Total unique documents: {df_all['doc_id'].nunique():,}")

    print("\n--- Document Counts per Split ---")
    for s in ["train", "val", "test"]:
        sub = df_all[df_all["split"] == s]
        print(f"  {s:6s}: {sub['doc_id'].nunique():4d} docs | {len(sub):5d} samples")

    print("\n--- Split x Task x Label Counts ---")
    ct = pd.crosstab([df_all["split"], df_all["task_type"]], df_all["label"])
    ct.columns = ["Clean (0)", "Injected (1)"]
    print(ct.to_string())

    print("\n--- Attack Categories per Split (Injected Samples) ---")
    for s in ["train", "val", "test"]:
        sub = df_all[(df_all["split"] == s) & (df_all["label"] == 1)]
        cats = sorted(sub["attack_type"].unique())
        print(f"\n[{s.upper()}] {len(cats)} unique attack categories:")
        for c in cats:
            cnt = (sub["attack_type"] == c).sum()
            print(f"    - {c:45s}: {cnt:3d} samples")


def main():
    parser = argparse.ArgumentParser(description="BIPIA Data Pipeline")
    parser.add_argument("--bipia_dir", default="data/raw/BIPIA")
    parser.add_argument("--benign_json", default="data/benign/benign_instructions.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--max_docs_per_task", type=int, default=300)
    parser.add_argument("--variants_per_doc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Building binary IPI dataset (leakage-free, held-out attacks)")
    df_all, df_train, df_val, df_test = build_dataset(
        bipia_dir=args.bipia_dir,
        benign_json=args.benign_json,
        max_docs_per_task=args.max_docs_per_task,
        variants_per_doc=args.variants_per_doc,
        seed=args.seed,
    )
    print_stats(df_all)

    os.makedirs(args.output_dir, exist_ok=True)
    df_all.to_csv(os.path.join(args.output_dir, "dataset.csv"), index=False)
    df_train.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    df_val.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    df_test.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
    print(f"\nSaved dataset to {args.output_dir}/")


if __name__ == "__main__":
    main()
