"""
demo.py
=======
Course demo (CS F429 Part 3) for the IPI detection project: a Gradio GUI plus the older CLI views.

GUI (python demo.py --mode gui, needs requirements-demo.txt):
  - Score:   enter a user request and a document; every cached detector scores it side by side.
  - Intent pairs: the same instruction in the same document under an aligned and a subverting request
    (pilot pairs v0: LLM-drafted, not human-validated, written on V2 test docs).
  - Results: the gated paper numbers, read from paper/numbers.tex.
  - Live training: a tiny TF-IDF+LR or DeBERTa run on a few V2 documents, on the laptop CPU.

Detectors are discovered from cached checkpoints (nothing is downloaded or trained to score):
  checkpoints/tfidf_lr_model.pkl          TF-IDF+LR (frozen V2 checkpoint)
  checkpoints/v2_cluster/deberta_s*/      DeBERTa-v3-base V2, one per seed (copied from the cluster)
  checkpoints/deberta_best/               DeBERTa V2, if a checkpoint is there
  checkpoints/v3*/**/config.json          V3 classifiers: input "Intent: <q> | Context: <doc>", like V2
  checkpoints/ict/**/ict_config.json      ICT (src/models/ict.py best/ dirs), scored with ict.score
Scores are P(injected); the verdict uses each model's default threshold (0.5, or the ICT config's).

Nothing here writes to results/ or data/. Live-training runs stay in memory and are demo output, not
paper numbers. CLI modes: menu, stats, compare, tasks, samples, errors, predict, train, full.
"""

import os
import re
import sys
import json
import time
import pickle
import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from src.preprocess import build_combined_text, preprocess_for_tfidf  # noqa: E402

DEFAULT_DATA_DIR = "data/processed"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_CHECKPOINT_DIR = "checkpoints"
PAIRS_CSV = ROOT / "data/pairs/pairs_v0.csv"
NUMBERS_TEX = ROOT / "paper/numbers.tex"
THRESHOLD = 0.5

V2_NOTE = "trained on the artifact-laden V2 benchmark"
PAIRS_NOTE = ("Pilot pairs v0: LLM-drafted, not human-validated, written on V2 test documents. "
              "Illustration only, not evidence.")


# ─── Detectors ────────────────────────────────────────────────────────────────

def _hf_dirs(pattern: str, ckpt_dir: Path):
    return sorted({p.parent for p in ckpt_dir.glob(pattern) if (p.parent / "config.json").exists()})


def discover_detectors(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Every cached detector as a dict: name, kind (tfidf|hf|ict), path, note."""
    ckpt = ROOT / checkpoint_dir
    found = []
    if (ckpt / "tfidf_lr_model.pkl").exists():
        found.append(dict(name="TF-IDF+LR (V2)", kind="tfidf", path=ckpt / "tfidf_lr_model.pkl", note=V2_NOTE))
    for d in _hf_dirs("v2_cluster/deberta_*/config.json", ckpt):
        found.append(dict(name=f"DeBERTa V2 ({d.name.split('_')[-1]})", kind="hf", path=d, note=V2_NOTE))
    if (ckpt / "deberta_best/config.json").exists():
        found.append(dict(name="DeBERTa V2 (deberta_best)", kind="hf", path=ckpt / "deberta_best", note=V2_NOTE))
    for d in _hf_dirs("v3*/**/config.json", ckpt):
        found.append(dict(name=f"V3 ({d.relative_to(ckpt)})", kind="hf", path=d, note="V3 checkpoint"))
    for d in sorted(p.parent for p in ckpt.glob("ict/**/ict_config.json")):
        found.append(dict(name=f"ICT ({d.relative_to(ckpt / 'ict')})", kind="ict", path=d,
                          note="ICT checkpoint (request/context segment pair)"))
    return found


@lru_cache(maxsize=None)
def _load_tfidf(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=None)
def _load_hf(path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from src.models.deberta import get_device
    device = get_device()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device).eval()
    return tok, model, device


@lru_cache(maxsize=None)
def _load_ict(path):
    from src.models import ict
    device = ict.get_device(None)
    model, tok, cfg = ict.load_model(path, device)
    return ict.make_encoder(tok, model, cfg), model, cfg, device


def detector_threshold(det) -> float:
    return _load_ict(str(det["path"]))[2].threshold if det["kind"] == "ict" else THRESHOLD


def score_detector(det, intents, contexts):
    """P(injected) for each (intent, context) pair."""
    if det["kind"] == "ict":
        from src.models import ict
        enc, model, _, device = _load_ict(str(det["path"]))
        return 1 / (1 + np.exp(-ict.score(model, enc, list(intents), list(contexts), device)))
    if det["kind"] == "tfidf":
        texts = [build_combined_text({"user_intent": q, "text": c}) for q, c in zip(intents, contexts)]
        return _load_tfidf(str(det["path"])).predict_proba(preprocess_for_tfidf(texts))[:, 1]
    import torch
    tok, model, device = _load_hf(str(det["path"]))
    texts = [build_combined_text({"user_intent": q, "text": c}) for q, c in zip(intents, contexts)]
    enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
    return torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()


def score_text(intent: str, context: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> pd.DataFrame:
    rows = []
    for det in discover_detectors(checkpoint_dir):
        try:
            p = float(score_detector(det, [intent], [context])[0])
            rows.append({"model": det["name"], "P(injected)": round(p, 4),
                         "verdict": "INJECTED" if p >= detector_threshold(det) else "clean", "note": det["note"]})
        except Exception as e:  # one broken checkpoint must not hide the others
            rows.append({"model": det["name"], "P(injected)": None, "verdict": "error", "note": str(e)[:120]})
    return pd.DataFrame(rows, columns=["model", "P(injected)", "verdict", "note"])


# ─── Intent pairs and paper numbers ───────────────────────────────────────────

@lru_cache(maxsize=1)
def load_pairs() -> pd.DataFrame:
    """One row per pair: pair_id, task, instruction, context, aligned and subverting intent."""
    df = pd.read_csv(PAIRS_CSV)
    wide = df.pivot(index="pair_id", columns="role", values="user_intent")
    first = df.drop_duplicates("pair_id").set_index("pair_id")
    assert (df.groupby("pair_id")["text"].nunique() == 1).all(), "pair members must share the document"
    return pd.DataFrame({"task": first["task_type"], "instruction": first["instruction"], "context": first["text"],
                         "aligned": wide["aligned"], "subverting": wide["subverting"]}).reset_index()


def score_pair(pair_id: str, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR) -> pd.DataFrame:
    p = load_pairs().set_index("pair_id").loc[pair_id]
    rows = []
    for det in discover_detectors(checkpoint_dir):
        a, s = score_detector(det, [p["aligned"], p["subverting"]], [p["context"]] * 2)
        rows.append({"model": det["name"], "P(inj | aligned)": round(float(a), 4),
                     "P(inj | subverting)": round(float(s), 4),
                     "pair correct": "yes" if s > a else "no"})
    return pd.DataFrame(rows)


def load_numbers(path: Path = NUMBERS_TEX):
    """(header, DataFrame[group, macro, value]) from the generated numbers.tex."""
    if not path.exists():
        return f"{path} not found (run scripts/make_tables.py)", pd.DataFrame(columns=["group", "macro", "value"])
    lines = path.read_text().splitlines()
    header = lines[0].lstrip("% ") if lines and lines[0].startswith("%") else ""
    rows = []
    for line in lines:
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", line)
        if m:
            group = re.match(r"(VTwo|VThree|[A-Z][a-z]+)", m.group(1))
            rows.append({"group": group.group(1) if group else "", "macro": m.group(1), "value": m.group(2)})
    return header, pd.DataFrame(rows)


# ─── Live training (tiny, CPU, in memory) ─────────────────────────────────────

# DeBERTa on a laptop CPU runs ~10 s per step: a tiny run shows the loop and the loss, too short to learn much.
TINY_DEBERTA = {"model": "microsoft/deberta-v3-base", "max_length": 128, "batch_size": 8, "lr": 5e-5}
# (train docs, eval docs per set) defaults; each doc gives 4 rows
LIVE_DEFAULTS = {"tfidf": (40, 20), "deberta": (12, 8)}


def _sample_docs(df: pd.DataFrame, n_docs: int, seed: int) -> pd.DataFrame:
    """About n_docs whole documents (all their variants), task mix preserved."""
    docs = df.drop_duplicates("doc_id")[["doc_id", "task_type"]]
    keep = []
    for _, g in docs.groupby("task_type"):
        k = max(1, round(n_docs * len(g) / len(docs)))
        keep += g.sample(min(k, len(g)), random_state=seed)["doc_id"].tolist()
    return df[df["doc_id"].isin(keep)].reset_index(drop=True)


def _eval_row(name: str, df: pd.DataFrame, prob) -> dict:
    from src.metrics import point_metrics
    m = point_metrics(df["label"].values, prob, THRESHOLD)
    return {"eval set": name, "rows": m["n"], "AUROC": round(m["roc_auc"], 4),
            "TPR@1%FPR": round(m["recall_at_1pct_fpr"], 4), "FPR@0.5 (clean)": round(m["flag_rate_clean"], 4),
            "macro-F1": round(m["f1_macro"], 4)}


def live_train(kind: str = "tfidf", n_train_docs: int = None, n_eval_docs: int = None, epochs: int = 1,
               seed: int = 0, data_dir: str = DEFAULT_DATA_DIR):
    """Yields events {"log": str, "loss": (step, value) | None, "metrics": DataFrame | None}.

    Trains on a few V2 train documents and scores a few V2 val (seen attack pool) and test (held-out pool)
    documents. Demo output only: tiny samples, nothing saved, never a paper number.
    """
    from src.preprocess import load_split
    n_train_docs = n_train_docs or LIVE_DEFAULTS[kind][0]
    n_eval_docs = n_eval_docs or LIVE_DEFAULTS[kind][1]
    tr = _sample_docs(load_split(data_dir, "train"), n_train_docs, seed)
    evals = {f"V2 val, seen attacks ({n_eval_docs} docs)": _sample_docs(load_split(data_dir, "val"), n_eval_docs, seed),
             f"V2 test, held-out attacks ({n_eval_docs} docs)": _sample_docs(load_split(data_dir, "test"), n_eval_docs, seed)}
    yield {"log": f"Train: {len(tr)} rows from {tr['doc_id'].nunique()} docs "
                  f"({tr['task_type'].value_counts().to_dict()}), seed {seed}", "loss": None, "metrics": None}
    t0 = time.time()
    if kind == "tfidf":
        from src.models.tfidf_lr import build_pipeline
        pipe = build_pipeline()
        pipe.fit(preprocess_for_tfidf(tr["combined_text"].tolist()), tr["label"].values)
        yield {"log": f"TF-IDF+LR fitted in {time.time() - t0:.1f}s "
                      f"({len(pipe.named_steps['tfidf'].vocabulary_)} n-gram features)", "loss": None, "metrics": None}
        predict = lambda df: pipe.predict_proba(preprocess_for_tfidf(df["combined_text"].tolist()))[:, 1]  # noqa: E731
    else:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
        from src.models.deberta import IPIDataset, resolve_model_name_or_path, set_all_seeds
        cfg = TINY_DEBERTA
        set_all_seeds(seed)
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        path = resolve_model_name_or_path(cfg["model"])
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(path, num_labels=2, dtype=torch.float32)
        collate = DataCollatorWithPadding(tok)
        loader = DataLoader(IPIDataset(tr, tok, cfg["max_length"]), batch_size=cfg["batch_size"],
                            shuffle=True, collate_fn=collate)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
        yield {"log": f"Loaded {cfg['model']} on CPU in {time.time() - t0:.0f}s; max_length {cfg['max_length']}, "
                      f"batch {cfg['batch_size']}, lr {cfg['lr']}, {epochs} epoch(s), {len(loader)} steps/epoch",
               "loss": None, "metrics": None}
        model.train()
        step = 0
        for ep in range(1, epochs + 1):
            for batch in loader:
                loss = model(**batch).loss
                loss.backward()
                opt.step()
                opt.zero_grad()
                step += 1
                yield {"log": f"epoch {ep} step {step}: loss {loss.item():.4f} ({time.time() - t0:.0f}s)",
                       "loss": (step, loss.item()), "metrics": None}
        model.eval()

        def predict(df):
            probs = []
            dl = DataLoader(IPIDataset(df, tok, cfg["max_length"]), batch_size=16, collate_fn=collate)
            with torch.no_grad():
                for b in dl:
                    b.pop("labels")
                    probs.append(torch.softmax(model(**b).logits, dim=-1)[:, 1].numpy())
            return np.concatenate(probs)

    rows = [_eval_row(name, df, predict(df)) for name, df in evals.items()]
    yield {"log": f"Done in {time.time() - t0:.0f}s. Demo-scale numbers: a few documents, not paper results.",
           "loss": None, "metrics": pd.DataFrame(rows)}


# ─── Gradio GUI ───────────────────────────────────────────────────────────────

def build_gui(checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    import gradio as gr

    pairs = load_pairs()
    pair_choices = [(f"{r.pair_id} ({r.task}): {r.instruction[:70]}", r.pair_id) for r in pairs.itertuples()]
    p0 = pairs.iloc[0]
    dets = discover_detectors(checkpoint_dir)
    det_md = "\n".join(f"- **{d['name']}**: `{Path(d['path']).relative_to(ROOT)}` ({d['note']})" for d in dets)
    missing = [f for f in ("V3", "ICT") if not any(d["name"].startswith(f) for d in dets)]

    with gr.Blocks(title="IPI detector demo") as app:
        gr.Markdown("# Indirect prompt-injection detectors\nGiven a user request and an external document, "
                    "does the document contain an injected instruction? Scores are P(injected).")
        with gr.Tab("Score"):
            with gr.Row():
                with gr.Column():
                    intent = gr.Textbox(label="User request", value=p0["subverting"], lines=2)
                    context = gr.Textbox(label="Content (email, table, code...)", value=p0["context"], lines=12)
                    btn = gr.Button("Score", variant="primary")
                with gr.Column():
                    out = gr.Dataframe(label="Scores, side by side", interactive=False)
                    plot = gr.BarPlot(x="model", y="P(injected)", y_lim=[0, 1], label="P(injected)")
            gr.Markdown(f"Loaded detectors:\n{det_md}" +
                        (f"\n\nNot available yet: {', '.join(missing)} (no checkpoint under `{checkpoint_dir}/`)."
                         if missing else ""))

            def _score(q, c):
                df = score_text(q, c, checkpoint_dir)
                return df, df.dropna(subset=["P(injected)"])
            btn.click(_score, [intent, context], [out, plot])

        with gr.Tab("Intent pairs"):
            gr.Markdown("Same instruction, same document, two requests. Under the *aligned* request the "
                        "instruction serves the user (label clean); under the *subverting* request it does not "
                        "(label injected). A detector gets the pair right if it scores the subverting member "
                        f"higher. Chance is 0.5.\n\n*{PAIRS_NOTE}*")
            pick = gr.Dropdown(pair_choices, value=p0["pair_id"], label="Pair")
            instr = gr.Textbox(label="Embedded instruction", value=p0["instruction"], interactive=False)
            with gr.Row():
                q_a = gr.Textbox(label="Aligned request (clean)", value=p0["aligned"], interactive=False)
                q_s = gr.Textbox(label="Subverting request (injected)", value=p0["subverting"], interactive=False)
            ctx = gr.Textbox(label="Document", value=p0["context"], lines=8, interactive=False)
            pbtn = gr.Button("Score both members", variant="primary")
            pout = gr.Dataframe(interactive=False)

            def _show(pid):
                r = pairs.set_index("pair_id").loc[pid]
                return r["instruction"], r["aligned"], r["subverting"], r["context"], None
            pick.change(_show, pick, [instr, q_a, q_s, ctx, pout])
            pbtn.click(lambda pid: score_pair(pid, checkpoint_dir), pick, pout)

        with gr.Tab("Results"):
            header, nums = load_numbers()
            gr.Markdown(f"Paper numbers from `paper/numbers.tex` (only findings with status full or "
                        f"recomputed-match; provenance in `docs/findings.md`).\n\n`{header}`")
            with gr.Row():
                grp = gr.Dropdown(["all"] + sorted(nums["group"].unique().tolist()), value="all", label="Group")
                flt = gr.Textbox(label="Filter macro name", placeholder="e.g. Pair, Tpr, Auroc")
            table = gr.Dataframe(nums, interactive=False, max_height=600)

            def _filter(g, f):
                d = nums if g == "all" else nums[nums["group"] == g]
                return d[d["macro"].str.contains(f, case=False, regex=False)] if f else d
            grp.change(_filter, [grp, flt], table)
            flt.change(_filter, [grp, flt], table)

        with gr.Tab("Live training"):
            gr.Markdown("Train a detector from scratch on a few V2 documents on this laptop's CPU, then score "
                        "a few documents with seen attacks (val) and held-out attacks (test). "
                        "TF-IDF takes seconds; tiny DeBERTa (12 docs) about 2 minutes and only shows the "
                        "training loop, it is too short to learn much. Nothing is saved; these are "
                        "demo-scale numbers, not paper results.")
            with gr.Row():
                kind = gr.Radio([("TF-IDF + LR", "tfidf"), ("DeBERTa-v3-base, tiny", "deberta")],
                                value="tfidf", label="Model")
                n_tr = gr.Slider(4, 200, value=LIVE_DEFAULTS["tfidf"][0], step=4, label="Train documents (x4 rows)")
                n_ev = gr.Slider(4, 60, value=LIVE_DEFAULTS["tfidf"][1], step=2, label="Eval documents per set")
                ep = gr.Slider(1, 3, value=1, step=1, label="Epochs (DeBERTa)")
                seed = gr.Number(0, precision=0, label="Seed")
            tbtn = gr.Button("Train", variant="primary")
            kind.change(lambda k: LIVE_DEFAULTS[k], kind, [n_tr, n_ev])
            log = gr.Textbox(label="Log", lines=10, autoscroll=True)
            loss = gr.LinePlot(x="step", y="loss", label="Training loss (DeBERTa)")
            met = gr.Dataframe(label="Evaluation", interactive=False)

            def _train(k, a, b, e, s):
                lines, losses = [], []
                for ev in live_train(k, int(a), int(b), int(e), int(s)):
                    lines.append(ev["log"])
                    if ev["loss"]:
                        losses.append(ev["loss"])
                    yield ("\n".join(lines), pd.DataFrame(losses, columns=["step", "loss"]),
                           ev["metrics"] if ev["metrics"] is not None else None)
            tbtn.click(_train, [kind, n_tr, n_ev, ep, seed], [log, loss, met])
    return app


# ─── CLI views ────────────────────────────────────────────────────────────────

def banner():
    print("\n  IPI detection demo: TF-IDF+LR, DeBERTa-v3 (V2), and V3/ICT when cached\n")


def section(title: str):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def show_stats(data_dir: str = DEFAULT_DATA_DIR):
    """Mode: stats — Show dataset statistics."""
    section("DATASET STATISTICS")

    full_path = os.path.join(data_dir, "dataset.csv")
    if not os.path.exists(full_path):
        print(f"[!] Dataset not found at {full_path}. Build it with: "
              f"python src/data_pipeline.py --output_dir <new dir>  (data/processed/ is frozen)")
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
    section("ERROR ANALYSIS")

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


def predict_interactive(text: str = "", intent: str = "", checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Mode: predict — score one (intent, text) with every cached detector."""
    section("INTERACTIVE PREDICTION")
    if not text:
        text = input("Enter text sample: ").strip()
    if not intent:
        intent = input("Enter user intent (optional): ").strip()
    print(score_text(intent, text, checkpoint_dir).to_string(index=False))


def show_preset_samples(data_dir: str = DEFAULT_DATA_DIR, checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR):
    """Run live inference on curated benchmark test samples."""
    section("LIVE INFERENCE ON CURATED TEST SET EXAMPLES")
    test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        print("  Test data missing.")
        return
    df = pd.read_csv(test_path)
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
        print(f"\n  ▶ Case: {title} (ID: {s_id})")
        print(f"    Intent:  {row['user_intent'][:85]}...")
        print(f"    Context: {row['text'].replace(chr(10), ' ')[:100]}...")
        print(f"    Ground Truth: {'INJECTED (1)' if row['label'] == 1 else 'CLEAN (0)'}")
        print(score_text(row["user_intent"], row["text"], checkpoint_dir).to_string(index=False))


def run_live_train_cli(kind: str):
    section(f"LIVE TRAINING ({kind}, tiny config, CPU)")
    for ev in live_train(kind):
        print("  " + ev["log"])
        if ev["metrics"] is not None:
            print(ev["metrics"].to_string(index=False))


def run_interactive_menu(args):
    """Interactive CLI menu."""
    while True:
        banner()
        print("  [1] Dataset Statistics & Splitting Verification")
        print("  [2] Main Model Benchmark (saved V2 metrics)")
        print("  [3] Per-Task Performance Breakdown (Email, Table, Code)")
        print("  [4] Live Inference: Test Curated Benchmark Samples")
        print("  [5] Live Inference: Custom User Input (Type Live)")
        print("  [6] Qualitative Error Analysis (Top FPs & FNs)")
        print("  [7] Live Training: tiny TF-IDF+LR on a few documents")
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
            run_live_train_cli("tfidf")
        elif choice in ("0", "q", "exit"):
            print("\nExiting.\n")
            break
        else:
            print("\nInvalid choice. Please select 0 to 7.")
        input("\nPress Enter to return to menu...")


def main():
    parser = argparse.ArgumentParser(description="IPI detection demo (GUI and CLI)")
    parser.add_argument(
        "--mode",
        choices=["gui", "menu", "stats", "train", "compare", "tasks", "samples", "errors", "predict", "full"],
        default="gui",
        help="Demo mode (default: Gradio GUI)",
    )
    parser.add_argument("--model", choices=["tfidf_lr", "deberta"], default="tfidf_lr",
                        help="Model for --mode train (tiny live-training config)")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Data directory")
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR, help="Results directory (read only)")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoints directory")
    parser.add_argument("--text", type=str, default="", help="Text for --mode predict")
    parser.add_argument("--intent", type=str, default="", help="Intent for --mode predict")
    parser.add_argument("--port", type=int, default=7860, help="Port for --mode gui")
    parser.add_argument("--share", action="store_true", help="Public Gradio link for --mode gui")
    args = parser.parse_args()

    if args.mode == "gui":
        build_gui(args.checkpoint_dir).launch(server_port=args.port, share=args.share)
    elif args.mode == "menu":
        run_interactive_menu(args)
    else:
        banner()
        if args.mode == "stats":
            show_stats(args.data_dir)
        elif args.mode == "train":
            run_live_train_cli("tfidf" if args.model == "tfidf_lr" else "deberta")
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
