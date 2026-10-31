"""
ict.py
======
Intent-Counterfactual Training (ICT, paper_plan C4): a request/content cross-encoder for IPI detection.

Input: the user request and the context as a segment pair, [CLS] request [SEP] context [SEP]. The request is kept
whole and the context is truncated from the end (paper_plan §3, "Truncation"). With `windowed` (the ModernBERT-large
option) the context is instead split into overlapping windows, each with the request in front, and the example's
score is the max over its windows' scores.

Model: the backbone with a one-logit classification head; score s = logit, P(injected) = sigmoid(s).

Loss on a batch: BCE over every member, plus margin_lambda * mean hinge over pairs,
    max(0, margin - (s_subverting - s_aligned)),
where a pair is the two members of a C3 TRAIN pair (same context, aligned request = 0, subverting request = 1).

Counterfactual request-swap augmentation (`request_swap`): each epoch, every TRAIN pair gets a third member, the same
context with a request drawn from a pair on another document of the same task, labelled 1, and a second margin
term (aligned vs swapped). Assumption: an instruction that serves a request written for one document deviates from a
request written for another. Safety rules (paper_plan A22): a candidate request is excluded if it shares a content
word (pairs.content_tokens: stop words removed, stemmed) with the pair's inserted instruction, or comes from the same
source thread (the V3 near-duplicate `component` of its document; the doc itself where no component is known). The
draw is deterministic in (seed, epoch). `swap_audit` exports 50 random swapped examples in the annotation format so
the label-noise rate can be measured (A22: above 10%, the swap is reported but switched off in full ICT).

Training data: V3 train rows, C3 TRAIN pairs, and the agent-style training data (src/agent_pipeline.py rows in V3
columns, plus agent TRAIN pairs if given), each gated by its config flag.

Ablations (paper_plan A22, A23; full ICT = every component on; each removes exactly one factor):
    a  full ICT minus pairs and margin (so no swap); same data      (pre-registered; H3 comparator d; data per A23)
    b  pairs as plain CE rows, no margin (margin_lambda = 0)
    c  full ICT without request-swap augmentation
    d  full ICT without the agent-style training data
    e  (a) with the request withheld (content only)

Model selection (A22, validation only): per epoch and across the hyperparameter grid, the score is the mean of the
task-macro AUROC on V3 val and the pair accuracy on held-out TRAIN-side pairs (`--val_pairs`; never EVAL pairs).
Grid (`GRID`, full ICT, training seed 0 only): margin_lambda x margin x lr. `select` picks the best grid run (ties:
first in GRID order) and writes configs/ict_hparams.json, which every later run (seeds 0-4, all ablations) loads;
an ablation preset overrides it where it fixes the same field (e.g. (b) margin_lambda = 0).

Batch invariance: every window is scored independently with an attention mask, so scores do not depend on the
batch (tests/test_ict.py: bs=1 vs batched within 1e-5, the accepted criterion).

    python src/models/ict.py train --grid_index i --split_dir data/v3/split_seed_0 --agent_dir data/agent_v2/split_seed_0
        --train_pairs P.jsonl --val_pairs V.jsonl --seed 0                      # grid, i in 0..11
    python src/models/ict.py select                                           # -> configs/ict_hparams.json
    python src/models/ict.py train --ablation full|a|b|c|d|e --split_dir ... --agent_dir ... --train_pairs P.jsonl
        --val_pairs V.jsonl --seed s [--backbone modernbert-large]
    python src/models/ict.py score --exp ict_full --ckpt checkpoints/ict/ict_full/split_seed_0_seed0/best
        --eval test=data/v3/split_seed_0/test.csv [--eval pairs=EVAL.jsonl] --seed 0
    python src/models/ict.py swap_audit --train_pairs P.jsonl --split_dir ... [--agent_dir ...] --out A.jsonl

train writes results/v3/ict_<ablation>[_<backbone>]/<run_id>/ (config.json, val predictions, metrics) and keeps
state.pt and best/ in --work_dir; rerunning the same command resumes from state.pt. Test sets are scored only by
`score`, from the val-selected checkpoint.
"""

import argparse
import dataclasses
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from metrics import group_metrics, pair_metrics, point_metrics  # noqa: E402
from pairs import content_tokens, overlap, overlap_class  # noqa: E402

REGISTRY = ROOT / "configs" / "detectors.yaml"
WINDOWED_BY_DEFAULT = {"deberta-v3-base": False, "modernbert-large": True}
REQUEST_INPUTS = ("segment", "none")
PAIR_KEYS = ("pair_id", "doc_id", "task", "context", "instruction", "request_aligned", "request_subverting")
HPARAMS = ROOT / "configs" / "ict_hparams.json"
GRID = [{"margin_lambda": lam, "margin": m, "lr": lr}   # A22; order = tie-break order
        for lam in (0.5, 1.0, 2.0) for m in (0.5, 1.0) for lr in (1e-5, 2e-5)]
GRID_KEYS = ("margin_lambda", "margin", "lr")
SWAP_AUDIT_N = 50


@dataclass
class ICTConfig:
    backbone: str = "deberta-v3-base"   # configs/detectors.yaml name (role: ours) or a local path
    max_length: int = 512
    windowed: Optional[bool] = None      # None: the backbone's default (WINDOWED_BY_DEFAULT)
    stride: int = 128                    # tokens shared by consecutive windows
    request_input: str = "segment"       # segment | none
    use_pairs: bool = True
    use_agent_data: bool = True
    margin_lambda: float = 1.0           # margin_lambda, margin, lr: placeholders until configs/ict_hparams.json
    margin: float = 1.0                  # in logit units
    request_swap: bool = True
    lr: float = 2e-5
    epochs: int = 3
    batch_size: int = 8                  # items (a row, or a pair with its swap) per micro-batch
    grad_accum: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 0
    threshold: float = 0.5
    bf16: bool = False                   # CUDA autocast only

    def __post_init__(self):
        if self.windowed is None:
            self.windowed = WINDOWED_BY_DEFAULT.get(self.backbone, False)
        if self.request_input not in REQUEST_INPUTS:
            raise ValueError(f"request_input must be one of {REQUEST_INPUTS}")
        if not self.use_pairs and (self.margin_lambda > 0 or self.request_swap):
            raise ValueError("margin_lambda > 0 and request_swap need use_pairs")
        if self.request_input == "none" and self.use_pairs:
            raise ValueError("request_input='none' makes pair members identical inputs with opposite labels")


ABLATIONS: Dict[str, Dict] = {
    "full": {},
    "a": {"use_pairs": False, "margin_lambda": 0.0, "request_swap": False},
    "b": {"margin_lambda": 0.0},
    "c": {"request_swap": False},
    "d": {"use_agent_data": False},
    "e": {"request_input": "none", "use_pairs": False, "margin_lambda": 0.0, "request_swap": False},
}


def make_config(ablation: str = "full", hparams: Optional[Dict] = None, **overrides) -> ICTConfig:
    """Defaults <- selected hyperparameters <- ablation preset <- explicit overrides (None = not given)."""
    hp = {k: v for k, v in (hparams or {}).items() if k in GRID_KEYS}
    return ICTConfig(**{**hp, **ABLATIONS[ablation], **{k: v for k, v in overrides.items() if v is not None}})


def selection_score(metrics: Dict) -> float:
    """A22: mean of task-macro AUROC on V3 val and pair accuracy on held-out TRAIN-side val pairs."""
    return float((metrics["val"]["task_macro_auroc"] + metrics["val_pairs"]["pair_accuracy"]) / 2)


# ─── Data ────────────────────────────────────────────────────────────────────

def v3_frame(df: pd.DataFrame) -> pd.DataFrame:
    """V3 rows (data_pipeline_v3.COLUMNS) as model members: request, context, label + prediction metadata."""
    out = pd.DataFrame({
        "id": df["id"], "doc_id": df["doc_id"], "split_seed": df.get("split_seed", ""), "task": df["task_type"],
        "label": df["label"].astype(int), "request": df["user_intent"].fillna("").astype(str),
        "context": df["text"].astype(str)})
    for c in ("attack_category", "attack_source", "insertion_mode", "clean_type", "component"):
        out[c] = df[c] if c in df else ""
    return out.reset_index(drop=True)


def attach_components(pairs: List[dict], *frames: pd.DataFrame) -> List[dict]:
    """Copy each document's near-duplicate component (its source thread) from V3-format rows onto its pairs."""
    comp = {d: c for fr in frames for d, c in zip(fr["doc_id"], fr["component"]) if c not in (None, "")}
    return [{**p, "component": p.get("component") or comp.get(p["doc_id"], "")} for p in pairs]


def load_pairs(path) -> List[dict]:
    pairs = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for p in pairs:
        missing = [k for k in PAIR_KEYS if k not in p]
        if missing:
            raise ValueError(f"{path}: pair {p.get('pair_id')} lacks {missing}")
    return pairs


def pairs_frame(pairs: List[dict]) -> pd.DataFrame:
    """Both members of every pair; id = <pair_id>_<role>, so metrics.pair_keys recovers the pair."""
    return pd.DataFrame([{"id": f"{p['pair_id']}_{role}", "pair_id": p["pair_id"], "doc_id": p["doc_id"],
                          "task": p["task"], "label": y, "request": p[f"request_{role}"], "context": p["context"]}
                         for p in pairs for role, y in (("aligned", 0), ("subverting", 1))])


def load_eval_set(path) -> Tuple[str, pd.DataFrame]:
    """('rows', V3 frame) for a .csv, ('pairs', pairs frame) for a .jsonl."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return "pairs", pairs_frame(load_pairs(path))
    return "rows", v3_frame(pd.read_csv(path, keep_default_na=False))


def _group(p: dict) -> str:
    g = p.get("component")
    return str(g) if g not in (None, "") and g == g else str(p["doc_id"])


def training_data(cfg: ICTConfig, v3_rows: pd.DataFrame, v3_pairs: List[dict],
                  agent_rows: Optional[pd.DataFrame] = None, agent_pairs: Optional[List[dict]] = None):
    """(rows, pairs) for training under cfg: V3, plus the agent-style data if cfg.use_agent_data; pairs only if
    cfg.use_pairs. Pairs carry their document's component (source thread) for the swap rules."""
    frames = [v3_rows] + ([agent_rows] if cfg.use_agent_data and agent_rows is not None else [])
    pairs = list(v3_pairs) + (list(agent_pairs or []) if cfg.use_agent_data else []) if cfg.use_pairs else []
    return pd.concat(frames, ignore_index=True), attach_components(pairs, *frames)


def swap_requests(pairs: List[dict], rng: np.random.Generator, tries: int = 50) -> List[Optional[str]]:
    """For each pair, a request from a pair of the same task in another source thread (_group) that shares no
    content word with the pair's instruction; None if no request qualifies."""
    pool = defaultdict(list)
    for p in pairs:
        for r in (p["request_aligned"], p["request_subverting"]):
            pool[p["task"]].append((_group(p), r, content_tokens(r)))
    out = []
    for p in pairs:
        cands, g, instr = pool[p["task"]], _group(p), content_tokens(p["instruction"])
        ok = lambda c: c[0] != g and not (c[2] & instr)  # noqa: E731
        pick = None
        for _ in range(tries if cands else 0):
            c = cands[rng.integers(len(cands))]
            if ok(c):
                pick = c[1]
                break
        if pick is None:  # rejection sampling failed: draw from the exhaustive list (empty -> no swap)
            others = [c[1] for c in cands if ok(c)]
            pick = others[rng.integers(len(others))] if others else None
        out.append(pick)
    return out


def swap_audit(pairs: List[dict], seed: int = 0, n: int = SWAP_AUDIT_N) -> List[dict]:
    """n random swapped examples (epoch 0's draw at `seed`) as pair records in the annotation format: the original
    aligned request as `request_aligned`, the swapped request as `request_subverting` (its intended label)."""
    swaps = swap_requests(pairs, np.random.default_rng([seed, 0]))
    idx = [i for i, s in enumerate(swaps) if s is not None]
    pick = sorted(np.random.default_rng([seed, 1]).choice(idx, size=min(n, len(idx)), replace=False).tolist())
    out = []
    for k, i in enumerate(pick):
        p, sw = pairs[i], swaps[i]
        out.append({**p, "pair_id": f"swap{k:03d}", "quad_id": f"swap{k:03d}", "source": f"swap_audit:{p['source']}"
                    if p.get("source") else "swap_audit", "request_subverting": sw,
                    "overlap_aligned": overlap(p["request_aligned"], p["instruction"]),
                    "overlap_subverting": overlap(sw, p["instruction"]),
                    "overlap_class": overlap_class(p["request_aligned"], sw, p["instruction"]),
                    "notes": f"request-swap audit of {p['pair_id']}"})
    return out


def epoch_items(cfg: ICTConfig, rows: pd.DataFrame, pairs: List[dict], epoch: int) -> List[dict]:
    """Shuffled training items for one epoch. An item is {members: [(request, context, label)], pairs: [(i, j)]},
    with (i, j) = (aligned, subverting-or-swapped) member indices for the margin term."""
    rng = np.random.default_rng([cfg.seed, epoch])
    items = [{"members": [(q, c, int(y))], "pairs": []}
             for q, c, y in zip(rows["request"], rows["context"], rows["label"])]
    if cfg.use_pairs:
        swaps = swap_requests(pairs, rng) if cfg.request_swap else [None] * len(pairs)
        for p, sw in zip(pairs, swaps):
            m = [(p["request_aligned"], p["context"], 0), (p["request_subverting"], p["context"], 1)]
            if sw is not None:
                m.append((sw, p["context"], 1))
            items.append({"members": m, "pairs": [(0, j) for j in range(1, len(m))]})
    return [items[i] for i in rng.permutation(len(items))]


# ─── Encoding and scoring ────────────────────────────────────────────────────

class Encoder:
    """(request, context) -> list of windows, each {input_ids, attention_mask[, token_type_ids]}. Cached."""

    def __init__(self, tok, cfg: ICTConfig, keep_token_types: bool = False):
        self.tok, self.cfg, self.keep_tt, self.cache = tok, cfg, keep_token_types, {}

    def __call__(self, request: str, context: str) -> List[dict]:
        key = (request, context)
        if key not in self.cache:
            self.cache[key] = self._encode(request, context)
        return self.cache[key]

    def _encode(self, request, context):
        cfg = self.cfg
        stride = 0
        if cfg.windowed:  # the overlap must be below the room a window leaves for the context: cap it at half
            first = len(self.tok(request, add_special_tokens=False)["input_ids"]) \
                if cfg.request_input == "segment" else 0
            room = cfg.max_length - first - self.tok.num_special_tokens_to_add(pair=cfg.request_input == "segment")
            if room < 2:
                raise ValueError(f"request of {first} tokens leaves no room for context at max_length {cfg.max_length}")
            stride = min(cfg.stride, room // 2)
        kw = dict(max_length=cfg.max_length, return_overflowing_tokens=cfg.windowed, stride=stride)
        if cfg.request_input == "segment":
            enc = self.tok(request, context, truncation="only_second", **kw)
        else:
            enc = self.tok(context, truncation=True, **kw)
        keys = ["input_ids", "attention_mask"] + (["token_type_ids"] if self.keep_tt and "token_type_ids" in enc
                                                  else [])
        if not cfg.windowed:
            return [{k: enc[k] for k in keys}]
        return [{k: enc[k][w] for k in keys} for w in range(len(enc["input_ids"]))]


def make_encoder(tok, model, cfg: ICTConfig) -> Encoder:
    return Encoder(tok, cfg, keep_token_types=getattr(model.config, "type_vocab_size", 0) > 0)


def example_scores(model, enc: Encoder, members: List[Tuple[str, str]], device, bf16: bool = False) -> torch.Tensor:
    """One logit per (request, context): the max over that example's windows. Grad flows if enabled."""
    encs = [enc(q, c) for q, c in members]
    windows = [w for e in encs for w in e]
    batch = enc.tok.pad(windows, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
        logits = model(**batch).logits[:, 0].float()
    return torch.stack([w.max() for w in torch.split(logits, [len(e) for e in encs])])


@torch.no_grad()
def score(model, enc: Encoder, requests, contexts, device, batch_size: int = 32, bf16: bool = False) -> np.ndarray:
    model.eval()
    members = list(zip(requests, contexts))
    out = [example_scores(model, enc, members[i:i + batch_size], device, bf16).cpu()
           for i in range(0, len(members), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros(0, dtype=np.float32)


# ─── Loss and training ───────────────────────────────────────────────────────

def ict_loss(scores: torch.Tensor, labels: torch.Tensor, pairs: List[Tuple[int, int]], cfg: ICTConfig):
    """(total, ce, margin): BCE over all members + margin_lambda * mean hinge over (aligned, subverting) pairs."""
    ce = F.binary_cross_entropy_with_logits(scores, labels.float())
    if pairs and cfg.margin_lambda > 0:
        a, s = (torch.tensor(x, device=scores.device) for x in zip(*pairs))
        margin = F.relu(cfg.margin - (scores[s] - scores[a])).mean()
    else:
        margin = scores.new_zeros(())
    return ce + cfg.margin_lambda * margin, ce, margin


def batch_loss(model, enc: Encoder, items: List[dict], cfg: ICTConfig, device):
    members, labels, pairs = [], [], []
    for it in items:
        off = len(members)
        members += [(q, c) for q, c, _ in it["members"]]
        labels += [y for _, _, y in it["members"]]
        pairs += [(off + i, off + j) for i, j in it["pairs"]]
    scores = example_scores(model, enc, members, device, cfg.bf16)
    return ict_loss(scores, torch.tensor(labels, device=device), pairs, cfg)


def evaluate(model, enc, cfg, eval_sets: Dict[str, Tuple[str, pd.DataFrame]], device) -> Tuple[Dict, Dict]:
    """{set: metrics}, {set: logits}. Rows get point metrics (+ per task); pairs get pair metrics."""
    metrics, logits = {}, {}
    for name, (kind, fr) in eval_sets.items():
        s = score(model, enc, fr["request"], fr["context"], device, bf16=cfg.bf16)
        prob = 1 / (1 + np.exp(-s))
        d = fr.assign(prob=prob)
        if kind == "pairs":
            metrics[name] = pair_metrics(d)
        else:
            by_task = group_metrics(d.assign(pred=(prob >= cfg.threshold).astype(int), threshold=cfg.threshold),
                                    "task")
            metrics[name] = {**point_metrics(d["label"], prob, cfg.threshold), "by_task": by_task,
                             "task_macro_auroc": float(np.mean([m["roc_auc"] for m in by_task.values()]))}
        logits[name] = s
    return metrics, logits


def _rng_state():
    st = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st):
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_model(model, tok, cfg: ICTConfig, path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    (path / "ict_config.json").write_text(json.dumps(asdict(cfg), indent=2))


def load_model(path, device="cpu"):
    path = Path(path)
    cfg = ICTConfig(**json.loads((path / "ict_config.json").read_text()))
    model = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).to(device)
    return model, AutoTokenizer.from_pretrained(path), cfg


def init_model(cfg: ICTConfig, device="cpu"):
    """Backbone at its configs/detectors.yaml revision (or a local path) with a fresh one-logit head."""
    reg = {m["name"]: m for m in yaml.safe_load(REGISTRY.read_text())["models"] if m.get("role") == "ours"}
    repo, rev = (reg[cfg.backbone]["repo"], reg[cfg.backbone]["revision"]) if cfg.backbone in reg \
        else (cfg.backbone, None)
    torch.manual_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev, num_labels=1,
                                                               dtype=torch.float32).to(device)
    return model, tok


def train(cfg: ICTConfig, model, tok, rows: pd.DataFrame, pairs: List[dict],
          eval_sets: Dict[str, Tuple[str, pd.DataFrame]], work_dir, device, log=print,
          stop_after: Optional[int] = None, save_every: int = 200) -> Dict:
    """Train with resume. State (model, optimizer, scheduler, RNG, position) goes to work_dir/state.pt at epoch ends,
    every `save_every` optimizer steps and when `stop_after` steps are reached; rerunning resumes from it. The
    best epoch by selection_score on eval_sets ('val' rows and 'val_pairs') is saved to work_dir/best (the last epoch if eval_sets is empty)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    state_path = work_dir / "state.pt"
    enc = make_encoder(tok, model, cfg)
    n_batches = math.ceil(len(epoch_items(cfg, rows, pairs, 0)) / cfg.batch_size)
    total = math.ceil(n_batches / cfg.grad_accum) * cfg.epochs
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup_ratio), total)
    st = {"epoch": 0, "batch": 0, "step": 0, "best": None, "history": [], "losses": []}

    if state_path.exists():
        saved = torch.load(state_path, map_location=device, weights_only=False)
        if saved["cfg"] != asdict(cfg):
            raise ValueError(f"{state_path} was written with a different config; use another --work_dir")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        _set_rng_state(saved["rng"])
        st = saved["st"]
        log(f"resumed from {state_path}: epoch {st['epoch']}, batch {st['batch']}, step {st['step']}")

    def save_state():
        tmp = state_path.with_suffix(".tmp")
        torch.save({"cfg": asdict(cfg), "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": _rng_state(), "st": st}, tmp)
        os.replace(tmp, state_path)

    while st["epoch"] < cfg.epochs:
        items = epoch_items(cfg, rows, pairs, st["epoch"])
        model.train()
        opt.zero_grad()
        for b in range(st["batch"], n_batches):
            loss, ce, mg = batch_loss(model, enc, items[b * cfg.batch_size:(b + 1) * cfg.batch_size], cfg, device)
            (loss / cfg.grad_accum).backward()
            st["losses"].append([loss.item(), ce.item(), mg.item()])
            if (b + 1) % cfg.grad_accum and b != n_batches - 1:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            st["batch"], st["step"] = b + 1, st["step"] + 1
            if st["step"] % max(1, total // 20) == 0:
                recent = np.mean([x[0] for x in st["losses"][-cfg.grad_accum * 10:]])
                log(f"epoch {st['epoch']} step {st['step']}/{total} loss {recent:.4f}")
            if stop_after is not None and st["step"] >= stop_after:
                save_state()
                return st
            if save_every and st["step"] % save_every == 0:
                save_state()
        metrics, _ = evaluate(model, enc, cfg, eval_sets, device) if eval_sets else ({}, None)
        sel = selection_score(metrics) if metrics else None
        if sel is None or st["best"] is None or sel > st["best"]["value"]:
            save_model(model, tok, cfg, work_dir / "best")
            st["best"] = {"epoch": st["epoch"], "value": sel}
        st["history"].append({"epoch": st["epoch"], "train_loss": float(np.mean([x[0] for x in st["losses"]])),
                              "select": sel, "metrics": metrics})
        log(f"epoch {st['epoch']} done; selection score {sel}; best epoch {st['best']['epoch']}")
        st["epoch"], st["batch"], st["losses"] = st["epoch"] + 1, 0, []
        save_state()
    return st


# ─── CLI ─────────────────────────────────────────────────────────────────────

def predictions(fr: pd.DataFrame, logits: np.ndarray, cfg: ICTConfig, eval_set: str, model_name: str):
    prob = 1 / (1 + np.exp(-logits))
    cols = [c for c in ("id", "doc_id", "split_seed", "task", "label", "attack_category", "attack_source",
                        "insertion_mode", "clean_type") if c in fr]
    return fr[cols].assign(prob=prob, pred=(prob >= cfg.threshold).astype(int), threshold=cfg.threshold,
                           model=model_name, eval_set=eval_set)


def get_device(name: Optional[str]) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _finish_run(run, model, enc, cfg, eval_sets, device, model_name):
    metrics, logits = evaluate(model, enc, cfg, eval_sets, device)
    preds = pd.concat([predictions(eval_sets[n][1], logits[n], cfg, n, model_name) for n in eval_sets])
    run.save_predictions(preds)
    return metrics


def load_rows(split_dir: Path, name: str) -> pd.DataFrame:
    return v3_frame(pd.read_csv(Path(split_dir) / name, keep_default_na=False))


def select_grid(results_root: Path = ROOT / "results", out: Path = HPARAMS) -> Dict:
    """A22: the grid run with the highest best-epoch selection score (ties: first in GRID order) -> out (JSON).
    Needs one finished, clean run per grid point; refuses to overwrite an existing choice."""
    if out.exists():
        raise FileExistsError(f"{out} exists: the hyperparameters are already fixed")
    runs = {}
    for d in sorted((results_root / "v3" / "ict_grid").glob("*_seed0")):
        c = json.loads((d / "config.json").read_text())
        if (d / "finished.json").exists() and not c["git_dirty"]:
            runs[c["args"]["grid_index"]] = (d, json.loads((d / "metrics.json").read_text())["best"]["value"])
    missing = [i for i in range(len(GRID)) if i not in runs]
    if missing:
        raise RuntimeError(f"grid points without a finished clean run: {missing}")
    best = max(range(len(GRID)), key=lambda i: (runs[i][1], -i))
    choice = {**GRID[best], "grid_index": best, "selection_score": runs[best][1], "run": str(runs[best][0]),
              "all": {i: runs[i][1] for i in range(len(GRID))}}
    out.write_text(json.dumps(choice, indent=2) + "\n")
    return choice


def main(argv=None):
    from runlog import add_run_args, assert_not_frozen, start_run  # local: keeps the module importable in tests

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--split_dir", type=Path, required=True, help="data/v3/split_seed_<k> (train.csv, val.csv)")
    tr.add_argument("--agent_dir", type=Path, help="agent-style data for the same split (train.csv)")
    tr.add_argument("--train_pairs", type=Path, help="TRAIN pairs .jsonl on this split's train docs")
    tr.add_argument("--agent_pairs", type=Path, help="agent-style TRAIN pairs .jsonl (optional)")
    tr.add_argument("--val_pairs", type=Path, required=True,
                    help="held-out TRAIN-side pairs on val docs (eval set 'val_pairs'; never EVAL pairs)")
    tr.add_argument("--ablation", choices=list(ABLATIONS), default="full")
    tr.add_argument("--grid_index", type=int, choices=range(len(GRID)), help="grid run (full ICT, seed 0)")
    tr.add_argument("--hparams", type=Path, default=HPARAMS, help="selected hyperparameters (from `select`)")
    tr.add_argument("--work_dir", type=Path, help="state.pt and best/ (default checkpoints/ict/<exp>/<split>_seed<s>)")
    tr.add_argument("--save_every", type=int, default=200)
    tr.add_argument("--limit", type=int, help="debug: first N train rows and pairs")
    as_bool = lambda s: {"true": True, "1": True, "false": False, "0": False}[s.lower()]  # noqa: E731
    for f in dataclasses.fields(ICTConfig):  # every field can override the ablation preset; unset = preset
        typ = as_bool if f.name == "windowed" or isinstance(f.default, bool) else type(f.default)
        tr.add_argument(f"--{f.name}", type=typ)
    sc = sub.add_parser("score")
    sc.add_argument("--ckpt", type=Path, required=True, help="a best/ directory written by train")
    sc.add_argument("--eval", action="append", required=True, help="name=path (.csv V3 rows or .jsonl pairs)")
    sc.add_argument("--exp", required=True, help="the training exp of --ckpt, e.g. ict_full; writes <exp>_score")
    sc.add_argument("--seed", type=int, required=True, help="the training seed of --ckpt (run naming only)")
    sub.add_parser("select", help="pick the grid winner -> configs/ict_hparams.json")
    au = sub.add_parser("swap_audit", help="export swapped examples for annotation")
    au.add_argument("--train_pairs", type=Path, required=True)
    au.add_argument("--split_dir", type=Path, required=True, help="V3 split dir (train.csv: doc components)")
    au.add_argument("--agent_dir", type=Path)
    au.add_argument("--agent_pairs", type=Path)
    au.add_argument("--seed", type=int, default=0)
    au.add_argument("--out", type=Path, required=True)
    for p in (tr, sc):
        p.add_argument("--device")
        p.add_argument("--cuda_mem_fraction", type=float, default=0.5, help="half-card cap; MPS does not enforce it")
        add_run_args(p)
    a = ap.parse_args(argv)

    if a.cmd == "select":
        print(json.dumps(select_grid(), indent=2))
        return
    if a.cmd == "swap_audit":
        frames = [load_rows(a.split_dir, "train.csv")] + ([load_rows(a.agent_dir, "train.csv")] if a.agent_dir else [])
        pairs = load_pairs(a.train_pairs) + (load_pairs(a.agent_pairs) if a.agent_pairs else [])
        if a.out.exists():
            raise FileExistsError(a.out)
        audit = swap_audit(attach_components(pairs, *frames), a.seed)
        a.out.write_text("".join(json.dumps(r) + "\n" for r in audit))
        print(f"{len(audit)} swapped examples -> {a.out}")
        return

    device = get_device(a.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.cuda_mem_fraction)

    if a.cmd == "score":
        model, tok, cfg = load_model(a.ckpt, device)
        eval_sets = {n: load_eval_set(p) for n, p in (e.split("=", 1) for e in a.eval)}
        exp = f"{a.exp}_score"
        run = start_run("v3", exp, a.seed, vars(a), data_version="v3",
                        data_files=[p for _, p in (e.split("=", 1) for e in a.eval)], allow_dirty=a.allow_dirty)
        run.log(f"scoring {a.ckpt} on {list(eval_sets)}")
        metrics = _finish_run(run, model, make_encoder(tok, model, cfg), cfg, eval_sets, device, exp)
        run.save_metrics({"ckpt": str(a.ckpt), "ict_config": asdict(cfg), **metrics})
        run.finish()
        return

    overrides = {f.name: getattr(a, f.name) for f in dataclasses.fields(ICTConfig)}
    if a.grid_index is not None:
        if a.ablation != "full" or (a.seed or 0) != 0:
            ap.error("the grid is run on full ICT with training seed 0 only (A22)")
        hparams, exp = GRID[a.grid_index], "ict_grid"
    else:
        if not a.hparams.exists():
            ap.error(f"{a.hparams} missing: run the grid and `select` first (A22)")
        hparams = json.loads(a.hparams.read_text())
        exp = f"ict_{a.ablation}"
    cfg = make_config(a.ablation, hparams, **overrides)
    if cfg.use_pairs and not a.train_pairs:
        ap.error(f"ablation {a.ablation} uses pairs: pass --train_pairs")
    if cfg.use_agent_data and not a.agent_dir:
        ap.error(f"ablation {a.ablation} uses the agent-style training data: pass --agent_dir")
    exp += "" if cfg.backbone == "deberta-v3-base" else f"_{Path(cfg.backbone).name}"
    run_name = f"{a.split_dir.name}_seed{cfg.seed}" if a.grid_index is None \
        else f"grid{a.grid_index:02d}_{a.split_dir.name}_seed{cfg.seed}"
    work_dir = a.work_dir or ROOT / "checkpoints" / "ict" / exp / run_name
    assert_not_frozen(work_dir)
    use_agent_pairs = cfg.use_agent_data and cfg.use_pairs and a.agent_pairs
    rows, pairs = training_data(cfg, load_rows(a.split_dir, "train.csv"),
                                load_pairs(a.train_pairs) if cfg.use_pairs else [],
                                load_rows(a.agent_dir, "train.csv") if cfg.use_agent_data else None,
                                load_pairs(a.agent_pairs) if use_agent_pairs else None)
    files = [a.split_dir / "train.csv", a.split_dir / "val.csv", a.val_pairs] + \
        ([a.train_pairs] if cfg.use_pairs else []) + ([a.agent_dir / "train.csv"] if cfg.use_agent_data else []) + \
        ([a.agent_pairs] if use_agent_pairs else [])
    if a.limit:
        rows, pairs = rows.head(a.limit), pairs[:a.limit]
    eval_sets = {"val": load_eval_set(a.split_dir / "val.csv"), "val_pairs": load_eval_set(a.val_pairs)}
    run = start_run("v3", exp, cfg.seed, {**vars(a), "ict_config": asdict(cfg), "hparams": hparams,
                                          "work_dir": str(work_dir)},
                    data_version=f"v3-{a.split_dir.name}", data_files=files, allow_dirty=a.allow_dirty)
    run.log(f"{exp}: {len(rows)} rows, {len(pairs)} pairs, device {device}, work_dir {work_dir}")
    model, tok = init_model(cfg, device)
    st = train(cfg, model, tok, rows, pairs, eval_sets, work_dir, device, log=run.log, save_every=a.save_every)
    model, tok, cfg = load_model(work_dir / "best", device)
    metrics = _finish_run(run, model, make_encoder(tok, model, cfg), cfg, eval_sets, device, exp)
    run.save_metrics({"best": st["best"], "history": st["history"], **metrics},
                     headline=("val_selection_score", selection_score(metrics)))
    run.finish()


if __name__ == "__main__":
    main()
