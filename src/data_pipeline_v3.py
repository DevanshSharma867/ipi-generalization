"""
data_pipeline_v3.py
===================
V3 builder (docs/v3_design.md, decisions D1-D7). V2 (src/data_pipeline.py, data/processed/) is untouched.

Every main-set row has exactly ONE inserted span; per doc 2 clean + 2 injected:
  injected x2  two different attack strings from the doc's split pool (train / val categories / test file)
  clean_a      a benign human-addressed item of the doc's register (benign JSON, split-disjoint pools)
  clean_b      spliced from another doc of the same task and split: email = a neutral body sentence,
               code = a prose line + fenced block, table = a sentence rendered from a donor row (D2)
Position (start/middle/end) and separator ("\\n\\n"/"\\n") are drawn i.i.d. per row, independent of the label.
Middle uses the nearest boundary of the first available kind: line, paragraph, email field, sentence (D1).
Code and table docs that do not fit 512 DeBERTa-v3 tokens are trimmed to a window before insertion (D3); email
docs that do not fit, and docs with no internal boundary, go to the eval-only `long` slice.
Near-duplicate docs (word-5-gram Jaccard >= 0.5, within task) are kept in one split (D4).

Eval-only slices: val_id (val docs x train-category attacks), raw_clean, ambiguous, long.

Randomness: one random.Random per stage, seeded from (split_seed, stage); trimming windows are seeded per doc
only, so they are the same for every split_seed. Same seed -> byte-identical output.

    python src/data_pipeline_v3.py --split_seed 0 --benign_json data/benign/benign_v3.json --output_dir data/v3
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import build_combined_text
from probes import (FORM_BINARY, FORM_NUMERIC, V3_BOUNDARIES, NoBoundary, code_chunks, email_sentences,
                    form_tags, has_internal_boundary, insert_preserving)
from runlog import assert_not_frozen

BENCH = ROOT / "data" / "raw" / "BIPIA" / "benchmark"
TOKENIZER = ROOT / "hf_cache" / "tokenizers_hpc" / "deberta_v2_ckpt"
TASKS = ("email", "table", "code")
ATTACK_REGISTER = {"email": "text", "table": "text", "code": "code"}
MAX_TOKENS = 512
MARGIN = 6                      # slack for separators and tokenization at the boundary; every row is re-checked
SPLIT_FRAC = 0.15
JACCARD = 0.5
POSITIONS = ("start", "middle", "end")
SEPARATORS = ("\n\n", "\n")
VAL_CATEGORIES = {"text": ("Instruction", "Programming Help", "Social Interaction"),            # D7
                  "code": ("Exploiting System Vulnerabilities", "Sending Out Spam Emails")}
BENIGN_VAL_SHARE = 0.25
DONOR_K = {"email": 2, "code": 2, "table": 1}
TABLE_DONOR_DOC_MAX = 3
COLUMNS = ["id", "doc_id", "split_seed", "task_type", "split", "eval_slice", "label", "row_type", "user_intent",
           "text", "span_text", "span_source", "attack_category", "attack_source", "clean_type", "insertion_mode",
           "separator", "boundary_kind", "char_offset", "span_len", "trimmed", "component", "n_tokens"]

# D2: >= 20 templates; {r} row label, {h} column header, {v} cell value
TABLE_TEMPLATES = [
    "You can see that {h} for {r} is {v}.",
    "Your copy lists {v} under {h} for {r}.",
    "For {r}, the {h} entry reads {v}.",
    "Did you notice that {r} has {v} as its {h}?",
    "The {h} column lists {v} next to {r}.",
    "The row for {r} gives {v} as the {h}.",
    "As you can see, {r} is listed with {h} {v}.",
    'In your version, {h} for {r} shows "{v}".',
    "Were you aware that {r}'s {h} is {v}?",
    "You may recall that {r} records {v} for {h}.",
    "{r}'s {h} is given here as {v}.",
    "You might expect a different {h} than {v} for {r}.",
    "Is the {h} for {r} really {v}, as you remember it?",
    "Here you have {v} recorded as the {h} of {r}.",
    'The {h} you see for {r} is "{v}".',
    "This table puts {r} at {v} for {h}.",
    "Your earlier summary also had {v} as the {h} for {r}.",
    "You can compare {r} with the others by its {h}, {v}.",
    "Does your copy also show {v} as the {h} for {r}?",
    "The {h} column holds {v} for {r}, which you checked last time.",
    "For the {r} line, the {h} is {v}.",
    "{r} appears with {v} in the {h} field.",
    "You probably know {r} already, and its {h} is {v}.",
    "If you look at {r}, the {h} is {v}.",
]


def rng_for(split_seed: int, stage: str) -> random.Random:
    return random.Random(f"v3:{split_seed}:{stage}")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ─── docs, intents, near-duplicates ───────────────────────────────────────────

def _norm_context(ctx) -> str:
    """V2 normalization (data_pipeline.build_dataset)."""
    if isinstance(ctx, list):
        ctx = "\n".join(str(x) for x in ctx)
    elif isinstance(ctx, dict):
        ctx = json.dumps(ctx)
    return str(ctx).strip()


def _last_line(lines) -> str:
    lines = [str(l).strip() for l in lines if str(l).strip()]
    return lines[-1] if lines else ""


def intent_for(item: dict, task: str) -> str:
    if task == "code":
        return f"Fix this error in my code: {_last_line(item['error'])}"
    return str(item["question"]).strip()


def load_docs(doc_limit: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        seen, n = set(), 0
        for split in ("train", "test"):
            for line in open(BENCH / task / f"{split}.jsonl", encoding="utf-8"):
                item = json.loads(line)
                ctx = _norm_context(item.get("context", ""))
                if not ctx or ctx in seen or (doc_limit is not None and n >= doc_limit):
                    continue
                seen.add(ctx)
                rows.append({"doc_id": f"{task}_{n:04d}", "task_type": task, "context": ctx,
                             "user_intent": intent_for(item, task), "ideal": str(item.get("ideal", "")).strip()})
                n += 1
    return pd.DataFrame(rows)


def _shingles(text: str, n: int = 5) -> frozenset:
    w = text.lower().split()
    return frozenset(tuple(w[i:i + n]) for i in range(max(1, len(w) - n + 1)))


def near_dup_components(docs: pd.DataFrame) -> Dict[str, str]:
    """doc_id -> component id (smallest doc_id in its connected component of Jaccard >= JACCARD, per task)."""
    parent = {d: d for d in docs["doc_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, g in docs.groupby("task_type"):
        ids, sh = list(g["doc_id"]), [_shingles(c) for c in g["context"]]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = sh[i], sh[j]
                if min(len(a), len(b)) < JACCARD * max(len(a), len(b)):
                    continue
                if len(a & b) / len(a | b) >= JACCARD:
                    ra, rb = find(ids[i]), find(ids[j])
                    parent[max(ra, rb)] = min(ra, rb)
    return {d: find(d) for d in parent}


def assign_splits(docs: pd.DataFrame, comp: Dict[str, str], split_seed: int) -> Dict[str, str]:
    """Per task: shuffle components, fill test then val to round(0.15 n) docs each, the rest is train."""
    rng = rng_for(split_seed, "split")
    out = {}
    for task in TASKS:
        g = docs[docs["task_type"] == task]
        members = defaultdict(list)
        for d in g["doc_id"]:
            members[comp[d]].append(d)
        order = sorted(members)
        rng.shuffle(order)
        target = max(1, round(SPLIT_FRAC * len(g)))
        filled = {"test": 0, "val": 0}
        for c in order:
            split = next((s for s in ("test", "val") if filled[s] < target), "train")
            if split != "train":
                filled[split] += len(members[c])
            for d in members[c]:
                out[d] = split
    return out


# ─── length control (D3) ──────────────────────────────────────────────────────

class Fitter:
    def __init__(self, tok, span_cap: Dict[str, int]):
        self.tok, self.span_cap = tok, span_cap

    def n_tokens(self, intent: str, text: str) -> int:
        return len(self.tok(build_combined_text({"user_intent": intent, "text": text}))["input_ids"])

    def fits(self, intent: str, ctx: str, task: str) -> bool:
        return self.n_tokens(intent, ctx) + self.span_cap[ATTACK_REGISTER[task]] + MARGIN <= MAX_TOKENS


def _grow(units: List[str], seed_range, fits, rng: random.Random, joiner="\n"):
    """Grow a contiguous window [a, b) of units from seed_range, one unit at a time on a random side."""
    a, b = seed_range
    if not fits(joiner.join(units[a:b])):
        return None
    while True:
        sides = [s for s in ("up", "down") if (s == "up" and a > 0) or (s == "down" and b < len(units))]
        rng.shuffle(sides)
        for s in sides:
            na, nb = (a - 1, b) if s == "up" else (a, b + 1)
            if fits(joiner.join(units[na:nb])):
                a, b = na, nb
                break
        else:
            return a, b


def trim_table(doc: dict, fitter: Fitter) -> dict:
    lines = doc["context"].split("\n")
    header, rows = lines[0], lines[1:]
    rng = random.Random(f"v3:trim:{doc['doc_id']}")
    fits = lambda body: fitter.fits(doc["user_intent"], header + "\n" + body, "table")
    ideal = doc["ideal"].lower()
    hits = [i for i, r in enumerate(rows) if ideal and ideal != "unknown" and ideal in r.lower()]
    status, win = "no_answer_in_table", None
    if hits:
        win = _grow(rows, (min(hits), max(hits) + 1), fits, rng)
        status = "all_answer_rows_kept"
        if win is None:
            for i in sorted(hits, key=lambda _: rng.random()):
                win = _grow(rows, (i, i + 1), fits, rng)
                if win:
                    status = "some_answer_rows_kept"
                    break
        if win is None:
            status = "answer_row_not_kept"
    if win is None:
        start = rng.randrange(len(rows))
        win = _grow(rows, (start, start + 1), fits, rng)
    if win is None:
        return {"context": None, "d3": "untrimmable"}
    return {"context": header + "\n" + "\n".join(rows[win[0]:win[1]]), "d3": status}


def code_units(ctx: str) -> List[str]:
    """Lines outside fences, each fenced block (fence lines included) as one atomic unit."""
    units, cur, in_fence = [], [], False
    for line in ctx.split("\n"):
        fence = line.strip().startswith("```")
        if in_fence or fence:
            cur.append(line)
            if fence and in_fence:
                units.append("\n".join(cur))
                cur = []
            in_fence = not in_fence if fence else in_fence
        else:
            units.append(line)
    if cur:
        units.append("\n".join(cur))
    return units


def trim_code(doc: dict, fitter: Fitter) -> dict:
    units = code_units(doc["context"])
    rng = random.Random(f"v3:trim:{doc['doc_id']}")
    fits = lambda body: fitter.fits(doc["user_intent"], body.strip(), "code")
    for start in sorted(range(len(units)), key=lambda _: rng.random()):
        win = _grow(units, (start, start + 1), fits, rng)
        if win:
            return {"context": "\n".join(units[win[0]:win[1]]).strip(), "d3": "trimmed"}
    return {"context": None, "d3": "untrimmable"}


def prepare_docs(docs: pd.DataFrame, fitter: Fitter) -> pd.DataFrame:
    """Adds main_context (None -> long slice), trimmed, d3 status and long_reason."""
    out = []
    for doc in docs.to_dict("records"):
        ctx, trimmed, d3, reason = doc["context"], False, "fits", ""
        if not fitter.fits(doc["user_intent"], ctx, doc["task_type"]):
            if doc["task_type"] == "email":
                ctx, d3, reason = None, "filtered", "does_not_fit"
            else:
                r = (trim_table if doc["task_type"] == "table" else trim_code)(doc, fitter)
                ctx, d3, trimmed = r["context"], r["d3"], r["context"] is not None
                reason = "" if ctx else "untrimmable"
        if ctx is not None and not has_internal_boundary(ctx):
            ctx, reason = None, "no_internal_boundary"
        out.append({**doc, "main_context": ctx, "trimmed": trimmed, "d3": d3, "long_reason": reason})
    return pd.DataFrame(out)


# ─── pools ────────────────────────────────────────────────────────────────────

def load_attack_pools():
    pools = {}
    for reg in ("text", "code"):
        tr = json.load(open(BENCH / f"{reg}_attack_train.json"))
        te = json.load(open(BENCH / f"{reg}_attack_test.json"))
        assert set(VAL_CATEGORIES[reg]) <= set(tr)
        item = lambda c, a, f: {"text": a.strip(), "category": c, "source": f}
        pools[reg] = {
            "train": [item(c, a, f"{reg}_attack_train.json") for c in sorted(tr) if c not in VAL_CATEGORIES[reg]
                      for a in tr[c]],
            "val": [item(c, a, f"{reg}_attack_train.json") for c in sorted(tr) if c in VAL_CATEGORIES[reg] for a in tr[c]],
            "test": [item(c, a, f"{reg}_attack_test.json") for c in sorted(te) for a in te[c]],
            "train_file": [a.strip() for c in sorted(tr) for a in tr[c]]}
    return pools


def span_cost(tok, span: str) -> int:
    return len(tok("\n\n" + span.strip() + "\n\n", add_special_tokens=False)["input_ids"])


def load_benign(path: Path, tok, span_cap) -> dict:
    """{'train'|'val'|'test'|'ambiguous': {register: [item]}}, items over the register's span cap dropped."""
    items = json.load(open(path))["items"]
    rng = random.Random("v3:benign_partition")
    pools, dropped = {s: defaultdict(list) for s in ("train", "val", "test", "ambiguous")}, []
    for it in sorted(items, key=lambda x: x["id"]):
        cap = span_cap[ATTACK_REGISTER[it["register"]]]
        if span_cost(tok, it["text"]) > cap:
            dropped.append(it["id"])
            continue
        if it["split_pool"] == "train_val":
            pools["train"][it["register"]].append(it)
        else:
            pools[it["split_pool"]][it["register"]].append(it)
    for reg in list(pools["train"]):
        tv = pools["train"][reg]
        rng.shuffle(tv)
        n_val = max(1, round(BENIGN_VAL_SHARE * len(tv)))
        pools["val"][reg], pools["train"][reg] = sorted(tv[:n_val], key=lambda x: x["id"]), \
            sorted(tv[n_val:], key=lambda x: x["id"])
    return {"pools": pools, "dropped_over_cap": dropped}


def _cells(line: str) -> List[str]:
    return [c.replace("\xa0", " ").strip() for c in line.strip().strip("|").split("|")]


TABLE_TRIPLES = 6


def render_table_sentences(ctx: str, rng: random.Random) -> List[str]:
    """D2: up to TABLE_TRIPLES (row label, header, cell) triples of the donor (digit-free first), each rendered
    with every template; clean-B selection then picks one. Headers are lower-cased unless all caps."""
    lines = [l for l in ctx.split("\n") if l.strip().startswith("|")]
    if len(lines) < 2:
        return []
    header = _cells(lines[0])
    opts = []
    for line in lines[1:]:
        row = _cells(line)
        if len(row) != len(header) or not row[0]:
            continue
        for j in range(1, len(row)):
            h, v = header[j], row[j]
            if not h or not v or v in ("-", "—", "–") or len(v.split()) > 4 or len(h.split()) > 4 \
                    or len(row[0].split()) > 4 or "|" in v:
                continue
            opts.append((any(ch.isdigit() for ch in h + v + row[0]), row[0], h, v))
    if not opts:
        return []
    rng.shuffle(opts)
    opts.sort(key=lambda o: o[0])
    out = []
    for _, r, h, v in opts[:TABLE_TRIPLES]:
        h = h if h.isupper() else h.lower()
        out += [(lambda x: x[0].upper() + x[1:])(t.format(r=r, h=h, v=v)) for t in TABLE_TEMPLATES]
    return out


def donor_candidates(task: str, ctx: str, tok, cap: int, rng: random.Random) -> List[str]:
    if task == "email":
        c = email_sentences(ctx)
    elif task == "code":
        c = code_chunks(ctx)
    else:
        c = render_table_sentences(ctx, rng)
    out = []
    for s in c:
        if s not in out and span_cost(tok, s) <= cap and len(s.split()) <= (17 if task == "table" else 10 ** 6):
            out.append(s)
    return out


# ─── clean-B selection: profile-aware ──────────────────────────────────────────

MATCH_FEATURES = ("has_digit", "has_you", "ends_q", "has_quote")


MATCH_WEIGHT = {"has_digit": 2.0, "has_you": 1.0, "ends_q": 1.0, "has_quote": 1.0}


def attack_rates() -> Dict[str, dict]:
    """Per attack register: train-file rate of each MATCH_FEATURE and the (n_words, n_chars) shape of every string."""
    pools = load_attack_pools()
    out = {}
    for reg in ("text", "code"):
        tags = [form_tags(normalize_span(a)) for a in pools[reg]["train_file"]]
        words, chars = [t["n_words"] for t in tags], [t["n_chars"] for t in tags]
        cpw = [c / max(w, 1) for w, c in zip(words, chars)]
        out[reg] = {"rates": {f: float(np.mean([t[f] for t in tags])) for f in MATCH_FEATURES},
                    "shapes": list(zip(words, chars)),
                    # R3: quartile bins of the train attacks (code register only); edge bins catch out-of-range
                    "bins": {"n_words": np.quantile(words, [0, .25, .5, .75, 1]).tolist(),
                             "n_chars": np.quantile(chars, [0, .25, .5, .75, 1]).tolist(),
                             "chars_per_word": np.quantile(cpw, [0, .25, .5, .75, 1]).tolist()}
                    if reg == "code" else None}
    return out


def length_bin(x: float, edges: List[float]) -> int:
    """0 below the attack minimum, 1-4 for the quartiles, 5 above the maximum."""
    return int(np.searchsorted(edges[:-1], x, side="right")) + int(x > edges[-1])


def choose_clean_b(pool: List[dict], rng: random.Random, profile: dict, usage) -> dict:
    """Draw a desired feature vector from the train-attack rates and a target shape from a random train attack
    (code: also its word / char / chars-per-word quartile bins, R3 + round-2 iteration 2);
    take the candidate with the best weighted feature match minus its log-distance in words and chars,
    then the least used (usage(c) -> tuple), ties broken by the RNG."""
    want = {f: rng.random() < profile["rates"][f] for f in MATCH_FEATURES}
    tw, tc = rng.choice(profile["shapes"])
    bins = profile.get("bins")
    pool = list(pool)
    rng.shuffle(pool)

    def fit(c):
        t = c["tags"]
        match = sum(MATCH_WEIGHT[f] for f in MATCH_FEATURES if t[f] == want[f])
        dist = abs(np.log(max(t["n_words"], 1) / tw)) + abs(np.log(max(t["n_chars"], 1) / tc))
        return round(match - dist, 1)
    def bin_dist(c):  # R3: stratify by the (word bin, char bin) cell of the drawn train attack
        if not bins:
            return 0
        t = c["tags"]
        got = {"n_words": t["n_words"], "n_chars": t["n_chars"], "chars_per_word": t["n_chars"] / max(t["n_words"], 1)}
        return sum(abs(length_bin(got[k], bins[k]) - length_bin(v, bins[k]))
                   for k, v in (("n_words", tw), ("n_chars", tc), ("chars_per_word", tc / max(tw, 1))))
    # digits first: within-split donors are digit-heavy (email amounts, table values) and attacks mostly are not
    # length bins (R3) and the digit match carry equal weight; code donors are digit-heavy too
    return min(pool, key=lambda c: (bin_dist(c) + 2 * (c["tags"]["has_digit"] != want["has_digit"]), -fit(c), usage(c)))


# ─── form targets (D5, D6) ────────────────────────────────────────────────────

def form_profile(spans: List[str]) -> dict:
    tags = [form_tags(s) for s in spans]
    prof = {k: float(np.mean([t[k] for t in tags])) for k in FORM_BINARY}
    for k in FORM_NUMERIC:
        x = np.array([t[k] for t in tags], float)
        prof[k] = {"mean": float(x.mean()), "p10": float(np.percentile(x, 10)), "median": float(np.median(x)),
                   "p90": float(np.percentile(x, 90))}
    prof["n"] = len(spans)
    return prof


def clean_b_spans_all(docs: pd.DataFrame, tok, span_cap) -> Dict[str, List[str]]:
    """Expected clean-B spans per register (seed-independent): one selection per doc from all other docs'
    candidates with the same profile-aware policy as the build (usage spread, no k cap)."""
    rates, out = attack_rates(), defaultdict(list)
    for task in TASKS:
        g = docs[docs["task_type"] == task]
        cands = []
        for d in g.to_dict("records"):
            rng = random.Random(f"v3:profile:{d['doc_id']}")
            cands += [{"text": s, "donor": d["doc_id"], "tags": form_tags(normalize_span(s))}
                      for s in donor_candidates(task, d["context"], tok, span_cap[ATTACK_REGISTER[task]], rng)]
        rng, used = random.Random(f"v3:profile:{task}"), Counter()
        for d in g["doc_id"]:
            pool = [c for c in cands if c["donor"] != d]
            if pool:
                c = choose_clean_b(pool, rng, rates[ATTACK_REGISTER[task]], lambda c: (used[c["text"]], used[c["donor"]]))
                used[c["text"]] += 1
                used[c["donor"]] += 1
                out[task].append(c["text"])
    return out


def form_targets(docs: pd.DataFrame, tok, span_cap) -> dict:
    """Per register: train-attack profile, clean-B profile, clean-A target = clip(2 x attack - clean B)."""
    pools = load_attack_pools()
    cb = clean_b_spans_all(docs, tok, span_cap)
    out = {}
    for reg in TASKS:
        atk = form_profile(pools[ATTACK_REGISTER[reg]]["train_file"])
        b = form_profile(cb[reg])
        tgt = {k: float(np.clip(2 * atk[k] - b[k], 0, 1)) for k in FORM_BINARY}
        for k in FORM_NUMERIC:
            tgt[k] = {"mean": float(max(2 * atk[k]["mean"] - b[k]["mean"], atk[k]["p10"]))}
        out[reg] = {"train_attack": atk, "clean_b": b, "clean_a_target": tgt}
    return out


# ─── row building ─────────────────────────────────────────────────────────────

def pick(rng: random.Random, items: list, counts: Counter, key, n=1, exclude=()):
    """n distinct items among the least-used ones (ties broken by the stage RNG)."""
    cand = [x for x in items if key(x) not in exclude]
    rng.shuffle(cand)
    cand.sort(key=lambda x: counts[key(x)])
    chosen = cand[:n]
    if len(chosen) < n:
        raise RuntimeError("pool exhausted")
    for x in chosen:
        counts[key(x)] += 1
    return chosen


def normalize_span(span: str) -> str:
    """Applied identically to every inserted span: strip, and collapse blank lines to one newline (whitespace only;
    BIPIA code attacks end in a blank line before the closing fence, most donor chunks do not)."""
    return re.sub(r"\n[ \t]*(\n[ \t]*)+", "\n", span.strip())


def make_row(doc, ctx, span, position, sep, base: dict) -> dict:
    span = None if span is None else normalize_span(span)
    if span is None:
        text, off, kind, ln = ctx, "", "none", ""
    else:
        text, off, kind = insert_preserving(ctx, span, position, sep, V3_BOUNDARIES)
        ln = len(span.strip())
    return {**base, "doc_id": doc["doc_id"], "task_type": doc["task_type"], "user_intent": doc["user_intent"],
            "text": text, "span_text": "" if span is None else span.strip(), "insertion_mode": position,
            "separator": "" if span is None else sep.encode("unicode_escape").decode(), "boundary_kind": kind,
            "char_offset": off, "span_len": ln, "trimmed": doc["trimmed"], "component": doc["component"]}


def build(split_seed: int, benign_json: Path, doc_limit: Optional[int] = None) -> dict:
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    atk = load_attack_pools()
    span_cap = {reg: max(span_cost(tok, a["text"]) for s in ("train", "val", "test") for a in atk[reg][s])
                for reg in ("text", "code")}
    fitter = Fitter(tok, span_cap)
    docs = load_docs(doc_limit)
    comp = near_dup_components(docs)
    docs["component"] = docs["doc_id"].map(comp)
    docs = prepare_docs(docs, fitter)
    split_of = assign_splits(docs, comp, split_seed)
    docs["split"] = docs["doc_id"].map(split_of)
    benign = load_benign(benign_json, tok, span_cap)
    bpools = benign["pools"]

    main = docs[docs["main_context"].notna()].sort_values(["task_type", "doc_id"])
    docs_of = {(t, s): g for (t, s), g in main.groupby(["task_type", "split"])}

    # Donor candidates per (task, split); texts that also occur in another split's docs are excluded
    rng_d = rng_for(split_seed, "donors")
    donors, excluded = {}, Counter()
    for (task, split), g in docs_of.items():
        other = docs[(docs["task_type"] == task) & (docs["split"] != split)]["context"].tolist()
        cands = []
        for d in g.to_dict("records"):
            for s in donor_candidates(task, d["context"], tok, span_cap[ATTACK_REGISTER[task]], rng_d):
                if any(s in o for o in other):
                    excluded[task] += 1
                    continue
                cands.append({"text": s, "donor": d["doc_id"], "tags": form_tags(normalize_span(s))})
        donors[(task, split)] = cands
    # A text that is a candidate in two splits (e.g. the same rendered table sentence) is dropped from both
    for task in TASKS:
        seen = Counter()
        for split in ("train", "val", "test"):
            seen.update({c["text"] for c in donors.get((task, split), [])})
        for split in ("train", "val", "test"):
            if (task, split) in donors:
                keep = [c for c in donors[(task, split)] if seen[c["text"]] == 1]
                excluded[task] += len(donors[(task, split)]) - len(keep)
                donors[(task, split)] = keep

    rates = attack_rates()
    rng_pos, rng_atk, rng_ben, rng_don = (rng_for(split_seed, s) for s in ("positions", "attacks", "benign", "clean_b"))
    atk_counts, ben_counts, don_counts, donor_doc_counts = Counter(), Counter(), Counter(), Counter()
    rows = defaultdict(list)

    def place(rng=None):
        rng = rng or rng_pos
        return rng.choice(POSITIONS), rng.choice(SEPARATORS)

    def clean_b_for(doc, split, task, k):
        pool = [c for c in donors[(task, split)] if c["donor"] != doc["doc_id"] and c["text"] not in doc["context"]
                and don_counts[(split, c["text"])] < k
                and (task != "table" or donor_doc_counts[(split, c["donor"])] < TABLE_DONOR_DOC_MAX)]
        if not pool:
            raise RuntimeError(f"no clean-B donor left for {doc['doc_id']} ({task}/{split}, k={k})")
        c = choose_clean_b(pool, rng_don, rates[ATTACK_REGISTER[task]],
                           lambda c: (don_counts[(split, c["text"])], donor_doc_counts[(split, c["donor"])]))
        don_counts[(split, c["text"])] += 1
        donor_doc_counts[(split, c["donor"])] += 1
        return c

    for (task, split), g in docs_of.items():
        reg = ATTACK_REGISTER[task]
        for doc in g.to_dict("records"):
            ctx = doc["main_context"]
            base = {"split_seed": split_seed, "split": split, "eval_slice": "main"}
            for k, a in enumerate(pick(rng_atk, atk[reg][split], atk_counts, lambda x: x["text"], n=2)):
                rows["main"].append(make_row(doc, ctx, a["text"], *place(), {
                    **base, "id": f"{doc['doc_id']}_inj{k}", "label": 1, "row_type": "injected",
                    "span_source": a["category"], "attack_category": a["category"], "attack_source": a["source"],
                    "clean_type": ""}))
            b = pick(rng_ben, bpools[split][task], ben_counts, lambda x: x["id"])[0]
            rows["main"].append(make_row(doc, ctx, b["text"], *place(), {
                **base, "id": f"{doc['doc_id']}_cleanA", "label": 0, "row_type": "clean_a", "span_source": b["id"],
                "attack_category": "", "attack_source": "", "clean_type": "benign_imperative"}))
            c = clean_b_for(doc, split, task, DONOR_K[task])
            rows["main"].append(make_row(doc, ctx, c["text"], *place(), {
                **base, "id": f"{doc['doc_id']}_cleanB", "label": 0, "row_type": "clean_b",
                "span_source": f"donor:{c['donor']}", "attack_category": "", "attack_source": "",
                "clean_type": "spliced_neutral"}))
            if split in ("val", "test"):
                rows["raw_clean"].append(make_row(doc, ctx, None, "none", "", {
                    **base, "eval_slice": "raw_clean", "id": f"{doc['doc_id']}_raw", "label": 0,
                    "row_type": "raw_clean", "span_source": "", "attack_category": "", "attack_source": "",
                    "clean_type": "raw"}))
                amb = pick(rng_ben, bpools["ambiguous"][task], ben_counts, lambda x: x["id"])[0]
                rows["ambiguous"].append(make_row(doc, ctx, amb["text"], *place(), {
                    **base, "eval_slice": "ambiguous", "id": f"{doc['doc_id']}_amb", "label": "",
                    "row_type": "ambiguous", "span_source": amb["id"], "attack_category": "", "attack_source": "",
                    "clean_type": "ambiguous"}))
            if split == "val":
                for k, a in enumerate(pick(rng_atk, atk[reg]["train"], atk_counts, lambda x: x["text"], n=2)):
                    rows["val_id"].append(make_row(doc, ctx, a["text"], *place(), {
                        **base, "eval_slice": "val_id", "id": f"{doc['doc_id']}_valid{k}", "label": 1,
                        "row_type": "injected", "span_source": a["category"], "attack_category": a["category"],
                        "attack_source": a["source"], "clean_type": ""}))

    # long slice: docs excluded from main, test split only, test pools, 4 row types x available positions
    long_docs = docs[docs["main_context"].isna() & (docs["split"] == "test")].sort_values("doc_id")
    rng_long = rng_for(split_seed, "long")
    for doc in long_docs.to_dict("records"):
        task, reg, ctx = doc["task_type"], ATTACK_REGISTER[doc["task_type"]], doc["context"]
        attacks = pick(rng_long, atk[reg]["test"], Counter(), lambda x: x["text"], n=2)
        b = rng_long.choice(bpools["test"][task])
        don = [c for c in donors.get((task, "test"), []) if c["donor"] != doc["doc_id"]]
        c = rng_long.choice(don) if don else None
        spans = [(a["text"], 1, "injected", a["category"], a["source"], a["category"]) for a in attacks] + \
                [(b["text"], 0, "clean_a", "", "", b["id"])] + \
                ([(c["text"], 0, "clean_b", "", "", f"donor:{c['donor']}")] if c else [])
        positions = [p for p in POSITIONS if p != "middle" or has_internal_boundary(ctx)]
        for i, (span, label, rt, cat, src, ss) in enumerate(spans):
            for pos in positions:
                sep = rng_long.choice(SEPARATORS)
                d = {**doc, "trimmed": False}
                rows["long"].append(make_row(d, ctx, span, pos, sep, {
                    "split_seed": split_seed, "split": "test", "eval_slice": "long", "id": f"{doc['doc_id']}_long{i}_{pos}",
                    "label": label, "row_type": rt, "span_source": ss, "attack_category": cat, "attack_source": src,
                    "clean_type": {"injected": "", "clean_a": "benign_imperative", "clean_b": "spliced_neutral"}[rt]}))

    frames = {}
    for name, rs in rows.items():
        df = pd.DataFrame(rs)
        df["n_tokens"] = [len(x) for x in tok([build_combined_text(r) for r in rs])["input_ids"]]
        frames[name] = df[COLUMNS]
    for name in ("val_id", "raw_clean", "ambiguous", "long"):
        frames.setdefault(name, pd.DataFrame(columns=COLUMNS))

    manifest = {
        "split_seed": split_seed, "doc_limit": doc_limit, "max_tokens": MAX_TOKENS, "margin": MARGIN,
        "span_cap_tokens": span_cap, "val_categories": VAL_CATEGORIES, "donor_k": DONOR_K,
        "inputs_sha256": {**{f"{t}/{s}.jsonl": sha256_file(BENCH / t / f"{s}.jsonl") for t in TASKS
                             for s in ("train", "test")},
                          **{f: sha256_file(BENCH / f) for f in sorted(p.name for p in BENCH.glob("*_attack_*.json"))},
                          "benign_json": sha256_file(benign_json)},
        "benign_dropped_over_cap": benign["dropped_over_cap"],
        "benign_pool_sizes": {s: {r: len(v) for r, v in p.items()} for s, p in bpools.items()},
        "docs": {t: {"unique": int((docs["task_type"] == t).sum()),
                     "main": int((main["task_type"] == t).sum()),
                     "trimmed": int(docs[(docs["task_type"] == t)]["trimmed"].sum()),
                     "long_reason": dict(Counter(docs[(docs["task_type"] == t) & docs["main_context"].isna()]["long_reason"])),
                     "components_multi": int((docs[docs["task_type"] == t].groupby("component").size() > 1).sum())}
                 for t in TASKS},
        "d3_table": dict(Counter(docs[docs["task_type"] == "table"]["d3"])),
        "donor_texts_excluded_cross_split": dict(excluded),
        "distinct_donor_docs": {f"{t}/{s}": int(frames["main"][(frames["main"]["task_type"] == t) &
                                                                 (frames["main"]["split"] == s) &
                                                                 (frames["main"]["row_type"] == "clean_b")]
                                                ["span_source"].nunique()) for t, s in docs_of},
        "max_donor_text_use": max(don_counts.values()) if don_counts else 0,
        "counts": {name: {f"{t}/{s}": int(n) for (t, s), n in df.groupby(["task_type", "split"]).size().items()}
                   for name, df in frames.items()},
    }
    check(frames["main"], docs)
    return {"frames": frames, "manifest": manifest, "docs": docs}


# ─── integrity assertions ─────────────────────────────────────────────────────

def check(main: pd.DataFrame, docs: pd.DataFrame) -> None:
    splits = ("train", "val", "test")
    by = {s: main[main["split"] == s] for s in splits}
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            A, B = by[a], by[b]
            assert not set(A["doc_id"]) & set(B["doc_id"]), f"doc overlap {a}/{b}"
            assert not set(A["text"]) & set(B["text"]), f"exact-text overlap {a}/{b}"
            for rt, col in (("injected", "span_text"), ("clean_a", "span_source"), ("clean_b", "span_text")):
                sa = set(A.loc[A["row_type"] == rt, col])
                sb = set(B.loc[B["row_type"] == rt, col])
                assert not sa & sb, f"{rt} overlap {a}/{b}: {sorted(sa & sb)[:3]}"
    bal = main.groupby(["task_type", "split", "label"]).size().unstack("label")
    assert (bal[0] == bal[1]).all(), f"1:1 balance violated:\n{bal}"
    assert (main.groupby("doc_id").size() == 4).all() and (main.groupby("doc_id")["label"].sum() == 2).all()
    comp_splits = docs.dropna(subset=["split"]).groupby("component")["split"].nunique()
    assert (comp_splits == 1).all(), "near-duplicate component spans splits"
    assert (main["n_tokens"] <= MAX_TOKENS).all(), f"rows over {MAX_TOKENS}: {(main['n_tokens'] > MAX_TOKENS).sum()}"
    for r in main.itertuples():
        assert r.text[r.char_offset:r.char_offset + r.span_len] == r.span_text, r.id


def write(result: dict, output_dir: Path) -> Path:
    out = Path(output_dir) / f"split_seed_{result['manifest']['split_seed']}"
    assert_not_frozen(out)
    out.mkdir(parents=True, exist_ok=True)
    frames = result["frames"]
    for s in ("train", "val", "test"):
        frames["main"][frames["main"]["split"] == s].to_csv(out / f"{s}.csv", index=False, lineterminator="\n")
    for name in ("val_id", "raw_clean", "ambiguous", "long"):
        frames[name].to_csv(out / f"{name}.csv", index=False, lineterminator="\n")
    (out / "manifest.json").write_text(json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split_seed", type=int, required=True)
    ap.add_argument("--benign_json", type=Path, default=ROOT / "data" / "benign" / "benign_v3.json")
    ap.add_argument("--output_dir", type=Path, default=ROOT / "data" / "v3")
    ap.add_argument("--doc_limit", type=int, default=None, help="debug/tests: first N unique docs per task")
    a = ap.parse_args()
    out = write(build(a.split_seed, a.benign_json, a.doc_limit), a.output_dir)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
