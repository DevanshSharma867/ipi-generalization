"""
v3_design_facts.py
==================
Read-only facts about the raw BIPIA data that docs/v3_design.md relies on. Builds nothing.

  - BIPIA code-row fields, and distinct intents per task under the proposed V3 CodeQA intent
  - unique docs per task (V2 normalization and exact-string dedup), line-boundary availability
  - attack categories per file, and the proposed seeded train/val category draw
  - form profile of the attack pools and the V2 benign pools (length, "?", quotes, digits, multi-line)
  - spliceable donor sentences per task (V2 probe extractors)
  - projected effect of the 512-token DeBERTa-v3 limit per task, and of a trimming fallback

Logged with runlog to results/v3/design_facts/<run_id>/metrics.json.

    python scripts/v3_design_facts.py
"""

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from build_probes_v2 import code_sentences
from data_pipeline import load_attacks, load_jsonl
from probes import _line_starts_outside_fences, code_chunks, email_sentences
from runlog import start_run
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
BENCH = ROOT / "data" / "raw" / "BIPIA" / "benchmark"
TOK = ROOT / "hf_cache" / "tokenizers_hpc" / "deberta_v2_ckpt"
TASKS = ("email", "table", "code")
ATTACK_TYPE = {"email": "text", "table": "text", "code": "code"}
CATEGORY_DRAW_SEED = 0
VAL_CATEGORIES = {"text": 3, "code": 2}
MAX_TOKENS = 512


def norm_context(ctx):
    """Exactly the V2 normalization in data_pipeline.build_dataset."""
    if isinstance(ctx, list):
        ctx = "\n".join(str(x) for x in ctx)
    elif isinstance(ctx, dict):
        ctx = json.dumps(ctx)
    return str(ctx).strip()


def last_line(lines):
    lines = [str(l).strip() for l in lines if str(l).strip()]
    return lines[-1] if lines else ""


def v3_intent(item, task):
    """Proposed V3 intent: the BIPIA question for email/table; for code, the final line of the traceback."""
    if task != "code":
        return str(item["question"]).strip()
    return f"Fix this error in my code: {last_line(item['error'])}"


def unique_docs(task):
    seen, docs = set(), []
    for split in ("train", "test"):
        for item in load_jsonl(BENCH / task / f"{split}.jsonl"):
            ctx = norm_context(item.get("context", ""))
            if ctx and ctx not in seen:
                seen.add(ctx)
                docs.append({"context": ctx, "item": item, "bipia_split": split})
    return docs


def quantiles(x):
    x = np.asarray(x, dtype=float)
    return {k: float(v) for k, v in zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                        np.percentile(x, [0, 10, 25, 50, 75, 90, 100]))} | {"mean": float(x.mean())}


def form_profile(strings):
    s = [x.strip() for x in strings]
    share = lambda f: float(np.mean([bool(f(x)) for x in s]))
    return {"n": len(s), "words": quantiles([len(x.split()) for x in s]), "chars": quantiles([len(x) for x in s]),
            "ends_question": share(lambda x: x.endswith("?")), "ends_period": share(lambda x: x.endswith(".")),
            "ends_other": share(lambda x: not x.endswith((".", "?"))),
            "has_question_mark": share(lambda x: "?" in x),
            "has_double_quote": share(lambda x: re.search(r"[\"“”]", x)), "has_single_quote_or_apostrophe":
            share(lambda x: re.search(r"['‘’]", x)), "has_backtick": share(lambda x: "`" in x),
            "has_digit": share(lambda x: re.search(r"\d", x)), "multi_line": share(lambda x: "\n" in x),
            "starts_upper": share(lambda x: x[:1].isupper()), "non_ascii": share(lambda x: not x.isascii()),
            "second_person": share(lambda x: re.search(r"\b(you|your)\b", x, re.I))}


def main():
    tok = AutoTokenizer.from_pretrained(str(TOK))
    train_atk, test_atk = load_attacks(BENCH)
    benign = json.load(open(ROOT / "data" / "benign" / "benign_instructions.json"))
    raw_files = {f: json.load(open(BENCH / f)) for f in
                 ("text_attack_train.json", "text_attack_test.json", "code_attack_train.json", "code_attack_test.json")}
    run = start_run(version="v3", exp="design_facts", seed=CATEGORY_DRAW_SEED,
                    args={"max_tokens": MAX_TOKENS, "category_draw_seed": CATEGORY_DRAW_SEED,
                          "val_categories": VAL_CATEGORIES, "tokenizer": str(TOK.relative_to(ROOT))},
                    data_version="bipia@a004b69",
                    data_files=[BENCH / t / f"{s}.jsonl" for t in TASKS for s in ("train", "test")]
                    + [BENCH / f for f in raw_files] + [ROOT / "data" / "benign" / "benign_instructions.json"])
    out = {}

    # 1. Code fields and intents
    code_items = load_jsonl(BENCH / "code" / "train.jsonl") + load_jsonl(BENCH / "code" / "test.jsonl")
    out["code_fields"] = {
        "rows": len(code_items), "keys": dict(Counter(k for it in code_items for k in it)),
        "has_question": sum("question" in it for it in code_items),
        "error_first_line": dict(Counter(str(it["error"][0]).strip() for it in code_items if it.get("error"))),
        "error_last_line_distinct": len({last_line(it["error"]) for it in code_items}),
        "error_lines": quantiles([len(it["error"]) for it in code_items]),
        "code_lines": quantiles([len(it["code"]) for it in code_items]),
        "exception_types": dict(Counter(last_line(it["error"]).split(":")[0] for it in code_items).most_common()),
        "examples_last_line": [last_line(it["error"]) for it in code_items[:5]]}

    docs = {t: unique_docs(t) for t in TASKS}
    out["docs"] = {}
    for t, ds in docs.items():
        intents = [v3_intent(d["item"], t) for d in ds]
        lines = [len(_line_starts_outside_fences(d["context"])) for d in ds]
        out["docs"][t] = {
            "raw_rows": sum(len(load_jsonl(BENCH / t / f"{s}.jsonl")) for s in ("train", "test")),
            "unique": len(ds), "distinct_v3_intents": len(set(intents)),
            "max_docs_per_intent": max(Counter(intents).values()),
            "docs_with_newline": sum("\n" in d["context"] for d in ds),
            "docs_with_internal_line_boundary": sum(n > 0 for n in lines),
            "internal_line_boundaries": quantiles(lines),
            "docs_with_code_fence": sum("```" in d["context"] for d in ds),
            "docs_with_paragraph_break": sum("\n\n" in d["context"] for d in ds),
            "docs_without_line_boundary_with_field_or_sentence_boundary": sum(
                n == 0 and bool(re.search(r"\|[A-Z_ ]+:|[.!?] +[A-Z]", d["context"])) for n, d in zip(lines, ds)),
            "docs_without_any_internal_boundary": sum(
                n == 0 and not re.search(r"\|[A-Z_ ]+:|[.!?] +[A-Z]", d["context"]) for n, d in zip(lines, ds))}

    # 2. Attack categories and the proposed draw
    out["attack_categories"] = {f: {c: len(v) for c, v in d.items()} for f, d in raw_files.items()}
    draw = {}
    for kind in ("text", "code"):
        cats = sorted(raw_files[f"{kind}_attack_train.json"])
        val = sorted(random.Random(CATEGORY_DRAW_SEED).sample(cats, VAL_CATEGORIES[kind]))
        draw[kind] = {"val_categories": val, "train_categories": [c for c in cats if c not in val],
                      "val_strings": sum(len(raw_files[f"{kind}_attack_train.json"][c]) for c in val),
                      "category_overlap_train_file_test_file": sorted(
                          set(raw_files[f"{kind}_attack_train.json"]) & set(raw_files[f"{kind}_attack_test.json"]))}
    out["category_draw"] = draw

    # 3. Form profiles
    out["form_profile"] = {
        "text_attack_train": form_profile([a["attack"] for a in train_atk["text"]]),
        "code_attack_train": form_profile([a["attack"] for a in train_atk["code"]]),
        "text_attack_test": form_profile([a["attack"] for a in test_atk["text"]]),
        "code_attack_test": form_profile([a["attack"] for a in test_atk["code"]]),
        "benign_train_v2": form_profile(benign["benign_train"]),
        "benign_test_v2": form_profile(benign["benign_test"])}
    out["note_templates_in_benign_train"] = [b for b in benign["benign_train"] if b.lower().startswith("note")]

    # 4. Donor sentences (V2 probe extractors; table has no natural declaratives)
    out["donors"] = {}
    for t, extract in (("email", email_sentences), ("code", code_sentences)):
        per_doc = {i: extract(d["context"]) for i, d in enumerate(docs[t])}
        sents = [s for v in per_doc.values() for s in v]
        out["donors"][t] = {"docs_with_candidate": sum(bool(v) for v in per_doc.values()),
                            "candidate_sentences": len(sents), "distinct_sentences": len(set(sents)),
                            "words": quantiles([len(s.split()) for s in sents]) if sents else None}
    out["donors"]["table"] = {"docs_with_candidate": 0, "note": "V2 table_sentences synthesizes 'In this table, ...'"}
    # Proposed V3 code donors: a prose line followed by a fenced block (the form of every code attack),
    # spliced verbatim from another doc, within the code-attack token budget
    code_budget = max(len(tok("\n\n" + a["attack"], add_special_tokens=False)["input_ids"])
                      for a in train_atk["code"] + test_atk["code"])
    chunks = {i: [c for c in code_chunks(d["context"])
                  if len(tok("\n\n" + c, add_special_tokens=False)["input_ids"]) <= code_budget]
              for i, d in enumerate(docs["code"])}
    flat = [c for v in chunks.values() for c in v]
    out["donors"]["code_v3_chunks"] = {"docs_with_candidate": sum(bool(v) for v in chunks.values()),
                                       "candidate_chunks": len(flat), "distinct_chunks": len(set(flat)),
                                       "token_budget": code_budget, "words": quantiles([len(c.split()) for c in flat])}
    # Donor docs available inside each split, per split_seed (V2 split rule), before the length filter
    per_seed = {}
    donor_ok = {"email": lambda d: bool(email_sentences(d["context"])), "table": lambda d: True,
                "code": None}
    for t in TASKS:
        has = [bool(chunks[i]) for i in range(len(docs[t]))] if t == "code" else [donor_ok[t](d) for d in docs[t]]
        rows = []
        for seed in range(5):
            idx = list(range(len(docs[t])))
            random.Random(seed).shuffle(idx)
            nt = max(1, round(0.15 * len(idx)))
            parts = {"test": idx[:nt], "val": idx[nt:2 * nt], "train": idx[2 * nt:]}
            rows.append({s: {"docs": len(p), "donor_docs": sum(has[i] for i in p)} for s, p in parts.items()})
        per_seed[t] = rows
    out["donor_docs_per_split_seed"] = per_seed

    # 5. Token budget: base = "Intent: <v3 intent> | Context: <ctx>" with specials; span cost incl. separator
    span_cost = lambda s: len(tok("\n\n" + s.strip(), add_special_tokens=False)["input_ids"])
    costs = {k: [span_cost(a["attack"]) for a in train_atk[k] + test_atk[k]] for k in ("text", "code")}
    out["span_tokens"] = {k: quantiles(v) for k, v in costs.items()}
    out["length_filter"] = {}
    for t, ds in docs.items():
        base = np.array([len(tok(f"Intent: {v3_intent(d['item'], t)} | Context: {d['context']}")["input_ids"])
                         for d in ds])
        c = np.array(costs[ATTACK_TYPE[t]])
        smax, s90 = int(c.max()), int(np.percentile(c, 90))
        keep_max, keep_p90 = base + smax <= MAX_TOKENS, base + s90 <= MAX_TOKENS
        # Trimming fallback: keep a window of whole lines whose tokens fit MAX_TOKENS - smax - intent overhead
        out["length_filter"][t] = {
            "docs": len(ds), "base_tokens": quantiles(base), "span_budget_max": smax, "span_budget_p90": s90,
            "fit_with_max_span": int(keep_max.sum()), "removed_with_max_span": int((~keep_max).sum()),
            "removed_share_max_span": float((~keep_max).mean()),
            "fit_with_p90_span": int(keep_p90.sum()), "removed_share_p90_span": float((~keep_p90).mean()),
            "base_over_512": int((base > MAX_TOKENS).sum())}
        need = ~keep_max
        if need.any():
            kept_frac = []
            for d, b in zip([d for d, n in zip(ds, need) if n], base[need]):
                intent_tok = len(tok(f"Intent: {v3_intent(d['item'], t)} | Context: ")["input_ids"])
                ctx_budget = MAX_TOKENS - smax - intent_tok
                ctx_tok = len(tok(d["context"], add_special_tokens=False)["input_ids"])
                kept_frac.append(max(ctx_budget, 0) / ctx_tok)
            out["length_filter"][t]["trim_context_budget_tokens"] = int(MAX_TOKENS - smax - min(
                len(tok(f"Intent: {v3_intent(d['item'], t)} | Context: ")["input_ids"]) for d in ds))
            out["length_filter"][t]["trimmed_docs_kept_fraction_of_context"] = quantiles(kept_frac)

    # 6. Projected split sizes (V2 rule: n_val = n_test = round(0.15 n), per task) for keep-max and all-docs
    def split_counts(n):
        nt = max(1, round(0.15 * n))
        return {"train": n - 2 * nt, "val": nt, "test": nt}
    out["projected_splits"] = {t: {"filtered": split_counts(out["length_filter"][t]["fit_with_max_span"]),
                                   "all_docs_with_trimming": split_counts(len(docs[t]))} for t in TASKS}

    for k, v in out.items():
        run.log(f"{k}: {json.dumps(v)[:400]}")
    run.save_metrics(out)
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
