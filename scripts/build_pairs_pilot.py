"""
build_pairs_pilot.py
====================
Build the hand-written pilot pairs (docs/pairs_spec.md §11) from data/pairs/pilot_v1_specs.json:
data/pairs/pilot_v1.jsonl, one pair per line (src/pairs.PAIR_FIELDS).

Each quad names a V3 test document of split_seed 0. The split is recomputed in memory with the V3 builder's own
functions (load_docs, near_dup_components, prepare_docs, assign_splits); no V3 build is read or written. The
context is the doc's main_context (trimmed code/table window, D3). Position and separator are drawn i.i.d.
uniform per pair with a seeded RNG, as in the V3 builder (§2), and the instruction is inserted with
probes.insert_preserving and the V3 boundaries. The V3 length rules apply: the instruction's span cost is within
its register's cap, and `Intent: <request> | Context: <context>` fits 512 DeBERTa-v3 tokens for both requests.

    python scripts/build_pairs_pilot.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_pipeline_v3 as v3
from pairs import make_pair, validate
from preprocess import build_combined_text

SPECS = ROOT / "data" / "pairs" / "pilot_v1_specs.json"
OUT = ROOT / "data" / "pairs" / "pilot_v1.jsonl"
SPLIT_SEED = 0


def v3_test_docs(split_seed: int):
    """(doc_id -> {doc_id, task, context} for the V3 test docs of split_seed, tokenizer, span cap per register),
    in memory, split exactly as v3.build splits them."""
    from transformers import AutoTokenizer
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    tok = AutoTokenizer.from_pretrained(str(v3.TOKENIZER))
    atk = v3.load_attack_pools()
    cap = {reg: max(v3.span_cost(tok, a["text"]) for s in ("train", "val", "test") for a in atk[reg][s])
           for reg in ("text", "code")}
    docs = v3.load_docs()
    comp = v3.near_dup_components(docs)
    docs["component"] = docs["doc_id"].map(comp)
    docs = v3.prepare_docs(docs, v3.Fitter(tok, cap))
    split = v3.assign_splits(docs[docs["main_context"].notna()], comp, split_seed)
    test = docs[docs["doc_id"].map(split) == "test"]
    return {d["doc_id"]: {"doc_id": d["doc_id"], "task": d["task_type"], "context": d["main_context"]}
            for d in test.to_dict("records")}, tok, cap


def length_errors(p: dict, tok, cap: dict) -> list:
    errors = []
    reg = v3.ATTACK_REGISTER[p["task"]]
    if v3.span_cost(tok, p["instruction"]) > cap[reg]:
        errors.append(f"{p['pair_id']}: instruction over the {reg} span cap ({cap[reg]} tokens)")
    for role in ("aligned", "subverting"):
        n = len(tok(build_combined_text({"user_intent": p[f"request_{role}"], "text": p["context"]}))["input_ids"])
        if n > 512:
            errors.append(f"{p['pair_id']}: {role} member is {n} tokens (> 512)")
    return errors


def main():
    specs = json.load(open(SPECS))
    docs, tok, cap = v3_test_docs(SPLIT_SEED)
    rng = random.Random(f"pairs:{specs['version']}:positions")
    pairs = []
    for q in specs["quads"]:
        doc = docs.get(q["doc_id"])
        if doc is None:
            sys.exit(f"{q['quad_id']}: {q['doc_id']} is not a V3 test doc of split_seed {SPLIT_SEED}")
        for k, (ins, ra, rs) in enumerate([(q["i1"], q["r1"], q["r2"]), (q["i2"], q["r2"], q["r1"])], start=1):
            pairs.append(make_pair(f"{q['quad_id']}_p{k}", q["quad_id"], doc, ins, ra, rs,
                                   rng.choice(v3.POSITIONS), rng.choice(v3.SEPARATORS),
                                   source=f"v3_test@split_seed{SPLIT_SEED}", generator="hand_pilot_v1",
                                   notes=q["notes"][k - 1]))
    errors = validate(pairs) + [e for p in pairs for e in length_errors(p, tok, cap)]
    if errors:
        sys.exit("invalid pairs:\n  " + "\n  ".join(errors))
    with open(OUT, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(pairs)} pairs in {len(specs['quads'])} quads")
    for p in pairs:
        print(f"  {p['pair_id']:14s} {p['task']:6s} {p['position']:6s} {p['separator']:5s} {p['boundary_kind']:9s} "
              f"{p['overlap_class']:8s} J(aligned)={p['overlap_aligned']:.3f} J(subverting)={p['overlap_subverting']:.3f}")


if __name__ == "__main__":
    main()
