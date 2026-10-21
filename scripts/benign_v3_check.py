"""
benign_v3_check.py
==================
Tools for the V3 benign pool (docs/v3_design.md D5/D6).

  assemble   turn hand-written source files (<src>/<register>_<pool>.txt, items separated by a line '---')
             into a pool JSON with ids and form_tags
  report     per register and pool: achieved form profile vs the clean-A target, top-20 attack-indicative tokens
             (benign vs attack usage), max similarity to any attack string and across train_val/test pools

    python scripts/benign_v3_check.py assemble --src <dir> --out data/benign/benign_v3_draft.json
    python scripts/benign_v3_check.py report --json data/benign/benign_v3_draft.json [--out report.json]
"""

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_pipeline_v3 as P
from probes import FORM_BINARY, form_tags

REGISTERS = ("email", "table", "code")
POOLS = ("train_val", "test", "ambiguous")
WORD = re.compile(r"[a-z']+")


def assemble(src: Path, out: Path):
    items = []
    for reg in REGISTERS:
        for pool in POOLS:
            f = src / f"{reg}_{pool}.txt"
            if not f.exists():
                continue
            texts = [t.strip() for t in re.split(r"(?m)^---\s*$", f.read_text()) if t.strip()]
            for i, t in enumerate(texts):
                items.append({"id": f"{reg}_{pool}_{i:03d}", "text": t, "split_pool": pool, "register": reg,
                              "form_tags": form_tags(t)})
    out.write_text(json.dumps({"version": "v3-draft", "items": items}, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(items)} items)")


def _words(s):
    return set(WORD.findall(s.lower()))


def indicative_tokens(attacks, reference, n=20, min_df=3):
    """Log-odds of per-item document frequency, attack strings vs reference units (lines/sentences of BIPIA
    contexts of the register), add-one smoothed."""
    da, dr = Counter(), Counter()
    for a in attacks:
        da.update(_words(a))
    for r in reference:
        dr.update(_words(r))
    na, nr = len(attacks), len(reference)
    score = {w: math.log((da[w] + 1) / (na + 2)) - math.log((dr[w] + 1) / (nr + 2)) for w in da if da[w] >= min_df}
    top = sorted(score, key=lambda w: (-score[w], w))[:n]
    return [{"token": w, "log_odds": round(score[w], 3), "attack_rate": da[w] / na} for w in top]


def reference_units(docs, tasks):
    units = []
    for c in docs[docs["task_type"].isin(tasks)]["context"]:
        units += [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", c) if len(u.split()) >= 3]
    return units


def report(items, out=None):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(P.TOKENIZER))
    atk = P.load_attack_pools()
    span_cap = {reg: max(P.span_cost(tok, a["text"]) for s in ("train", "val", "test") for a in atk[reg][s])
                for reg in ("text", "code")}
    docs = P.load_docs()
    targets = P.form_targets(docs, tok, span_cap)
    res = {"targets": targets, "registers": {}}
    all_attacks = [a["text"] for reg in ("text", "code") for s in ("train", "val", "test") for a in atk[reg][s]]
    for reg in REGISTERS:
        areg = P.ATTACK_REGISTER[reg]
        train_atk = atk[areg]["train_file"]
        ref = reference_units(docs, ["code"] if areg == "code" else ["email", "table"])
        top = indicative_tokens(train_atk, ref)
        r = {"counts": {}, "profile": {}, "tokens": [], "over_cap": []}
        for pool in POOLS:
            texts = [it["text"] for it in items if it["register"] == reg and it["split_pool"] == pool]
            r["counts"][pool] = len(texts)
            if texts:
                r["profile"][pool] = P.form_profile(texts)
            r["over_cap"] += [it["id"] for it in items if it["register"] == reg and it["split_pool"] == pool
                              and P.span_cost(tok, it["text"]) > span_cap[areg]]
        clean_a = [it["text"] for it in items if it["register"] == reg and it["split_pool"] in ("train_val", "test")]
        for t in top:
            rate = float(np.mean([t["token"] in _words(x) for x in clean_a])) if clean_a else 0.0
            r["tokens"].append({**t, "benign_rate": rate})
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(all_attacks + clean_a)
        if clean_a:
            sim = cosine_similarity(vec.transform(clean_a), vec.transform(all_attacks)).max(axis=1)
            r["max_sim_to_attack"] = float(sim.max())
            r["max_sim_item"] = clean_a[int(sim.argmax())]
        tv = [it["text"] for it in items if it["register"] == reg and it["split_pool"] == "train_val"]
        te = [it["text"] for it in items if it["register"] == reg and it["split_pool"] == "test"]
        if tv and te:
            cs = cosine_similarity(vec.transform(te), vec.transform(tv))
            i, j = np.unravel_index(cs.argmax(), cs.shape)
            r["max_cross_pool_sim"] = float(cs.max())
            r["max_cross_pool_pair"] = [te[i], tv[j]]
        res["registers"][reg] = r
    print_report(res)
    if out:
        Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    return res


def print_report(res):
    for reg, r in res["registers"].items():
        t = res["targets"][reg]
        print(f"\n== {reg}: counts {r['counts']}  over span cap: {r['over_cap']}")
        print(f"{'feature':26s} {'attack':>7s} {'cleanB':>7s} {'target':>7s} " +
              " ".join(f"{p:>9s}" for p in ("train_val", "test")))
        for k in FORM_BINARY + ("n_words", "n_chars", "cap_ratio"):
            a, b, g = t["train_attack"][k], t["clean_b"][k], t["clean_a_target"][k]
            a, b, g = (x["mean"] if isinstance(x, dict) else x for x in (a, b, g))
            ach = []
            for p in ("train_val", "test"):
                v = r["profile"].get(p, {}).get(k)
                ach.append(f"{(v['mean'] if isinstance(v, dict) else v):9.3f}" if v is not None else f"{'-':>9s}")
            print(f"{k:26s} {a:7.3f} {b:7.3f} {g:7.3f} " + " ".join(ach))
        print("top-20 attack-indicative tokens (attack rate / clean-A rate):")
        print("  " + ", ".join(f"{x['token']} {x['attack_rate']:.2f}/{x['benign_rate']:.2f}" for x in r["tokens"]))
        if "max_sim_to_attack" in r:
            print(f"max sim to attack {r['max_sim_to_attack']:.3f}; max cross-pool sim {r.get('max_cross_pool_sim', 0):.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assemble")
    a.add_argument("--src", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report")
    r.add_argument("--json", type=Path, required=True)
    r.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.cmd == "assemble":
        assemble(args.src, args.out)
    else:
        report(json.load(open(args.json))["items"], args.out)


if __name__ == "__main__":
    main()
