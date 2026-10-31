"""
agent_overlap_audit.py
======================
Overlap audit of the agent-style synthetic data (src/agent_pipeline.py, paper_plan A3) against:

  agentdojo   AgentDojo 0.1.34 templates: every .yaml under agentdojo/data/ and every string literal in
              agentdojo/attacks/*.py and default_suites/**/{injection,user}_tasks.py, read from the release wheel
              (--agentdojo_wheel, sha256 recorded), plus the rebuilt AD tool outputs in data/transfer/sets
  taubench    tau-bench tool outputs and user requests in data/transfer/sets/taubench_*.jsonl
  injecagent  InjecAgent attacker/user instructions and tool-response templates (data/raw/InjecAgent/data)
  bipia_test  BIPIA test attacks (text_attack_test.json, code_attack_test.json)

Units checked: the calibration pool outputs, every training/val output (raw context), and every inserted span
(clean-A, clean-B, injected) of each split seed given.

Measures (fixed before the first run; text normalised to lowercase word tokens, \\w+):
  - containment: a reference instruction (InjecAgent attacker/user instructions, BIPIA test attacks, AgentDojo
    injection-task GOALs and attack-template strings; >= 5 words) occurring whole inside one of our texts.
  - shared word 8-grams: the share of our texts that share >= 1 word 8-gram with the reference corpus, and the
    shared 8-grams themselves (top 30 by count). The same share for 5-grams is reported, not gated.
  - positive control: the first 50 corpus texts with >= 8 words, audited against their own corpus, must all share an
    8-gram (and every instruction must be found in itself); otherwise the audit is void.
PASS iff containment = 0 for every corpus and unit, and no clean output or clean span (calibration pool, contexts,
clean-A, clean-B) shares an 8-gram with agentdojo or taubench. Injected spans are BIPIA train attacks; their 8-gram
overlap with each corpus (e.g. InjecAgent, BIPIA test) is reported, not gated.

    python scripts/agent_overlap_audit.py --agentdojo_wheel <path>/agentdojo-0.1.34-py3-none-any.whl \\
        --data_dir data/agent_v2 --split_seeds 0 1 2 3 4
"""

import argparse
import ast
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from runlog import add_run_args, start_run

N = 8
MIN_WORDS = 5
SETS = ROOT / "data" / "transfer" / "sets"
TOK = re.compile(r"\w+")


def words(s: str):
    return TOK.findall(str(s).lower())


def ngrams(s: str, n=N) -> set:
    w = words(s)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def py_strings(src: str):
    out = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            out.append(node.value)
        elif isinstance(node, ast.JoinedStr):  # f-strings: keep the literal parts
            out.append(" ".join(v.value for v in node.values if isinstance(v, ast.Constant)))
    return out


def jsonl(p: Path, keys):
    out = []
    for line in open(p):
        d = json.loads(line)
        out += [str(d[k]) for k in keys if d.get(k)]
    return out


def corpora(wheel: Path):
    """{name: {"texts": [...], "instructions": [...]}}; instructions are the containment references."""
    ad_texts, ad_instr = [], []
    with zipfile.ZipFile(wheel) as z:
        for n in z.namelist():
            if n.startswith("agentdojo/data/") and n.endswith(".yaml"):
                ad_texts.append(z.read(n).decode())
            elif n.endswith(".py") and (n.startswith("agentdojo/attacks/") or
                                        re.search(r"default_suites/.*/(injection|user)_tasks\.py$", n)):
                strs = py_strings(z.read(n).decode())
                ad_texts += strs
                if "/attacks/" in n or "injection_tasks" in n:
                    ad_instr += strs
    for f in ("agentdojo_benign_outputs.jsonl", "agentdojo_pairs.jsonl", "agentdojo_pairs_shift.jsonl"):
        ad_texts += jsonl(SETS / f, ["text", "user_request"])
    tb = [t for f in ("taubench_clean.jsonl", "taubench_eval.jsonl", "taubench_pairs.jsonl")
          for t in jsonl(SETS / f, ["text", "user_request"])]
    ia_dir = ROOT / "data" / "raw" / "InjecAgent" / "data"
    ia_texts, ia_instr = [], []
    for f in sorted(ia_dir.glob("test_cases_*.json")):
        for d in json.load(open(f)):
            ia_instr += [d["Attacker Instruction"], d["User Instruction"]]
            ia_texts += [d["Attacker Instruction"], d["User Instruction"], d.get("Tool Response Template", ""),
                         d.get("Expected Achievements", "")]
    bench = ROOT / "data" / "raw" / "BIPIA" / "benchmark"
    bt = [a for f in ("text_attack_test.json", "code_attack_test.json") for v in json.load(open(bench / f)).values()
          for a in v]
    return {"agentdojo": {"texts": ad_texts, "instructions": ad_instr},
            "taubench": {"texts": tb, "instructions": []},
            "injecagent": {"texts": ia_texts, "instructions": ia_instr},
            "bipia_test": {"texts": bt, "instructions": bt}}


def audit_unit(texts, corpus):
    grams = set().union(*(ngrams(t) for t in corpus["texts"])) if corpus["texts"] else set()
    grams5 = set().union(*(ngrams(t, 5) for t in corpus["texts"])) if corpus["texts"] else set()
    instr = {" ".join(words(s)) for s in corpus["instructions"] if len(words(s)) >= MIN_WORDS}
    long_ = {s: " ".join(s.split()[:N]) for s in instr if len(s.split()) >= N}  # prefilter on the first 8-gram
    short = [s for s in instr if s not in long_]
    shared, n_hit, contained = Counter(), 0, Counter()
    for t in texts:
        tg = ngrams(t)
        g = tg & grams
        n_hit += bool(g)
        shared.update(g)
        norm = " " + " ".join(words(t)) + " "
        contained.update(s for s in short if f" {s} " in norm)
        contained.update(s for s, first in long_.items() if first in tg and f" {s} " in norm)
    n5 = sum(bool(ngrams(t, 5) & grams5) for t in texts)
    return {"n": len(texts), "n_sharing_5gram": n5, "n_sharing_8gram": n_hit, "share_sharing_8gram": n_hit / len(texts) if texts else 0.0,
            "top_shared_8grams": shared.most_common(30), "n_containments": sum(contained.values()),
            "contained_instructions": contained.most_common(20)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agentdojo_wheel", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, default=ROOT / "data" / "agent_v2")
    ap.add_argument("--calibration", type=Path, default=ROOT / "data" / "agent_v1" / "calibration_pool.csv")
    ap.add_argument("--split_seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--exp", default="agent_overlap_audit")
    a = add_run_args(ap).parse_args()
    cal_path = a.calibration
    split_files = [a.data_dir / f"split_seed_{k}" / f"{s}.csv" for k in a.split_seeds for s in ("train", "val")]
    wheel_sha = hashlib.sha256(a.agentdojo_wheel.read_bytes()).hexdigest()
    run = start_run(version=a.data_dir.name, exp=a.exp, seed=a.split_seeds[0],
                    args={"data_dir": str(a.data_dir.relative_to(ROOT)), "split_seeds": a.split_seeds, "ngram": N,
                          "min_instruction_words": MIN_WORDS, "agentdojo_wheel": a.agentdojo_wheel.name,
                          "agentdojo_wheel_sha256": wheel_sha},
                    data_version=a.data_dir.name, data_files=[cal_path, *split_files], allow_dirty=a.allow_dirty)
    C = corpora(a.agentdojo_wheel)
    run.log("corpus sizes: " + ", ".join(f"{k} {len(v['texts'])} texts / {len(v['instructions'])} instructions"
                                          for k, v in C.items()))
    control = {}
    for c, corpus in C.items():
        own = [t for t in corpus["texts"] if len(words(t)) >= N][:50]
        r = audit_unit(own, corpus)
        ins = [s for s in corpus["instructions"] if len(words(s)) >= MIN_WORDS][:50]
        ri = audit_unit(ins, {"texts": [], "instructions": ins}) if ins else {"n_containments": 0}
        control[c] = {"texts_hit": r["n_sharing_8gram"], "texts": len(own), "instr_found": ri["n_containments"],
                      "instr": len(ins)}
        assert r["n_sharing_8gram"] == len(own) and ri["n_containments"] >= len(ins), f"positive control failed: {c}"
    run.log(f"positive control: {control}")
    cal = pd.read_csv(cal_path, keep_default_na=False)
    rows = pd.concat([pd.read_csv(f, keep_default_na=False) for f in split_files], ignore_index=True)
    ctx = rows.drop_duplicates("doc_id")
    contexts = [t[:int(o)] + t[int(o) + int(n):]  # the raw output with the inserted span cut out
                for t, o, n in zip(ctx["text"], ctx["char_offset"], ctx["span_len"])]
    units = {"calibration_pool": cal["text"].tolist(), "contexts": contexts}
    for rt in ("clean_a", "clean_b", "injected"):
        units[f"span_{rt}"] = sorted(set(rows.loc[rows["row_type"] == rt, "span_text"]))
    res, fails = {}, []
    for u, texts in units.items():
        res[u] = {}
        for c, corpus in C.items():
            r = audit_unit(texts, corpus)
            res[u][c] = r
            run.log(f"{u:18s} vs {c:10s}: n={r['n']:6d} sharing 5-gram {r['n_sharing_5gram']:5d}, 8-gram {r['n_sharing_8gram']:5d} "
                    f"({r['share_sharing_8gram']:.4f}), containments {r['n_containments']}")
            if r["n_containments"]:
                fails.append(f"{u}/{c}: containment")
            if u != "span_injected" and c in ("agentdojo", "taubench") and r["n_sharing_8gram"]:
                fails.append(f"{u}/{c}: shared 8-gram")
    verdict = {"pass": not fails, "fail": fails}
    run.log(f"audit: {'PASS' if not fails else 'FAIL ' + '; '.join(fails)}")
    run.save_metrics({"units": res, "verdict": verdict, "positive_control": control, "corpus_sizes": {k: len(v["texts"]) for k, v in C.items()}},
                     headline=("audit_pass", float(not fails)))
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
