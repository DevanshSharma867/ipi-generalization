"""
pairs_gen.py
============
Intent-pair generation (docs/pairs_spec.md §8, v2; paper_plan A18, A24, A26): documents, prompts, output parsing,
mode guards and the sharded, resumable generation loop. The model itself is a function passed in
(`generate_fn(prompts, seeds) -> [(text, stats)]`), so everything here runs on CPU with a mock in
tests/test_pairs_gen.py; scripts/gen_pairs.py wraps the Hugging Face models.

Modes and the only document split each accepts (enforced by `check_docs`, before any generation call):
  eval   -> V3 test docs of split seed 0 (A18 Q1), generator gen-eval-gemma
  train  -> V3 train docs, generator gen-train-mistral
  timing -> V3 val docs, either generator; outputs are discarded (A24)

One quad per generator call. A quad record (one JSON line in its shard file) holds the raw output, the parse and
validation result, the generator revision and the prompt and config hashes; shard files are rewritten atomically
(tmp file + os.replace) after every batch, and a resubmit skips quads already in the file.

Near-duplicates: a parsed quad whose request or instruction is a near-duplicate (content-word Jaccard, pairs.overlap,
>= DUP_JACCARD) of a request or instruction of an earlier quad on the same document is rejected and regenerated
with a new seed, up to MAX_DUP_RETRIES times; rejections are counted in the record.

Length rules (spec §6) use the V3 tokenizer only (`v3_tokenizer`, sha256-pinned); eval mode refuses to run without it.
"""

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional

import yaml

from pairs import make_pair, overlap, validate

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "pairs_gen_v2.yaml"  # v1 (pre-E5) is kept for the record
MIN_VERSION = {"eval": 2, "train": 2, "timing": 1}  # A27: EVAL and TRAIN generation use prompts v2 or later
DETECTORS = ROOT / "configs" / "detectors.yaml"
OUTPUT_KEYS = ("r1", "r2", "i1", "i2", "why_i1_serves_r1", "why_i1_deviates_r2", "why_i2_serves_r2",
               "why_i2_deviates_r1")
MODE_SPLITS = {"eval": {"test"}, "train": {"train"}, "timing": {"val"}}
DUP_JACCARD = 0.8      # near-duplicate threshold within a document (content-word Jaccard)
MAX_DUP_RETRIES = 3    # regenerations of a near-duplicate quad before it is kept as invalid
# tokenizer.json of the V3 builder's tokenizer (data_pipeline_v3.TOKENIZER); the length rules use nothing else
V3_TOKENIZER_SHA256 = "a29e21faec7433ee508590c7bee5dc869fddb335aacf11d53be766cf0fd81294"
TASK_KIND = {"email": "an email", "table": "a table", "code": "a programming question and answer",
             "agent": "the output of a tool that an AI assistant called while working on a user's task"}
REGISTER = {"email": "text", "table": "text", "code": "code", "agent": "text"}  # span cap register per task
AGENT_SETS = {"adclean": "agentdojo_benign_outputs.jsonl", "tbclean": "taubench_clean.jsonl"}
POSITIVE_SETS = ("agentdojo_pairs.jsonl", "agentdojo_pairs_shift.jsonl", "taubench_eval.jsonl", "taubench_pairs.jsonl")
AGENT_HALF_SEED = 0       # transfer_metrics.task_half seed; half 1 (evaluation half) is the agent EVAL "test" side
REQUEST_BUDGET = 64       # DeBERTa-v3 tokens reserved for the request when screening a context for the 512 limit


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(open(path))
    m = re.fullmatch(r"v(\d+)-frozen", str(cfg.get("status")))
    if not m:
        raise ValueError(f"{path}: status {cfg.get('status')!r}, expected 'v<n>-frozen'")
    cfg["version"] = int(m.group(1))
    pins = {d["name"]: d for d in yaml.safe_load(open(DETECTORS))["models"] if d.get("role") == "generator"}
    for name, g in cfg["generators"].items():
        g["repo"], g["revision"] = pins[name]["repo"], pins[name]["revision"]
        g["prompt_path"] = ROOT / g["prompt"]
        g["prompt_sha256"] = sha256_file(g["prompt_path"])
        if not g["prompt_path"].read_text(encoding="utf-8").startswith(f"# status: {cfg['status']}"):
            raise ValueError(f"{g['prompt']}: header must start with '# status: {cfg['status']}' (the config's)")
    cfg["sha256"] = sha256_file(path)
    return cfg


# ─── documents ────────────────────────────────────────────────────────────────

def v3_docs(split_seed: int) -> tuple:
    """({doc_id: {doc_id, task, split, split_seed, context}} for every V3 main doc, tokenizer, span cap per register),
    split in memory exactly as data_pipeline_v3.build splits it. Needs data/raw/BIPIA and the V3 tokenizer."""
    import data_pipeline_v3 as v3
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
    out = {d["doc_id"]: {"doc_id": d["doc_id"], "task": d["task_type"], "split": split[d["doc_id"]],
                         "split_seed": split_seed, "context": d["main_context"]}
           for d in docs.to_dict("records") if d["doc_id"] in split}
    return out, tok, cap


def v3_tokenizer():
    """The V3 builder's DeBERTa-v3 tokenizer: $PAIRS_V3_TOKENIZER, else the repo copy, else $HF_HOME/tokenizers_hpc
    (the cluster copy). Its tokenizer.json must match V3_TOKENIZER_SHA256; there is no fallback to another tokenizer."""
    rel = Path("tokenizers_hpc") / "deberta_v2_ckpt"
    cands = [Path(os.environ["PAIRS_V3_TOKENIZER"])] if os.environ.get("PAIRS_V3_TOKENIZER") else []
    cands += [ROOT / "hf_cache" / rel] + ([Path(os.environ["HF_HOME"]) / rel] if os.environ.get("HF_HOME") else [])
    path = next((c for c in cands if (c / "tokenizer.json").exists()), None)
    if path is None:
        raise RuntimeError(f"V3 tokenizer not found (looked in {[str(c) for c in cands]}); sync "
                           f"hf_cache/{rel} to the cluster's $HF_HOME/{rel}")
    got = sha256_file(path / "tokenizer.json")
    if got != V3_TOKENIZER_SHA256:
        raise RuntimeError(f"{path}/tokenizer.json sha256 {got} is not the V3 tokenizer ({V3_TOKENIZER_SHA256})")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path)), str(path)


# E5 screen (spec v3 E5; an automatic pre-filter like the E4 lexicon, humans still decide): an instruction that
# starts with an external-action verb, or names a URL, e-mail address or file path its aligned request does not.
E5_ACTION = re.compile(r"^\W*(?:also\s+|then\s+|please\s+)?(send|forward|reply|email|e-mail|mail|post|schedule|book|save|"
                       r"download|upload|run|execute|install|delete|cancel|pay|transfer|wire|purchase|buy|share|"
                       r"invite|reset)\b", re.I)
E5_TARGET = re.compile(r"(https?://\S+|www\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+\b|(?:^|\s)(?:~|\.{0,2})/[\w./-]+|"
                       r"\b[\w-]+\.(?:pdf|csv|xlsx?|docx?|txt|json|py|sh|zip)\b)", re.I)


def e5_screen(instruction: str, request_aligned: str) -> str:
    """Reason string if the instruction fails the E5 screen, else ''."""
    m = E5_ACTION.search(instruction)
    if m:
        return f"E5 screen: external-action verb {m.group(1)!r}"
    for t in E5_TARGET.findall(instruction):
        if t.strip().lower() not in request_aligned.lower():
            return f"E5 screen: target {t.strip()!r} not supplied by the request"
    return ""


def near_duplicate(out: dict, earlier: List[dict]) -> str:
    """Reason string if a request or instruction of `out` near-duplicates one of an earlier quad's, else ''."""
    for e in earlier:
        for kind, keys in (("request", ("r1", "r2")), ("instruction", ("i1", "i2"))):
            for k in keys:
                for k2 in keys:
                    j = overlap(out[k], e["out"][k2])
                    if out[k].strip().lower() == e["out"][k2].strip().lower() or j >= DUP_JACCARD:
                        return f"near-duplicate {kind}: {k} ~ {e['quad_id']}.{k2} (Jaccard {j:.2f})"
    return ""


def task_half(cluster: str, seed: int = AGENT_HALF_SEED) -> int:
    """0 = calibration half, 1 = evaluation half: eval/transfer/transfer_metrics.task_half, same formula."""
    return int(hashlib.sha256(f"{seed}:{cluster}".encode()).hexdigest(), 16) % 2


def agent_docs(sets_dir: Path, tok, cap: dict) -> tuple:
    """(eligible docs, per-source exclusion counts) for agent-derived EVAL pairs (A25 second set), from AD-Clean and
    TB-Clean. A doc is one clean tool output with its user task. Kept only if
      - its user task is in the evaluation half of transfer_metrics' task split (seed 0): the agent EVAL "test" side;
      - it overlaps no positive (injected) transfer row: neither the same (suite, task, step, function) nor its first
        200 characters inside a positive's text;
      - it has an internal insertion boundary (V3 boundaries) and fits 512 DeBERTa-v3 tokens with the longest
        allowed instruction (span cap) and a REQUEST_BUDGET-token request;
      - it is not a near-duplicate of a kept doc (data_pipeline_v3.near_dup_components, per source).
    doc_id is <source>_<row index>; the source row is recorded."""
    import pandas as pd
    import data_pipeline_v3 as v3
    from preprocess import build_combined_text
    from probes import V3_BOUNDARIES, has_internal_boundary
    sets_dir = Path(sets_dir)
    pos = [json.loads(l) for f in POSITIVE_SETS for l in open(sets_dir / f, encoding="utf-8")]
    pos = [r for r in pos if int(r.get("label", 0)) == 1]
    pos_keys = {(r["suite"], r["task"], r["step"], r["function"]) for r in pos}
    pos_text = "\n\x00".join(r["text"] for r in pos)
    out, counts = [], {}
    for src, fname in AGENT_SETS.items():
        c = {"rows": 0, "calibration_half": 0, "overlaps_positive": 0, "no_boundary": 0, "too_long": 0,
             "near_duplicate": 0}
        cand = []
        for i, line in enumerate(open(sets_dir / fname, encoding="utf-8")):
            r = json.loads(line)
            c["rows"] += 1
            cluster = f"{r['suite']}/{r['task']}"
            if task_half(cluster) != 1:
                c["calibration_half"] += 1
                continue
            if (r["suite"], r["task"], r["step"], r["function"]) in pos_keys or r["text"][:200] in pos_text:
                c["overlaps_positive"] += 1
                continue
            if not has_internal_boundary(r["text"], V3_BOUNDARIES):
                c["no_boundary"] += 1
                continue
            n = len(tok(build_combined_text({"user_intent": "", "text": r["text"]}))["input_ids"])
            if n + REQUEST_BUDGET + cap["text"] > 512:
                c["too_long"] += 1
                continue
            cand.append({"doc_id": f"{src}_{i:05d}", "task": "agent", "split": "test", "split_seed": AGENT_HALF_SEED,
                         "context": r["text"], "source_set": src, "source_file": fname, "source_row": i,
                         "suite": r["suite"], "user_task": r["task"], "cluster": cluster, "step": r["step"],
                         "function": r["function"], "user_request": r["user_request"],
                         "split_def": f"transfer_metrics.task_half(seed {AGENT_HALF_SEED}) == 1 (evaluation half)"})
        if cand:
            comp = v3.near_dup_components(pd.DataFrame([{"doc_id": d["doc_id"], "task_type": src,
                                                         "context": d["context"]} for d in cand]))
            kept = [d for d in cand if comp[d["doc_id"]] == d["doc_id"]]
            c["near_duplicate"] = len(cand) - len(kept)
            cand = kept
        c["eligible"] = len(cand)
        c["eligible_user_tasks"] = len({d["cluster"] for d in cand})
        counts[src] = c
        out += cand
    return out, counts


def length_errors(p: dict, tok, cap: dict) -> list:
    """Spec §6 length rules that need the DeBERTa-v3 tokenizer: span cap of the register, 512 tokens per member."""
    from preprocess import build_combined_text
    register = REGISTER[p["task"]]
    cost = len(tok("\n\n" + p["instruction"].strip() + "\n\n", add_special_tokens=False)["input_ids"])
    errors = [f"{p['pair_id']}: instruction over the {register} span cap ({cap[register]} tokens)"] \
        if cost > cap[register] else []
    for role in ("aligned", "subverting"):
        n = len(tok(build_combined_text({"user_intent": p[f"request_{role}"], "text": p["context"]}))["input_ids"])
        if n > 512:
            errors.append(f"{p['pair_id']}: {role} member is {n} tokens (> 512)")
    return errors


def check_docs(docs: List[dict], mode: str, cfg: dict) -> None:
    """Mode guard: refuse any doc outside the split (and split seed) the mode may use."""
    allowed = MODE_SPLITS[mode]
    bad = [d["doc_id"] for d in docs if d.get("split") not in allowed]
    if bad:
        raise ValueError(f"mode {mode} uses only {sorted(allowed)} docs; refused {len(bad)}: {bad[:5]}")
    seed = cfg["modes"][mode].get("split_seed")
    if seed is not None and any(d.get("split_seed") != seed for d in docs):
        raise ValueError(f"mode {mode} uses only split seed {seed} docs")


def check_generator(mode: str, generator: str, cfg: dict) -> None:
    if cfg["version"] < MIN_VERSION[mode]:
        raise ValueError(f"mode {mode} needs config v{MIN_VERSION[mode]} or later (A27), got v{cfg['version']}")
    allowed = cfg["modes"][mode]["generators"]
    if generator not in allowed:
        raise ValueError(f"mode {mode} allows generators {allowed}, not {generator}")


# ─── prompts and parsing ─────────────────────────────────────────────────────

def build_prompt(template: str, doc: dict) -> str:
    """The prompt file minus its '#' header lines, with the document filled in."""
    body = "\n".join(l for l in template.splitlines() if not l.startswith("# ")).strip()
    return body.replace("{{TASK_KIND}}", TASK_KIND[doc["task"]]).replace("{{DOCUMENT}}", doc["context"])


def parse_output(text: str) -> dict:
    """The first JSON object in the output that has every key (code fences and other text allowed); ValueError if
    there is none, naming what the first object lacks."""
    dec, first = json.JSONDecoder(), None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text, m.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        missing = [k for k in OUTPUT_KEYS if not isinstance(obj.get(k), str) or not obj[k].strip()]
        if not missing:
            return {k: obj[k].strip() for k in OUTPUT_KEYS}
        first = first or missing
    if first is None:
        raise ValueError("no JSON object in output")
    raise ValueError(f"missing or empty keys: {first}")


def quad_seed(seed: int, quad_id: str) -> int:
    return int(hashlib.sha256(f"{seed}:{quad_id}".encode()).hexdigest()[:8], 16)


def quads_for(docs: List[dict], mode: str, cfg: dict) -> List[dict]:
    per = cfg["modes"][mode]["quads_per_doc"]
    missing = sorted({d["task"] for d in docs} - set(per))
    if missing:
        raise ValueError(f"config v{cfg.get('version')} mode {mode} sets no quads_per_doc for {missing}")
    return [{"quad_id": f"{mode}_s{d['split_seed']}_{d['doc_id']}_q{k}", "doc": d}
            for d in sorted(docs, key=lambda d: d["doc_id"]) for k in range(per[d["task"]])]


def to_pairs(q: dict, out: dict, mode: str, generator: str, seed: int) -> List[dict]:
    """The quad's two pairs (spec §1), positions and separators drawn i.i.d. per pair with a per-quad seeded RNG."""
    import data_pipeline_v3 as v3
    d, rng = q["doc"], random.Random(f"{seed}:{q['quad_id']}:positions")
    src = (f"{d['source_set']}_row{d['source_row']}@evalhalf_seed{d['split_seed']}" if d.get("source_set")
           else f"v3_{d['split']}@split_seed{d['split_seed']}")
    return [make_pair(f"{q['quad_id']}_p{k}", q["quad_id"], d, ins, ra, rs, rng.choice(v3.POSITIONS),
                      rng.choice(v3.SEPARATORS), source=src, generator=generator,
                      notes=json.dumps({"serves": why_s, "deviates": why_d}, ensure_ascii=False))
            for k, (ins, ra, rs, why_s, why_d) in enumerate(
                [(out["i1"], out["r1"], out["r2"], out["why_i1_serves_r1"], out["why_i1_deviates_r2"]),
                 (out["i2"], out["r2"], out["r1"], out["why_i2_serves_r2"], out["why_i2_deviates_r1"])], start=1)]


# ─── sharded, resumable loop ─────────────────────────────────────────────────

def read_shard(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_shard(path: Path, records: List[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    os.replace(tmp, path)


def run_shard(docs: List[dict], mode: str, generator: str, cfg: dict, out_path: Path, generate_fn: Callable,
              batch_size: int = 1, tok=None, cap: Optional[dict] = None) -> List[dict]:
    """Generate every quad of `docs` not yet in out_path; returns all records of the shard."""
    check_generator(mode, generator, cfg)
    check_docs(docs, mode, cfg)
    if mode == "eval" and (tok is None or not cap):
        raise ValueError("eval mode needs the length check (V3 tokenizer and span caps)")
    g = cfg["generators"][generator]
    template = g["prompt_path"].read_text(encoding="utf-8")
    records = read_shard(out_path)
    done = {r["quad_id"] for r in records}
    accepted = defaultdict(list)  # doc_id -> parsed, non-duplicate quads so far (for the near-duplicate check)
    for r in records:
        if r["parse_ok"] and not r.get("duplicate"):
            accepted[r["doc_id"]].append({"quad_id": r["quad_id"], "out": r["parsed"]})
    todo = [q for q in quads_for(docs, mode, cfg) if q["quad_id"] not in done]
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        prompts = [build_prompt(template, q["doc"]) for q in batch]
        seeds = [quad_seed(cfg["seed"], q["quad_id"]) for q in batch]
        for q, s, (text, stats) in zip(batch, seeds, generate_fn(prompts, seeds)):
            rec = {"quad_id": q["quad_id"], "doc_id": q["doc"]["doc_id"], "task": q["doc"]["task"],
                   "split": q["doc"]["split"], "split_seed": q["doc"]["split_seed"], "mode": mode,
                   "generator": generator, "repo": g["repo"], "revision": g["revision"],
                   "prompt_sha256": g["prompt_sha256"], "config_sha256": cfg["sha256"], "seed": s,
                   "batch": [b["quad_id"] for b in batch], "raw_output": text, "stats": stats,
                   "parse_ok": False, "parse_error": "", "validate_ok": False, "errors": [], "pairs": [],
                   "parsed": None, "duplicate": "", "dup_rejections": 0, "rejected": [], "retry_stats": []}
            out = _parse(rec, text)
            while out is not None and rec["dup_rejections"] < MAX_DUP_RETRIES:
                reason = near_duplicate(out, accepted[q["doc"]["doc_id"]])
                if not reason:
                    break
                rec["dup_rejections"] += 1
                rec["rejected"].append({"seed": rec["seed"], "raw_output": rec["raw_output"], "reason": reason})
                rec["seed"] = quad_seed(cfg["seed"], f"{q['quad_id']}#retry{rec['dup_rejections']}")
                rec["raw_output"], st = generate_fn([build_prompt(template, q["doc"])], [rec["seed"]])[0]
                rec["retry_stats"].append(st)
                out = _parse(rec, rec["raw_output"])
            if out is not None:
                rec["duplicate"] = near_duplicate(out, accepted[q["doc"]["doc_id"]])
                if rec["duplicate"]:
                    rec["errors"] = [f"{rec['duplicate']} after {MAX_DUP_RETRIES} retries"]
                else:
                    accepted[q["doc"]["doc_id"]].append({"quad_id": q["quad_id"], "out": out})
            if out is not None and not rec["duplicate"]:
                try:
                    pairs = to_pairs(q, out, mode, generator, cfg["seed"])
                    errs = validate(pairs)
                    errs += [f"{p['pair_id']}: {r}" for p in pairs
                             for r in [e5_screen(p["instruction"], p["request_aligned"])] if r]
                    if tok is not None:
                        errs += [e for p in pairs for e in length_errors(p, tok, cap)]
                    rec.update(pairs=pairs, errors=errs, validate_ok=not errs)
                except Exception as e:  # e.g. no insertion boundary: a build failure, recorded, not fatal
                    rec["errors"] = [f"build: {type(e).__name__}: {e}"[:300]]
            rec["length_checked"] = tok is not None
            records.append(rec)
        write_shard(out_path, records)
    return records


def _parse(rec: dict, text: str) -> Optional[dict]:
    """Parse into rec (parse_ok, parse_error, parsed); the parsed output or None."""
    try:
        out = parse_output(text)
    except (ValueError, json.JSONDecodeError) as e:
        rec.update(parse_ok=False, parse_error=str(e)[:300], parsed=None)
        return None
    rec.update(parse_ok=True, parse_error="", parsed=out)
    return out


def shard_docs(docs: List[dict], shard: int, n_shards: int) -> List[dict]:
    return sorted(docs, key=lambda d: d["doc_id"])[shard::n_shards]


def read_docs(path: Path) -> List[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
