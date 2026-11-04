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
CONFIG = ROOT / "configs" / "pairs_gen_v1.yaml"
DETECTORS = ROOT / "configs" / "detectors.yaml"
OUTPUT_KEYS = ("r1", "r2", "i1", "i2", "why_i1_serves_r1", "why_i1_deviates_r2", "why_i2_serves_r2",
               "why_i2_deviates_r1")
MODE_SPLITS = {"eval": {"test"}, "train": {"train"}, "timing": {"val"}}
DUP_JACCARD = 0.8      # near-duplicate threshold within a document (content-word Jaccard)
MAX_DUP_RETRIES = 3    # regenerations of a near-duplicate quad before it is kept as invalid
# tokenizer.json of the V3 builder's tokenizer (data_pipeline_v3.TOKENIZER); the length rules use nothing else
V3_TOKENIZER_SHA256 = "a29e21faec7433ee508590c7bee5dc869fddb335aacf11d53be766cf0fd81294"
TASK_KIND = {"email": "an email", "table": "a table", "code": "a programming question and answer"}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path: Path = CONFIG) -> dict:
    cfg = yaml.safe_load(open(path))
    if cfg.get("status") != "v1-frozen":
        raise ValueError(f"{path}: status {cfg.get('status')!r}, expected 'v1-frozen'")
    pins = {d["name"]: d for d in yaml.safe_load(open(DETECTORS))["models"] if d.get("role") == "generator"}
    for name, g in cfg["generators"].items():
        g["repo"], g["revision"] = pins[name]["repo"], pins[name]["revision"]
        g["prompt_path"] = ROOT / g["prompt"]
        g["prompt_sha256"] = sha256_file(g["prompt_path"])
        if not g["prompt_path"].read_text(encoding="utf-8").startswith("# status: v1-frozen"):
            raise ValueError(f"{g['prompt']}: header must start with '# status: v1-frozen'")
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


def length_errors(p: dict, tok, cap: dict) -> list:
    """Spec §6 length rules that need the DeBERTa-v3 tokenizer: span cap of the register, 512 tokens per member."""
    from preprocess import build_combined_text
    register = {"email": "text", "table": "text", "code": "code"}[p["task"]]
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
    return [{"quad_id": f"{mode}_s{d['split_seed']}_{d['doc_id']}_q{k}", "doc": d}
            for d in sorted(docs, key=lambda d: d["doc_id"]) for k in range(per[d["task"]])]


def to_pairs(q: dict, out: dict, mode: str, generator: str, seed: int) -> List[dict]:
    """The quad's two pairs (spec §1), positions and separators drawn i.i.d. per pair with a per-quad seeded RNG."""
    import data_pipeline_v3 as v3
    d, rng = q["doc"], random.Random(f"{seed}:{q['quad_id']}:positions")
    src = f"v3_{d['split']}@split_seed{d['split_seed']}"
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
