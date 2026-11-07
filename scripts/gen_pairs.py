"""
gen_pairs.py
============
Intent-pair generation with the pinned generators (configs/pairs_gen_v4.yaml, the default; v1-v3 kept for the record; src/pairs_gen.py; spec v3 §8).

  docs     build a docs file (JSONL: doc_id, task, split, split_seed, context) from the V3 builder, in memory.
           Needs data/raw/BIPIA and the V3 tokenizer (laptop). Writes data/pairs/docs/<name>.jsonl + .manifest.json.
             python scripts/gen_pairs.py docs --mode timing --split_seed 0 --n_per_task email=7,table=7,code=6
  generate one shard of quads with one generator (GPU). Resumable: quads already in the shard file are skipped.
             python scripts/gen_pairs.py generate --mode timing --generator gen-eval-gemma \
                 --docs data/pairs/docs/timing_s0.jsonl --out_dir <dir> [--shard 0 --n_shards 1] [--batch_size 1]
  timing   summarise generate runs (load time, prefill/decode tok/s, s/quad, peak VRAM, parse and validate
           rates, retries, GPU-hour extrapolation) into a runlog run results/pairs/gen_timing/<run_id>/. Only the
           raw text of failed attempts is copied (failures.jsonl + manifest.json; val docs only, A29).
             python scripts/gen_pairs.py timing --run <out_dir> [--run <out_dir> ...]
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pairs_gen as pg
from runlog import add_run_args, start_run

DOCS_DIR = ROOT / "data" / "pairs" / "docs"
TARGET_QUADS = {"gen-eval-gemma": 210, "gen-train-mistral": 3141}  # spec §8 planning volumes (A18)


def cmd_agent_docs(a):
    """Agent-derived EVAL docs (AD-Clean, TB-Clean evaluation half). The docs JSONL holds transfer-set text, which
    is not redistributed, so it goes under data/transfer/ (gitignored); the manifest (counts, hashes) is committed."""
    tok, tok_path = pg.v3_tokenizer()
    cap = {"text": a.span_cap_text, "code": 0}
    docs, counts = pg.agent_docs(a.sets_dir, tok, cap)
    out = ROOT / "data" / "transfer" / "pairs_docs" / f"eval_agent_s{pg.AGENT_HALF_SEED}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    sets_manifest = json.loads((Path(a.sets_dir) / "manifest.json").read_text())
    manifest = {"docs": str(out.relative_to(ROOT)), "sha256": pg.sha256_file(out), "mode": "eval", "split": "test",
                "split_def": f"transfer_metrics.task_half(seed {pg.AGENT_HALF_SEED}) == 1", "split_seed": pg.AGENT_HALF_SEED,
                "sets_data_version": sets_manifest.get("data_version"), "span_cap_tokens": {"text": a.span_cap_text},
                "request_budget_tokens": pg.REQUEST_BUDGET, "length_tokenizer": tok_path, "counts": counts,
                "doc_ids": [d["doc_id"] for d in docs]}
    m = DOCS_DIR / f"eval_agent_s{pg.AGENT_HALF_SEED}.manifest.json"
    m.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(counts, indent=1))
    print(f"wrote {out.relative_to(ROOT)} ({len(docs)} docs) and {m.relative_to(ROOT)}")


def cmd_docs(a):
    if a.source == "agent":
        return cmd_agent_docs(a)
    mode_split = {"timing": "val", "eval": "test", "train": "train"}[a.mode]
    docs, _, cap = pg.v3_docs(a.split_seed)
    pool = sorted((d for d in docs.values() if d["split"] == mode_split), key=lambda d: d["doc_id"])
    if a.n_per_task:
        want = dict(kv.split("=") for kv in a.n_per_task.split(","))
        rng = random.Random(f"pairs_docs:{a.mode}:{a.split_seed}")
        pool = sorted([d for t, n in want.items() for d in rng.sample([d for d in pool if d["task"] == t], int(n))],
                      key=lambda d: d["doc_id"])
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / f"{a.mode}_s{a.split_seed}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest = {"docs": out.name, "sha256": pg.sha256_file(out), "mode": a.mode, "split": mode_split,
                "split_seed": a.split_seed, "n_per_task": a.n_per_task, "span_cap_tokens": cap,
                "counts": {t: sum(d["task"] == t for d in pool) for t in ("email", "table", "code")},
                "builder": "src/data_pipeline_v3.py (in memory, pairs_gen.v3_docs); pre-freeze V3 split"}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {manifest['counts']}")


def hf_generator(g: dict, device: str = "cuda"):
    """(generate_fn, load stats, peak-memory fn) for one generator dict (repo, revision, precision, model_class,
    decoding); batches with left padding, one seed per call. On CUDA the process is capped at half the card (A24)."""
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    cuda = device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    if cuda:
        torch.cuda.set_per_process_memory_fraction(0.5)
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(g["repo"], revision=g["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=g["revision"], device_map={"": 0 if cuda else "cpu"},  # MPS: load on CPU, then move
              dtype={"nf4": torch.bfloat16, "bf16": torch.bfloat16, "fp32": torch.float32}[g["precision"]])
    if g["precision"] == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    cls = {"image-text-to-text": transformers.AutoModelForImageTextToText,
           "causal-lm": transformers.AutoModelForCausalLM}[g["model_class"]]
    model = cls.from_pretrained(g["repo"], **kw).eval()
    if not cuda:
        model = model.to(device)
    peak = (lambda: round(torch.cuda.max_memory_allocated() / 1e9, 2)) if cuda else (lambda: None)
    load = {"load_s": round(time.time() - t0, 2), "vram_after_load_gb": peak()}
    dec = g["decoding"]

    class FirstToken(StoppingCriteria):  # records when the first new token exists (prefill + 1 step)
        def __init__(self):
            self.t = None

        def __call__(self, input_ids, scores, **kw):
            if self.t is None:
                sync()
                self.t = time.time()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def generate_fn(prompts, seeds):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        torch.manual_seed(seeds[0])
        first = FirstToken()
        sync()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=dec["do_sample"], temperature=dec["temperature"], top_p=dec["top_p"],
                                 max_new_tokens=dec["max_new_tokens"], pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([first]))
        sync()
        t1 = time.time()
        new = out[:, enc["input_ids"].shape[1]:]
        res = []
        for i in range(len(prompts)):
            res.append((tok.decode(new[i], skip_special_tokens=True),
                        {"prompt_tokens": int(enc["attention_mask"][i].sum()),
                         "new_tokens": int((new[i] != tok.pad_token_id).sum()),
                         "batch_size": len(prompts), "batch_prompt_tokens": int(enc["attention_mask"].sum()),
                         "batch_new_tokens": int((new != tok.pad_token_id).sum()),
                         "batch_s": round(t1 - t0, 3), "prefill_s": round(first.t - t0, 3),
                         "decode_s": round(t1 - first.t, 3), "hit_max_new_tokens": new.shape[1] >= dec["max_new_tokens"]}))
        return res

    return generate_fn, load, peak


def cmd_generate(a):
    cfg = pg.load_config(a.config)
    pg.check_generator(a.mode, a.generator, cfg)
    docs = pg.read_docs(a.docs)
    pg.check_docs(docs, a.mode, cfg)  # before loading any model
    mpath = Path(a.docs).with_suffix(".manifest.json")
    if not mpath.exists():  # agent docs live under data/transfer/, their manifest under data/pairs/docs/
        mpath = DOCS_DIR / mpath.name
    manifest = json.loads(mpath.read_text())
    docs = pg.shard_docs(docs, a.shard, a.n_shards)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{a.shard:03d}_of_{a.n_shards:03d}.jsonl"
    todo = {q["quad_id"] for q in pg.quads_for(docs, a.mode, cfg)} - {r["quad_id"] for r in pg.read_shard(out_path)}
    if not todo:
        print(f"{out_path}: complete, skipped")
        return
    g = cfg["generators"][a.generator]
    if a.length_tokenizer:
        print(f"--length_tokenizer {a.length_tokenizer} ignored: the length rules use the V3 tokenizer only")
    if a.no_length_check and a.mode == "eval":
        sys.exit("eval mode always runs the length check")
    tok, tok_path = (None, "") if a.no_length_check else pg.v3_tokenizer()  # fails loudly if missing
    generate_fn, load, peak = hf_generator(g)
    t0 = time.time()
    recs = pg.run_shard(docs, a.mode, a.generator, cfg, out_path, generate_fn, batch_size=a.batch_size, tok=tok,
                        cap=manifest["span_cap_tokens"])
    meta = {**load, "generator": a.generator, "repo": g["repo"], "revision": g["revision"],
            "prompt_sha256": g["prompt_sha256"], "config_sha256": cfg["sha256"], "docs": str(a.docs),
            "docs_sha256": manifest["sha256"], "mode": a.mode, "batch_size": a.batch_size, "n_quads": len(recs),
            "generated_this_call": len(todo), "generate_s": round(time.time() - t0, 2),
            "peak_vram_gb": peak(), "gpu": __import__("torch").cuda.get_device_name(0),
            "length_tokenizer": tok_path}
    (out_dir / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def summarise(out_dir: Path) -> dict:
    recs = [r for p in sorted(out_dir.glob("shard_*.jsonl")) for r in pg.read_shard(p)]
    metas = [json.loads(p.read_text()) for p in sorted(out_dir.glob("shard_*.meta.json"))]
    if not recs or not metas:
        raise ValueError(f"{out_dir}: no shard records or meta")
    m = metas[0]
    batches = list({tuple(r["batch"]): r["stats"] for r in recs}.values())  # one entry per generate call
    batches += [s for r in recs for s in r.get("retry_stats", [])]          # regenerations (A29)
    gen_s = sum(s["batch_s"] for s in batches)
    pre_s = sum(s["prefill_s"] for s in batches)
    dec_s = sum(s["decode_s"] for s in batches)
    n, n_parse, n_valid = len(recs), sum(r["parse_ok"] for r in recs), sum(r["validate_ok"] for r in recs)
    s_quad = gen_s / n
    target = TARGET_QUADS[m["generator"]]
    err_kinds, attempt_kinds = {}, {}
    for r in recs:
        for e in ([r["parse_error"]] if r["parse_error"] else []) + r["errors"]:
            k = e.split(": ", 1)[-1].split("(")[0].strip()[:80]
            err_kinds[k] = err_kinds.get(k, 0) + 1
        for a in r.get("rejected", []):
            for e in a.get("reasons", [a.get("reason", "")]):
                k = e.split(": ", 1)[-1].split("(")[0].strip()[:80]
                attempt_kinds[k] = attempt_kinds.get(k, 0) + 1
    return {
        "generator": m["generator"], "revision": m["revision"], "prompt_sha256": m["prompt_sha256"],
        "config_sha256": m["config_sha256"], "docs_sha256": m["docs_sha256"], "gpu": m["gpu"],
        "batch_size": m["batch_size"], "n_quads": n, "by_task": {t: sum(r["task"] == t for r in recs)
                                                                 for t in ("email", "table", "code")},
        "load_s": max(x["load_s"] for x in metas), "peak_vram_gb": max(x["peak_vram_gb"] for x in metas),
        "prefill_tok_s": round(sum(s["batch_prompt_tokens"] for s in batches) / pre_s, 1),
        "decode_tok_s": round(sum(s["batch_new_tokens"] for s in batches) / dec_s, 1),
        "s_per_quad": round(s_quad, 2),
        "mean_new_tokens": round(statistics.mean(r["stats"]["new_tokens"] for r in recs), 1),
        "n_hit_max_new_tokens": sum(r["stats"]["hit_max_new_tokens"] for r in recs),
        "n_dup_rejections": sum(r.get("dup_rejections", 0) for r in recs),
        "n_duplicate_kept_invalid": sum(bool(r.get("duplicate")) for r in recs),
        "n_retries": sum(r.get("n_retries", 0) for r in recs),
        "first_attempt_pass_rate": round(sum(r.get("first_attempt_ok", False) for r in recs) / n, 4),
        "json_parse_rate": round(n_parse / n, 4), "validate_pass_rate": round(n_valid / n, 4),
        "length_checked": all(r.get("length_checked") for r in recs),
        "error_kinds": dict(sorted(err_kinds.items(), key=lambda kv: -kv[1])),           # final attempts
        "rejected_attempt_error_kinds": dict(sorted(attempt_kinds.items(), key=lambda kv: -kv[1])),
        "target_quads": target,
        "gpu_hours_target": round(target * s_quad / 3600, 2),
        "gpu_hours_target_per_valid": round(target * s_quad / 3600 / (n_valid / n), 2) if n_valid else None,
    }


def failures(out_dir: Path) -> list:
    """Raw text of every failed attempt (the final failure and each rejected retry) of a TIMING run. Timing runs
    use val docs only; refuses anything else, so EVAL and TRAIN text can never land in a results directory."""
    rows = []
    for r in (r for p in sorted(out_dir.glob("shard_*.jsonl")) for r in pg.read_shard(p)):
        if r["mode"] != "timing" or r["split"] != "val":
            raise ValueError(f"{r['quad_id']}: failures are kept for timing runs on val docs only")
        base = {"run": out_dir.name, "quad_id": r["quad_id"], "doc_id": r["doc_id"], "task": r["task"],
                "split": r["split"], "split_seed": r["split_seed"], "generator": r["generator"],
                "final_validate_ok": r["validate_ok"]}
        rows += [{**base, **a} for a in r.get("rejected", [])]
        if not r["validate_ok"]:
            rows.append({**base, "attempt": r.get("n_retries", 0), "seed": r["seed"],
                         "stage": "parse" if not r["parse_ok"] else "duplicate" if r["duplicate"] else "validate",
                         "reasons": [r["parse_error"]] if not r["parse_ok"] else r["errors"],
                         "raw_output": r["raw_output"]})
    return rows


def cmd_timing(a):
    runs = {Path(r).name: summarise(Path(r)) for r in a.run}
    fails = [f for r in a.run for f in failures(Path(r))]
    run = start_run(version="pairs", exp="gen_timing", seed=0, args={"runs": list(runs)},
                    data_version=next(iter(runs.values()))["docs_sha256"][:12], allow_dirty=a.allow_dirty)
    fpath = run.dir / "failures.jsonl"
    fpath.write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in fails), encoding="utf-8")
    (run.dir / "manifest.json").write_text(json.dumps({
        "failures.jsonl": {"rows": len(fails), "sha256": pg.sha256_file(fpath),
                           "content": "raw generator output of every failed attempt (final failures and rejected "
                                      "retries) of this timing run, with stage and reasons",
                           "docs": "V3 val docs of split seed 0 only (timing mode, enforced by gen_pairs.failures); "
                                   "never EVAL or TRAIN pairs, never used for training or scoring"}}, indent=1) + "\n")
    for k, v in runs.items():
        run.log(f"{k}: {v['s_per_quad']} s/quad, parse {v['json_parse_rate']}, validate {v['validate_pass_rate']} "
                f"(first attempt {v['first_attempt_pass_rate']}, {v['n_retries']} retries), "
                f"{v['gpu_hours_target']} GPU-h for {v['target_quads']} quads ({v['gpu_hours_target_per_valid']} per valid)")
    run.save_metrics({"runs": runs})
    run.finish()
    print(json.dumps(runs, indent=1))
    print(f"wrote {run.dir.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("docs")
    d.add_argument("--mode", required=True, choices=list(pg.MODE_SPLITS))
    d.add_argument("--split_seed", type=int, default=0)
    d.add_argument("--n_per_task", default="", help="e.g. email=7,table=7,code=6 (seeded sample); default all")
    d.add_argument("--source", default="v3", choices=["v3", "agent"], help="agent: AD-Clean/TB-Clean EVAL docs")
    d.add_argument("--sets_dir", type=Path, default=ROOT / "data" / "transfer" / "sets_adliu")
    d.add_argument("--span_cap_text", type=int, default=30, help="V3 text span cap (tokens; timing_s0 manifest)")
    g = sub.add_parser("generate")
    g.add_argument("--mode", required=True, choices=list(pg.MODE_SPLITS))
    g.add_argument("--generator", required=True)
    g.add_argument("--config", type=Path, default=pg.CONFIG, help="generation config (default: the current version)")
    g.add_argument("--docs", required=True, type=Path)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--n_shards", type=int, default=1)
    g.add_argument("--batch_size", type=int, default=1)
    g.add_argument("--length_tokenizer", default="", help="deprecated, ignored: the length rules use the V3 tokenizer")
    g.add_argument("--no_length_check", action="store_true", help="skip the §6 length rules (not allowed in eval)")
    t = add_run_args(sub.add_parser("timing"))
    t.add_argument("--run", action="append", required=True)
    a = ap.parse_args()
    {"docs": cmd_docs, "generate": cmd_generate, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
