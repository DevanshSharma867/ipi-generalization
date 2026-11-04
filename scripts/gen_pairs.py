"""
gen_pairs.py
============
Intent-pair generation with the pinned generators (configs/pairs_gen_v1.yaml; src/pairs_gen.py; spec v2 §8).

  docs     build a docs file (JSONL: doc_id, task, split, split_seed, context) from the V3 builder, in memory.
           Needs data/raw/BIPIA and the V3 tokenizer (laptop). Writes data/pairs/docs/<name>.jsonl + .manifest.json.
             python scripts/gen_pairs.py docs --mode timing --split_seed 0 --n_per_task email=7,table=7,code=6
  generate one shard of quads with one generator (GPU). Resumable: quads already in the shard file are skipped.
             python scripts/gen_pairs.py generate --mode timing --generator gen-eval-gemma \
                 --docs data/pairs/docs/timing_s0.jsonl --out_dir <dir> [--shard 0 --n_shards 1] [--batch_size 1]
  timing   summarise generate runs (load time, prefill/decode tok/s, s/quad, peak VRAM, parse and validate
           rates, GPU-hour extrapolation) into a runlog run results/pairs/gen_timing/<run_id>/. No generated text
           is copied into the run.
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


def cmd_docs(a):
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


def hf_generator(g: dict):
    """(generate_fn, load stats) for one pinned generator; batch with left padding; per-batch seed."""
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    torch.cuda.set_per_process_memory_fraction(0.5)  # half a card (A24)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(g["repo"], revision=g["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=g["revision"], device_map={"": 0}, dtype=torch.bfloat16)
    if g["precision"] == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    cls = {"image-text-to-text": transformers.AutoModelForImageTextToText,
           "causal-lm": transformers.AutoModelForCausalLM}[g["model_class"]]
    model = cls.from_pretrained(g["repo"], **kw).eval()
    load = {"load_s": round(time.time() - t0, 2), "vram_after_load_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
    dec = g["decoding"]

    class FirstToken(StoppingCriteria):  # records when the first new token exists (prefill + 1 step)
        def __init__(self):
            self.t = None

        def __call__(self, input_ids, scores, **kw):
            if self.t is None:
                torch.cuda.synchronize()
                self.t = time.time()
            return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def generate_fn(prompts, seeds):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(0)
        torch.manual_seed(seeds[0])
        first = FirstToken()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**enc, do_sample=dec["do_sample"], temperature=dec["temperature"], top_p=dec["top_p"],
                                 max_new_tokens=dec["max_new_tokens"], pad_token_id=tok.pad_token_id,
                                 stopping_criteria=StoppingCriteriaList([first]))
        torch.cuda.synchronize()
        t1 = time.time()
        new = out[:, enc["input_ids"].shape[1]:]
        res = []
        for i in range(len(prompts)):
            n_new = int((new[i] != tok.pad_token_id).sum())
            res.append((tok.decode(new[i], skip_special_tokens=True),
                        {"prompt_tokens": int(enc["attention_mask"][i].sum()), "new_tokens": n_new,
                         "batch_size": len(prompts), "batch_prompt_tokens": int(enc["attention_mask"].sum()),
                         "batch_new_tokens": int((new != tok.pad_token_id).sum()),
                         "batch_s": round(t1 - t0, 3), "prefill_s": round(first.t - t0, 3),
                         "decode_s": round(t1 - first.t, 3), "hit_max_new_tokens": new.shape[1] >= dec["max_new_tokens"]}))
        return res

    return generate_fn, load, torch


def cmd_generate(a):
    cfg = pg.load_config()
    pg.check_generator(a.mode, a.generator, cfg)
    docs = pg.read_docs(a.docs)
    pg.check_docs(docs, a.mode, cfg)  # before loading any model
    manifest = json.loads(Path(a.docs).with_suffix(".manifest.json").read_text())
    docs = pg.shard_docs(docs, a.shard, a.n_shards)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard_{a.shard:03d}_of_{a.n_shards:03d}.jsonl"
    todo = {q["quad_id"] for q in pg.quads_for(docs, a.mode, cfg)} - {r["quad_id"] for r in pg.read_shard(out_path)}
    if not todo:
        print(f"{out_path}: complete, skipped")
        return
    g = cfg["generators"][a.generator]
    tok = None
    if a.length_tokenizer:
        import yaml
        from transformers import AutoTokenizer
        pin = next(d for d in yaml.safe_load(open(pg.DETECTORS))["models"] if d["name"] == a.length_tokenizer)
        tok = AutoTokenizer.from_pretrained(pin["repo"], revision=pin["revision"])
    generate_fn, load, torch = hf_generator(g)
    t0 = time.time()
    recs = pg.run_shard(docs, a.mode, a.generator, cfg, out_path, generate_fn, batch_size=a.batch_size, tok=tok,
                        cap=manifest["span_cap_tokens"])
    meta = {**load, "generator": a.generator, "repo": g["repo"], "revision": g["revision"],
            "prompt_sha256": g["prompt_sha256"], "config_sha256": cfg["sha256"], "docs": str(a.docs),
            "docs_sha256": manifest["sha256"], "mode": a.mode, "batch_size": a.batch_size, "n_quads": len(recs),
            "generated_this_call": len(todo), "generate_s": round(time.time() - t0, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "gpu": torch.cuda.get_device_name(0), "length_tokenizer": a.length_tokenizer}
    (out_dir / f"{out_path.stem}.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))


def summarise(out_dir: Path) -> dict:
    recs = [r for p in sorted(out_dir.glob("shard_*.jsonl")) for r in pg.read_shard(p)]
    metas = [json.loads(p.read_text()) for p in sorted(out_dir.glob("shard_*.meta.json"))]
    if not recs or not metas:
        raise ValueError(f"{out_dir}: no shard records or meta")
    m = metas[0]
    batches = {tuple(r["batch"]): r["stats"] for r in recs}  # one entry per generate call
    gen_s = sum(s["batch_s"] for s in batches.values())
    pre_s = sum(s["prefill_s"] for s in batches.values())
    dec_s = sum(s["decode_s"] for s in batches.values())
    n, n_parse, n_valid = len(recs), sum(r["parse_ok"] for r in recs), sum(r["validate_ok"] for r in recs)
    s_quad = gen_s / n
    target = TARGET_QUADS[m["generator"]]
    err_kinds = {}
    for r in recs:
        for e in ([r["parse_error"]] if r["parse_error"] else []) + r["errors"]:
            k = e.split(": ", 1)[-1].split("(")[0].strip()[:80]
            err_kinds[k] = err_kinds.get(k, 0) + 1
    return {
        "generator": m["generator"], "revision": m["revision"], "prompt_sha256": m["prompt_sha256"],
        "config_sha256": m["config_sha256"], "docs_sha256": m["docs_sha256"], "gpu": m["gpu"],
        "batch_size": m["batch_size"], "n_quads": n, "by_task": {t: sum(r["task"] == t for r in recs)
                                                                 for t in ("email", "table", "code")},
        "load_s": max(x["load_s"] for x in metas), "peak_vram_gb": max(x["peak_vram_gb"] for x in metas),
        "prefill_tok_s": round(sum(s["batch_prompt_tokens"] for s in batches.values()) / pre_s, 1),
        "decode_tok_s": round(sum(s["batch_new_tokens"] for s in batches.values()) / dec_s, 1),
        "s_per_quad": round(s_quad, 2),
        "mean_new_tokens": round(statistics.mean(r["stats"]["new_tokens"] for r in recs), 1),
        "n_hit_max_new_tokens": sum(r["stats"]["hit_max_new_tokens"] for r in recs),
        "json_parse_rate": round(n_parse / n, 4), "validate_pass_rate": round(n_valid / n, 4),
        "length_checked": all(r.get("length_checked") for r in recs),
        "error_kinds": dict(sorted(err_kinds.items(), key=lambda kv: -kv[1])),
        "target_quads": target,
        "gpu_hours_target": round(target * s_quad / 3600, 2),
        "gpu_hours_target_per_valid": round(target * s_quad / 3600 / (n_valid / n), 2) if n_valid else None,
    }


def cmd_timing(a):
    runs = {Path(r).name: summarise(Path(r)) for r in a.run}
    run = start_run(version="pairs", exp="gen_timing", seed=0, args={"runs": list(runs)},
                    data_version=next(iter(runs.values()))["docs_sha256"][:12], allow_dirty=a.allow_dirty)
    for k, v in runs.items():
        run.log(f"{k}: {v['s_per_quad']} s/quad, parse {v['json_parse_rate']}, validate {v['validate_pass_rate']}, "
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
    g = sub.add_parser("generate")
    g.add_argument("--mode", required=True, choices=list(pg.MODE_SPLITS))
    g.add_argument("--generator", required=True)
    g.add_argument("--docs", required=True, type=Path)
    g.add_argument("--out_dir", required=True)
    g.add_argument("--shard", type=int, default=0)
    g.add_argument("--n_shards", type=int, default=1)
    g.add_argument("--batch_size", type=int, default=1)
    g.add_argument("--length_tokenizer", default="", help="detectors.yaml name of the tokenizer for the §6 length rules (deberta-v3-base)")
    t = add_run_args(sub.add_parser("timing"))
    t.add_argument("--run", action="append", required=True)
    a = ap.parse_args()
    {"docs": cmd_docs, "generate": cmd_generate, "timing": cmd_timing}[a.cmd](a)


if __name__ == "__main__":
    main()
