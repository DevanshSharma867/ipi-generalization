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
  selfcheck  spec §8 TRAIN-side filter (A32; config v5+): the TRAIN generator labels each member of each valid pair
           blind (prefill only, A vs B) and a pair is kept only if both members get their intended label.
           --in_dir: a finished train/val generate dir -> <in_dir>/selfcheck.jsonl (resumable) + selfcheck.meta.json.
           --failures: dry run on a timing run's failed attempts (val docs) -> results/pairs/selfcheck_dryrun/<run_id>/.
             python scripts/gen_pairs.py selfcheck --in_dir data/pairs/val_gen/<dir>
             python scripts/gen_pairs.py selfcheck --failures results/pairs/gen_timing/<run>/failures.jsonl
"""

import argparse
from collections import defaultdict
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
    mode_split = {"timing": "val", "eval": "test", "train": "train", "val": "val"}[a.mode]
    docs, _, cap = pg.v3_docs(a.split_seed)
    pool = sorted((d for d in docs.values() if d["split"] == mode_split), key=lambda d: d["doc_id"])
    if a.n_per_task:
        want = dict(kv.split("=") for kv in a.n_per_task.split(","))
        rng = random.Random(f"pairs_docs:{a.mode}:{a.split_seed}")
        pool = sorted([d for t, n in want.items() for d in rng.sample([d for d in pool if d["task"] == t], int(n))],
                      key=lambda d: d["doc_id"])
    frozen_dir = ROOT / "data" / "v3" / f"split_seed_{a.split_seed}"
    frozen = "absent"
    if frozen_dir.is_dir():  # since v3-frozen: the in-memory split must equal the frozen build's split
        import pandas as pd
        fz = pd.concat([pd.read_csv(f, usecols=["doc_id", "split"]) for f in sorted(frozen_dir.glob("*.csv"))])
        want = set(fz.loc[fz["split"] == mode_split, "doc_id"])
        got = {d["doc_id"] for d in docs.values() if d["split"] == mode_split}
        if got - want or (fz.loc[fz["split"] != mode_split, "doc_id"].isin(got)).any():
            sys.exit(f"{mode_split} docs differ from {frozen_dir}: {sorted(got ^ want)[:10]}")
        frozen = (f"{frozen_dir.relative_to(ROOT)}: same {mode_split} docs ({len(got)}; "
                  f"{len(want - got)} frozen {mode_split} docs have no main context, long slice only)")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / f"{a.mode}_s{a.split_seed}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for d in pool:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    manifest = {"docs": out.name, "sha256": pg.sha256_file(out), "mode": a.mode, "split": mode_split,
                "split_seed": a.split_seed, "n_per_task": a.n_per_task, "span_cap_tokens": cap,
                "counts": {t: sum(d["task"] == t for d in pool) for t in ("email", "table", "code")},
                "builder": "src/data_pipeline_v3.py (in memory, pairs_gen.v3_docs)", "frozen_check": frozen}
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}: {manifest['counts']}")


def load_hf(g: dict, device: str = "cuda"):
    """(tokenizer, model, load stats, peak-memory fn) for one generator dict (repo, revision, precision,
    model_class); left padding. On CUDA the process is capped at half the card (A24)."""
    import torch
    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig
    cuda = device == "cuda"
    # fp16 (pairs_gen_v6, Kaggle T4s): whole cards, the model spread over every visible GPU; otherwise the
    # cluster's half-card cap (A24) on one GPU
    multi = cuda and g["precision"] == "fp16"
    if cuda:
        for i in range(torch.cuda.device_count() if multi else 1):
            if not multi:
                torch.cuda.set_per_process_memory_fraction(0.5)
            torch.cuda.reset_peak_memory_stats(i)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(g["repo"], revision=g["revision"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = dict(revision=g["revision"], device_map="auto" if multi else {"": 0 if cuda else "cpu"},  # MPS: CPU, then move
              dtype={"nf4": torch.bfloat16, "bf16": torch.bfloat16, "fp16": torch.float16,
                     "fp32": torch.float32}[g["precision"]])
    if g["precision"] == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    cls = {"image-text-to-text": transformers.AutoModelForImageTextToText,
           "causal-lm": transformers.AutoModelForCausalLM}[g["model_class"]]
    model = cls.from_pretrained(g["repo"], **kw).eval()
    if not cuda:
        model = model.to(device)
    ndev = torch.cuda.device_count() if multi else 1
    peak = (lambda: round(sum(torch.cuda.max_memory_allocated(i) for i in range(ndev)) / 1e9, 2)) if cuda \
        else (lambda: None)
    load = {"load_s": round(time.time() - t0, 2), "vram_after_load_gb": peak()}
    return tok, model, load, peak


def hf_generator(g: dict, device: str = "cuda"):
    """(generate_fn, load stats, peak-memory fn) for one generator dict (load_hf plus decoding); batches with left
    padding, one seed per call."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    cuda = device == "cuda"
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    tok, model, load, peak = load_hf(g, device)
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
    generate_fn, load, peak = hf_generator(g, a.device)
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


# ─── self-consistency filter (spec §8; A32) ─────────────────────────────────

def hf_scorer(g: dict, device: str = "cuda"):
    """(score_fn, load stats, peak-memory fn): score_fn(prompts) -> [(logp "A", logp "B")] of the next token after
    the chat template's generation prompt. Prefill only, no sampling; each letter's logp sums its token variants."""
    import torch
    tok, model, load, peak = load_hf(g, device)
    ids = {L: sorted({i for t in (L, "\u2581" + L) for i in [tok.convert_tokens_to_ids(t)]
                      if isinstance(i, int) and i != tok.unk_token_id}) for L in pg.SELFCHECK_LETTERS}
    if not all(ids.values()):
        raise RuntimeError(f"no vocabulary token for a label letter: {ids}")

    def score_fn(prompts):
        chats = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False)
                 for p in prompts]
        enc = tok(chats, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            lp = torch.log_softmax(model(**enc, logits_to_keep=1).logits[:, -1].float(), dim=-1)
        return [tuple(float(torch.logsumexp(lp[i, ids[L]], 0)) for L in pg.SELFCHECK_LETTERS)
                for i in range(len(prompts))]
    return score_fn, {**load, "letter_token_ids": ids}, peak


def selfcheck_score(pairs: list, score_fn, template: str, batch_size: int, out_path: Path, extra=None) -> list:
    """Decisions for `pairs` (pairs_gen.selfcheck_decide), appended to out_path (JSONL, no text) after every batch;
    pair_ids already in out_path are skipped, so a rerun resumes."""
    done = {d["pair_id"]: d for d in pg.read_shard(out_path)}
    todo = [p for p in pairs if p["pair_id"] not in done]
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        members = [m for p in chunk for m in pg.selfcheck_members(p)]
        scores = score_fn([pg.selfcheck_prompt(template, m) for m in members])
        with open(out_path, "a", encoding="utf-8") as f:
            for k, p in enumerate(chunk):
                d = {**pg.selfcheck_decide(p, scores[2 * k:2 * k + 2]), **((extra or {}).get(p["pair_id"], {}))}
                done[p["pair_id"]] = d
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return [done[p["pair_id"]] for p in pairs]


def generated_pairs(in_dir: Path, cfg: dict) -> tuple:
    """(valid pairs, mode, docs sha256) of a finished generate output dir. Refuses a mode the filter does not
    cover (eval: humans label every EVAL pair) and an unfinished dir (fewer quads than its docs file asks for)."""
    recs = [r for p in sorted(in_dir.glob("shard_*.jsonl")) for r in pg.read_shard(p)]
    metas = [json.loads(p.read_text()) for p in sorted(in_dir.glob("shard_*.meta.json"))]
    if not recs or not metas:
        raise ValueError(f"{in_dir}: no shard records or meta")
    modes = {r["mode"] for r in recs} | {m["mode"] for m in metas}
    if len(modes) != 1 or next(iter(modes)) not in cfg["selfcheck"]["modes"]:
        raise ValueError(f"{in_dir}: modes {sorted(modes)}; the self-check covers {cfg['selfcheck']['modes']} only")
    mode = next(iter(modes))
    docs_files = {(m["docs"], m["docs_sha256"]) for m in metas}
    if len(docs_files) != 1:
        raise ValueError(f"{in_dir}: shards from different docs files {docs_files}")
    docs_path, docs_sha = next(iter(docs_files))
    want = {q["quad_id"] for q in pg.quads_for(pg.read_docs(ROOT / docs_path), mode, cfg)}
    have = {r["quad_id"] for r in recs}
    if want != have:
        raise ValueError(f"{in_dir}: {len(want - have)} quads not generated yet, {len(have - want)} unexpected")
    return [p for r in recs if r["validate_ok"] for p in r["pairs"]], mode, docs_sha


def failure_pairs(rows: list, docs: dict, cfg: dict) -> tuple:
    """Dry run input: pairs rebuilt from the failed attempts of a timing run (val docs only), one id per attempt;
    plus counts of attempts that do not parse or build."""
    pairs, extra, skipped = [], {}, defaultdict(int)
    for r in rows:
        if r["split"] != "val" or not r["quad_id"].startswith("timing_"):
            raise ValueError(f"{r['quad_id']}: the dry run uses timing failures on val docs only")
        out = pg._parse({}, r["raw_output"])
        if out is None:
            skipped[f"{r['generator']}: parse"] += 1
            continue
        try:
            built = pg.to_pairs({"quad_id": r["quad_id"], "doc": docs[r["doc_id"]]}, out, "timing", r["generator"],
                                cfg["seed"])
        except Exception:  # no insertion boundary for this instruction
            skipped[f"{r['generator']}: build"] += 1
            continue
        for p in built:
            p["pair_id"] = f"{p['pair_id']}#{r['run']}#a{r['attempt']}"
            pairs.append(p)
            extra[p["pair_id"]] = {"run": r["run"], "generator": r["generator"], "attempt": r["attempt"],
                                   "failed_stage": r["stage"], "final_validate_ok": r["final_validate_ok"]}
    return pairs, extra, dict(skipped)


def cmd_selfcheck(a):
    cfg = pg.load_config(a.config)
    sc = cfg["selfcheck"]
    g = cfg["generators"][sc["generator"]]
    template = sc["prompt_path"].read_text(encoding="utf-8")
    info = {"generator": sc["generator"], "repo": sc["repo"], "revision": sc["revision"],
            "prompt": sc["prompt"], "prompt_sha256": sc["prompt_sha256"], "config_sha256": cfg["sha256"]}
    if a.failures:  # dry run on a timing run's failed attempts (val docs); labels and counts only, no text
        rows = [json.loads(l) for l in open(a.failures, encoding="utf-8") if l.strip()]
        docs = {d["doc_id"]: d for d in pg.read_docs(a.docs)}
        pairs, extra, skipped = failure_pairs(rows, docs, cfg)
        run = start_run(version="pairs", exp="selfcheck_dryrun", seed=0, args={**vars(a), **info},
                        data_version=pg.sha256_file(a.failures)[:12], data_files=[a.failures, a.docs],
                        allow_dirty=a.allow_dirty)
        score_fn, load, _ = hf_scorer(g, a.device)
        t0 = time.time()
        dec = selfcheck_score(pairs, score_fn, template, a.batch_size, run.dir / "decisions.jsonl", extra)
        by = {k: pg.selfcheck_summary([d for d in dec if d["run"] == k]) for k in sorted({d["run"] for d in dec})}
        metrics = {**info, **load, "score_s": round(time.time() - t0, 2), "attempt_rows": len(rows),
                   "not_scored": skipped, "all": pg.selfcheck_summary(dec), "by_run": by,
                   "note": "pairs rebuilt from FAILED generation attempts (the valid timing pairs were deleted by "
                           "design, A24): not representative of TRAIN pairs; dry run only, no threshold is set from it"}
        run.save_metrics(metrics)
        run.log(json.dumps(metrics["all"]))
        run.finish()
        print(json.dumps(metrics, indent=1))
        return
    in_dir = Path(a.in_dir)
    pairs, mode, docs_sha = generated_pairs(in_dir, cfg)
    score_fn, load, peak = hf_scorer(g, a.device)
    t0 = time.time()
    dec = selfcheck_score(pairs, score_fn, template, a.batch_size, in_dir / "selfcheck.jsonl")
    meta = {**info, **load, "mode": mode, "docs_sha256": docs_sha, "score_s": round(time.time() - t0, 2),
            "peak_vram_gb": peak(), **pg.selfcheck_summary(dec)}
    (in_dir / "selfcheck.meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta, indent=1))


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
    g.add_argument("--device", default="cuda", help="cuda (cluster, Kaggle); cpu only for a code-path dry run")
    t = add_run_args(sub.add_parser("timing"))
    t.add_argument("--run", action="append", required=True)
    c = add_run_args(sub.add_parser("selfcheck"))
    c.add_argument("--config", type=Path, default=pg.CONFIG, help="config with a selfcheck section (v5 or later)")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--in_dir", help="a finished train/val generate output dir; writes selfcheck.jsonl + .meta.json")
    src.add_argument("--failures", type=Path, help="dry run: a timing run's failures.jsonl (val docs only)")
    c.add_argument("--docs", type=Path, default=DOCS_DIR / "timing_s0.jsonl", help="docs file of --failures")
    c.add_argument("--batch_size", type=int, default=8, help="pairs per forward batch (two members each)")
    c.add_argument("--device", default="cuda")
    a = ap.parse_args()
    {"docs": cmd_docs, "generate": cmd_generate, "timing": cmd_timing, "selfcheck": cmd_selfcheck}[a.cmd](a)


if __name__ == "__main__":
    main()
