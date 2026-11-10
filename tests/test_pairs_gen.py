"""CPU tests for the pair-generation pipeline (src/pairs_gen.py) with a mock generator."""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import pairs_gen as pg

EMAIL = "Hi David,\n\nYou sent a payment of $150.00 to Julia on 3 May.\n\nThanks for using our service.\nThe Team"
GOOD = {"r1": "How much money did I send, and to whom?",
        "r2": "Translate this whole email into Spanish for my accountant.",
        "i1": "State the payment amount together with the person who received it.",
        "i2": "Render the full message in Spanish for a bookkeeper.",
        "why_i1_serves_r1": "a", "why_i1_deviates_r2": "b", "why_i2_serves_r2": "c", "why_i2_deviates_r1": "d"}


def doc(i, split, task="email", split_seed=0):
    return {"doc_id": f"{task}_{i:04d}", "task": task, "split": split, "split_seed": split_seed, "context": EMAIL}


class Mock:
    def __init__(self, text=None):
        self.calls, self.text = [], text or json.dumps(GOOD)

    def __call__(self, prompts, seeds):
        self.calls.append(list(seeds))
        return [(self.text, {"prompt_tokens": 10, "new_tokens": 5, "batch_size": len(prompts),
                             "batch_prompt_tokens": 10 * len(prompts), "batch_new_tokens": 5 * len(prompts),
                             "batch_s": 1.0, "prefill_s": 0.2, "decode_s": 0.8, "hit_max_new_tokens": False})
                for _ in prompts]


@pytest.fixture(scope="module")
def cfg():
    return pg.load_config()


def body(path):
    return pg.build_prompt(Path(path).read_text(encoding="utf-8"), {"task": "email", "context": ""})


def test_prompts_share_no_word_8gram(cfg):
    grams = []
    for g in cfg["generators"].values():
        w = re.findall(r"[a-z0-9_]+", body(g["prompt_path"]).lower())
        grams.append({tuple(w[i:i + 8]) for i in range(len(w) - 7)})
    assert not grams[0] & grams[1]


def test_v1_config_refused_for_eval_and_train(tmp_path):
    v1 = pg.load_config(pg.ROOT / "configs" / "pairs_gen_v1.yaml")
    assert v1["version"] == 1
    with pytest.raises(ValueError, match="needs config and prompts v2"):
        pg.check_generator("eval", "gen-eval-gemma", v1)
    with pytest.raises(ValueError, match="needs config and prompts v2"):
        pg.check_generator("train", "gen-train-mistral", v1)
    v2 = pg.load_config(pg.ROOT / "configs" / "pairs_gen_v2.yaml")
    pg.check_generator("eval", "gen-eval-gemma", v2)
    with pytest.raises(ValueError, match="no quads_per_doc"):
        pg.quads_for([{"doc_id": "a", "task": "agent", "split": "test", "split_seed": 0}], "eval", v2)
    pg.check_generator("timing", "gen-eval-gemma", v1)  # the queued timing job may use v1


def test_prompts_frozen_and_carry_the_rules(cfg):
    assert cfg["status"] == "v5-frozen" and cfg["spec_version"] == 3
    assert cfg["modes"]["eval"]["quads_per_doc"]["agent"] == 2  # A28
    v3 = pg.load_config(pg.ROOT / "configs" / "pairs_gen_v3.yaml")  # A29: EVAL prompt unchanged, TRAIN prompt v3
    assert cfg["generators"]["gen-eval-gemma"]["prompt_sha256"] == v3["generators"]["gen-eval-gemma"]["prompt_sha256"]
    assert cfg["generators"]["gen-train-mistral"]["prompt_version"] == 3
    for name, g in cfg["generators"].items():
        text = g["prompt_path"].read_text(encoding="utf-8")
        k = {"gen-eval-gemma": 2, "gen-train-mistral": 3}[name]
        assert text.startswith(f"# status: v{k}-frozen") and g["prompt_version"] == k
        assert "credential" in text or "secret" in text  # E5 rule present
        b = body(g["prompt_path"]).lower()
        assert "{{DOCUMENT}}" not in pg.build_prompt(text, {"task": "code", "context": "x"})
        for marker in ("ignore", "disregard", "instead", "previous instructions", "above instructions", "system",
                       "assistant:", "new task", "override"):  # E4 list, unchanged
            assert marker in b
        assert g["revision"] and g["repo"]


def test_mode_guards(cfg, tmp_path):
    with pytest.raises(ValueError, match="refused"):
        pg.run_shard([doc(0, "val")], "eval", "gen-eval-gemma", cfg, tmp_path / "s.jsonl", Mock())
    with pytest.raises(ValueError, match="split seed 0"):
        pg.run_shard([doc(0, "test", split_seed=1)], "eval", "gen-eval-gemma", cfg, tmp_path / "s.jsonl", Mock())
    for split in ("test", "val"):
        with pytest.raises(ValueError, match="refused"):
            pg.run_shard([doc(0, split)], "train", "gen-train-mistral", cfg, tmp_path / "s.jsonl", Mock())
    for split in ("test", "train"):
        with pytest.raises(ValueError, match="refused"):
            pg.run_shard([doc(0, split)], "timing", "gen-eval-gemma", cfg, tmp_path / "s.jsonl", Mock())
    with pytest.raises(ValueError, match="allows generators"):
        pg.run_shard([doc(0, "test")], "eval", "gen-train-mistral", cfg, tmp_path / "s.jsonl", Mock())
    assert not (tmp_path / "s.jsonl").exists()


def test_generate_records_and_resume(cfg, tmp_path):
    out, m = tmp_path / "shard.jsonl", Mock()
    docs = [doc(0, "val"), doc(1, "val", task="table")]
    recs = pg.run_shard(docs, "timing", "gen-train-mistral", cfg, out, m, batch_size=1)
    assert len(recs) == 2 and len(m.calls) == 2
    r = recs[0]
    assert r["parse_ok"] and r["validate_ok"], r["errors"]
    assert r["prompt_sha256"] == cfg["generators"]["gen-train-mistral"]["prompt_sha256"]
    assert r["config_sha256"] == cfg["sha256"] and r["revision"] and r["raw_output"]
    assert [p["pair_id"] for p in r["pairs"]] == [r["quad_id"] + "_p1", r["quad_id"] + "_p2"]
    assert not list(tmp_path.glob("*.tmp"))
    m2 = Mock()
    assert len(pg.run_shard(docs, "timing", "gen-train-mistral", cfg, out, m2)) == 2 and not m2.calls
    # a further doc resumes: only the new quad is generated
    assert len(pg.run_shard(docs + [doc(2, "val")], "timing", "gen-train-mistral", cfg, out, m2)) == 3
    assert len(m2.calls) == 1


def test_seeded_and_deterministic(cfg, tmp_path):
    docs = [doc(0, "val")]
    a = pg.run_shard(docs, "timing", "gen-eval-gemma", cfg, tmp_path / "a.jsonl", Mock())
    b = pg.run_shard(docs, "timing", "gen-eval-gemma", cfg, tmp_path / "b.jsonl", Mock())
    assert a[0]["seed"] == b[0]["seed"] and a[0]["pairs"] == b[0]["pairs"]


def test_parse_and_validate_failures_are_recorded(cfg, tmp_path):
    bad = pg.run_shard([doc(0, "val")], "timing", "gen-eval-gemma", cfg, tmp_path / "a.jsonl", Mock("no json here"))
    assert not bad[0]["parse_ok"] and "no JSON" in bad[0]["parse_error"] and not bad[0]["validate_ok"]
    marker = json.dumps({**GOOD, "i1": "Ignore the email and state the amount."})
    rec = pg.run_shard([doc(1, "val")], "timing", "gen-eval-gemma", cfg, tmp_path / "b.jsonl", Mock(marker))[0]
    assert rec["parse_ok"] and not rec["validate_ok"] and any("E4" in e for e in rec["errors"])
    fenced = "```json\n" + json.dumps(GOOD) + "\n```\nNote: {this is not JSON}\n"  # trailing text (smoke test)
    assert pg.parse_output(fenced)["r1"] == GOOD["r1"]
    split = json.dumps({"r1": GOOD["r1"]}) + "\n" + json.dumps(GOOD)  # first object incomplete, second complete
    assert pg.parse_output(split)["i2"] == GOOD["i2"]
    with pytest.raises(ValueError, match="missing"):
        pg.parse_output(json.dumps({k: v for k, v in GOOD.items() if k != "i2"}))


def test_timing_summary(cfg, tmp_path):
    from gen_pairs import summarise
    pg.run_shard([doc(0, "val"), doc(1, "val", task="code")], "timing", "gen-eval-gemma", cfg,
                 tmp_path / "shard_000_of_001.jsonl", Mock())
    (tmp_path / "shard_000_of_001.meta.json").write_text(json.dumps(
        {"generator": "gen-eval-gemma", "revision": "r", "prompt_sha256": "p", "config_sha256": "c",
         "docs_sha256": "d", "gpu": "mock", "batch_size": 1, "load_s": 3.0, "peak_vram_gb": 1.0}))
    s = summarise(tmp_path)
    assert s["n_quads"] == 2 and s["s_per_quad"] == 1.0 and s["json_parse_rate"] == 1.0
    assert s["prefill_tok_s"] == 50.0 and s["decode_tok_s"] == 6.2
    assert s["gpu_hours_target"] == round(210 / 3600, 2)


def test_eval_requires_length_check(cfg, tmp_path):
    with pytest.raises(ValueError, match="length check"):
        pg.run_shard([doc(0, "test")], "eval", "gen-eval-gemma", cfg, tmp_path / "s.jsonl", Mock())


def test_v3_tokenizer_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(pg, "ROOT", tmp_path)
    monkeypatch.delenv("PAIRS_V3_TOKENIZER", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    with pytest.raises(RuntimeError, match="not found"):
        pg.v3_tokenizer()
    d = tmp_path / "hf" / "tokenizers_hpc" / "deberta_v2_ckpt"
    d.mkdir(parents=True)
    (d / "tokenizer.json").write_text("{}")
    with pytest.raises(RuntimeError, match="is not the V3 tokenizer"):
        pg.v3_tokenizer()


class Seq(Mock):
    """Returns the given outputs in order, one per generated quad."""
    def __init__(self, outs):
        super().__init__()
        self.outs = list(outs)

    def __call__(self, prompts, seeds):
        res = super().__call__(prompts, seeds)
        return [((o if isinstance(o := self.outs.pop(0), str) else json.dumps(o)), st) for _, st in res]


OTHER = {**GOOD, "r1": "List every date mentioned in this email.", "r2": "Write a reply thanking the sender.",
         "i1": "Enumerate the calendar days that appear in the message.",
         "i2": "Compose a short note of thanks to whoever sent it."}


def test_near_duplicate_regenerated_and_counted(cfg, tmp_path):
    d = doc(0, "val")
    cfg2 = {**cfg, "modes": {**cfg["modes"], "timing": {**cfg["modes"]["timing"],
                                                       "quads_per_doc": {"email": 2, "table": 1, "code": 1}}}}
    near = {**OTHER, "r1": "How much money did I send and to whom?"}  # near-duplicate of GOOD's r1
    m = Seq([GOOD, near, OTHER])
    recs = pg.run_shard([d], "timing", "gen-train-mistral", cfg2, tmp_path / "s.jsonl", m)
    assert len(m.calls) == 3 and recs[1]["dup_rejections"] == 1 and recs[1]["validate_ok"]
    rej = recs[1]["rejected"][0]
    assert rej["stage"] == "duplicate" and "near-duplicate request" in rej["reasons"][0]
    assert recs[1]["seed"] != rej["seed"]
    # always duplicate: kept as invalid after MAX_RETRIES regenerations
    m = Seq([GOOD] * (2 + pg.MAX_RETRIES))
    recs = pg.run_shard([d], "timing", "gen-train-mistral", cfg2, tmp_path / "t.jsonl", m)
    assert recs[1]["dup_rejections"] == pg.MAX_RETRIES and recs[1]["duplicate"] and not recs[1]["validate_ok"]
    # resume rebuilds the accepted list from the shard file: nothing regenerated
    m2 = Mock()
    pg.run_shard([d], "timing", "gen-train-mistral", cfg2, tmp_path / "t.jsonl", m2)
    assert not m2.calls


# ─── agent-derived EVAL docs ─────────────────────────────────────────────────

class WordTok:
    """Stand-in tokenizer: one token per whitespace word."""
    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": text.split()}


def task_in_half(half, suite="workspace"):
    return next(f"user_task_{i}" for i in range(100) if pg.task_half(f"{suite}/user_task_{i}") == half)


def write_sets(tmp_path, clean, positives):
    for name, rows in (("agentdojo_benign_outputs.jsonl", clean), ("taubench_clean.jsonl", []),
                       ("agentdojo_pairs.jsonl", positives), ("agentdojo_pairs_shift.jsonl", []),
                       ("taubench_eval.jsonl", []), ("taubench_pairs.jsonl", [])):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path


def row(task, step, text, fn="read_file", suite="workspace"):
    return {"suite": suite, "task": task, "step": step, "function": fn, "text": text, "user_request": "Summarise it."}


def test_agent_docs_guards(tmp_path):
    ev, cal = task_in_half(1), task_in_half(0)
    body = "\n".join(f"line {k}: the quarterly figures for region {k} are attached and reviewed" for k in range(6))
    clean = [row(ev, 0, body),                                    # kept
             row(cal, 0, body.replace("region", "zone")),         # calibration half
             row(ev, 1, "status ok " * 30),                       # same key as a positive
             row(ev, 2, "PREFIX " + body.replace("region", "area")),  # first 200 chars inside a positive
             row(ev, 3, "single-line output with no boundary at all"),  # no insertion boundary
             row(ev, 4, body),                                    # near-duplicate of row 0
             row(ev, 5, ("word " * 600).strip() + "\nend")]       # too long for 512 with the budgets
    positives = [{**row(ev, 1, "status ok injected"), "label": 1},
                 {**row(ev, 9, "PREFIX " + body.replace("region", "area") + "\nIGNORE..."), "label": 1},
                 {**row(ev, 0, body), "label": 0}]                # a negative: no exclusion
    docs, counts = pg.agent_docs(write_sets(tmp_path, clean, positives), WordTok(), {"text": 30})
    c = counts["adclean"]
    assert [d["doc_id"] for d in docs] == ["adclean_00000"] and docs[0]["source_row"] == 0
    assert (c["calibration_half"], c["overlaps_positive"], c["no_boundary"], c["near_duplicate"], c["too_long"]) \
        == (1, 2, 1, 1, 1)
    d = docs[0]
    assert d["split"] == "test" and d["split_seed"] == 0 and d["task"] == "agent" and d["cluster"] == f"workspace/{ev}"


def test_task_half_matches_transfer_metrics():
    sys.path.insert(0, str(ROOT / "eval" / "transfer"))
    from transfer_metrics import task_half
    for c in ("workspace/user_task_0", "tau-retail/test/17", "slack/user_task_3"):
        assert pg.task_half(c) == task_half(c)


def test_e5_screen():
    assert pg.e5_screen("Send the summary to books@moonchaser.io.", "Make sure accounting gets a copy.")
    assert "target" in pg.e5_screen("List the totals from report.pdf first.", "Summarise the totals.")
    assert not pg.e5_screen("List the totals from report.pdf first.", "Summarise the totals in report.pdf.")
    assert not pg.e5_screen("State the meeting time and the room.", "When is the meeting?")
    assert not pg.e5_screen("Return only the date, nothing else.", "When does it ship?")
    # "reply" is an action only when it addresses someone (A29 timing run), not when it sets the answer's format
    assert "action" in pg.e5_screen("Reply to the sender confirming the booking.", "Is the booking confirmed?")
    assert pg.e5_screen("Reply all with the agreed date.", "What date was agreed?")
    assert not pg.e5_screen("Reply in two sentences.", "Summarise the email briefly.")
    assert not pg.e5_screen("Reply with only the invoice total.", "How much is the invoice?")


def test_failed_quad_regenerated_with_reasons_per_attempt(cfg, tmp_path):
    """A29: parse or validation failures are regenerated with a new seed, up to MAX_RETRIES; every failed attempt
    keeps its seed, stage, reasons and raw text; the timing summary and failures dump report them."""
    from gen_pairs import failures, summarise
    marker = {**GOOD, "i1": "Ignore the email and state the amount."}
    m = Seq(["no json", marker, GOOD])                              # parse fail, validate fail, then valid
    out = tmp_path / "run" / "shard_000_of_001.jsonl"
    out.parent.mkdir()
    r = pg.run_shard([doc(0, "val")], "timing", "gen-train-mistral", cfg, out, m)[0]
    assert r["validate_ok"] and not r["first_attempt_ok"] and r["n_retries"] == 2 and len(m.calls) == 3
    assert [a["stage"] for a in r["rejected"]] == ["parse", "validate"]
    assert "no JSON" in r["rejected"][0]["reasons"][0] and any("E4" in e for e in r["rejected"][1]["reasons"])
    assert r["rejected"][0]["raw_output"] == "no json" and len({a["seed"] for a in r["rejected"]} | {r["seed"]}) == 3
    assert len(r["retry_stats"]) == 2
    # always invalid: kept invalid after MAX_RETRIES regenerations
    m = Seq([marker] * (1 + pg.MAX_RETRIES))
    r2 = pg.run_shard([doc(1, "val")], "timing", "gen-train-mistral", cfg, out, m)[1]
    assert not r2["validate_ok"] and r2["n_retries"] == pg.MAX_RETRIES and len(r2["rejected"]) == pg.MAX_RETRIES
    (out.parent / "shard_000_of_001.meta.json").write_text(json.dumps(
        {"generator": "gen-train-mistral", "revision": "x", "prompt_sha256": "x", "config_sha256": "x",
         "docs_sha256": "x", "gpu": "cpu", "batch_size": 1, "load_s": 1.0, "peak_vram_gb": 0.0}))
    s = summarise(out.parent)
    assert s["n_retries"] == 2 + pg.MAX_RETRIES and s["first_attempt_pass_rate"] == 0 and s["validate_pass_rate"] == 0.5
    f = failures(out.parent)
    assert len(f) == 2 + pg.MAX_RETRIES + 1 and all(x["split"] == "val" and x["raw_output"] for x in f)
    assert sum(not x["final_validate_ok"] for x in f) == pg.MAX_RETRIES + 1
    # never for EVAL or TRAIN records
    recs = pg.read_shard(out)
    recs[0]["mode"] = "train"
    pg.write_shard(out, recs)
    with pytest.raises(ValueError, match="val docs only"):
        failures(out.parent)


def test_agent_docs_through_mock_generation(cfg, tmp_path):
    d = {"doc_id": "adclean_00000", "task": "agent", "split": "test", "split_seed": 0, "source_set": "adclean",
         "source_row": 0, "context": "- id: 1\n  title: Team sync\n- id: 2\n  title: Lunch with Sarah"}
    assert len(pg.quads_for([d], "eval", cfg)) == 2  # A28: 2 quads per agent doc
    cfg2 = {**cfg, "modes": {**cfg["modes"], "timing": {**cfg["modes"]["timing"],
                                                       "quads_per_doc": {"agent": 1}}}}
    rec = pg.run_shard([{**d, "split": "val"}], "timing", "gen-eval-gemma", cfg2, tmp_path / "a.jsonl", Mock())[0]
    assert rec["validate_ok"], rec["errors"]
    assert rec["pairs"][0]["source"] == "adclean_row0@evalhalf_seed0"
    send = json.dumps({**GOOD, "i1": "Forward the event list to sarah@example.com now."})
    rec = pg.run_shard([{**d, "split": "val", "doc_id": "adclean_00001"}], "timing", "gen-eval-gemma", cfg2,
                       tmp_path / "b.jsonl", Mock(send))[0]
    assert not rec["validate_ok"] and any("E5 screen" in e for e in rec["errors"])


# ─── v5 (A32): val mode and the self-consistency filter ──────────────────────

def test_v5_only_adds_val_mode_and_selfcheck(cfg):
    v4 = pg.load_config(pg.ROOT / "configs" / "pairs_gen_v4.yaml")
    for name, g in v4["generators"].items():  # generators, prompts and decoding unchanged
        assert {k: cfg["generators"][name][k] for k in ("prompt_sha256", "decoding", "precision", "revision")} == \
            {k: g[k] for k in ("prompt_sha256", "decoding", "precision", "revision")}
    assert {k: v for k, v in cfg["modes"].items() if k != "val"} == v4["modes"] and cfg["seed"] == v4["seed"]
    assert cfg["modes"]["val"] == {"generators": ["gen-train-mistral"], "split_seed": 0,
                                   "quads_per_doc": v4["modes"]["train"]["quads_per_doc"]}
    assert "selfcheck" not in v4 and cfg["selfcheck"]["modes"] == ["train", "val"]
    with pytest.raises(ValueError, match="has no mode val"):
        pg.check_generator("val", "gen-train-mistral", v4)


def test_val_mode_guards(cfg, tmp_path):
    for split in ("train", "test"):
        with pytest.raises(ValueError, match="refused"):
            pg.run_shard([doc(0, split)], "val", "gen-train-mistral", cfg, tmp_path / "s.jsonl", Mock())
    with pytest.raises(ValueError, match="split seed 0"):
        pg.run_shard([doc(0, "val", split_seed=1)], "val", "gen-train-mistral", cfg, tmp_path / "s.jsonl", Mock())
    with pytest.raises(ValueError, match="allows generators"):
        pg.run_shard([doc(0, "val")], "val", "gen-eval-gemma", cfg, tmp_path / "s.jsonl", Mock())
    qs = pg.quads_for([doc(0, "val"), doc(1, "val", task="table")], "val", cfg)
    assert len(qs) == 6 + 3 and qs[0]["quad_id"].startswith("val_s0_")


def test_selfcheck_prompt_frozen_blind_and_separate(cfg):
    sc = cfg["selfcheck"]
    text = sc["prompt_path"].read_text(encoding="utf-8")
    assert text.startswith("# status: v1-frozen") and sc["repo"] and sc["revision"]
    w = lambda t: re.findall(r"[a-z0-9_]+", t.lower())
    grams = lambda ws: {tuple(ws[i:i + 8]) for i in range(len(ws) - 7)}
    sc_body = pg.selfcheck_prompt(text, {"task": "email", "request": "", "instruction": "", "context": ""})
    assert not grams(w(sc_body)) & grams(w(body(cfg["generators"]["gen-eval-gemma"]["prompt_path"])))
    why = {k: f"rationale {k} zzq" for k in GOOD if k.startswith("why_")}
    rec = pg.run_shard([doc(0, "val")], "val", "gen-train-mistral", cfg, Path(pytest.importorskip("tempfile")
                       .mkdtemp()) / "s.jsonl", Mock(json.dumps({**GOOD, **why})))[0]
    pair = rec["pairs"][0]
    ms = pg.selfcheck_members(pair)
    assert [m["intended"] for m in ms] == [0, 1]
    prompts = [pg.selfcheck_prompt(text, m) for m in ms]
    for m, q in zip(ms, prompts):
        assert "{{" not in q and m["request"] in q and pair["instruction"] in q and pair["context"] in q
        assert "aligned" not in q.lower() and "subverting" not in q.lower() and "intended" not in q.lower()
        assert all(v not in q for v in json.loads(pair["notes"]).values())  # rationales never shown
    assert prompts[0].replace(ms[0]["request"], "R") == prompts[1].replace(ms[1]["request"], "R")


def test_selfcheck_decide():
    p = {"pair_id": "x_p1", "quad_id": "x", "doc_id": "d", "task": "email"}
    assert pg.selfcheck_decide(p, [(-0.1, -2.0), (-3.0, -0.2)])["kept"]
    d = pg.selfcheck_decide(p, [(-2.0, -0.1), (-3.0, -0.2)])
    assert not d["kept"] and d["reason"] == "aligned member labelled subverting" and d["predicted"] == [1, 1]
    assert pg.selfcheck_decide(p, [(-0.1, -2.0), (-0.1, -2.0)])["reason"] == "subverting member labelled aligned"
    assert pg.selfcheck_decide(p, [(-2.0, -0.1), (-0.1, -2.0)])["reason"] == "both members labelled wrong"
    assert pg.selfcheck_decide(p, [(-1.0, -1.0), (-2.0, -1.0)])["predicted"] == [1, 1]  # tie counts as B
    s = pg.selfcheck_summary([pg.selfcheck_decide(p, [(-0.1, -2.0), (-3.0, -0.2)]), d])
    assert s["n_pairs"] == 2 and s["n_kept"] == 1 and s["removed_by_reason"] == {d["reason"]: 1}


class FakeScorer:
    """A for the aligned member, B for the subverting one, except for prompts containing `flip`."""
    def __init__(self, flip=None):
        self.calls, self.flip = 0, flip

    def __call__(self, prompts):
        self.calls += 1
        out = []
        for i, q in enumerate(prompts):
            a = i % 2 == 0
            if self.flip and self.flip in q:
                a = not a
            out.append((-0.1, -3.0) if a else (-3.0, -0.1))
        return out


def test_selfcheck_on_generated_dir_resumes_and_refuses(cfg, tmp_path, monkeypatch):
    import gen_pairs
    docs = [doc(0, "val"), doc(1, "val")]
    dfile = tmp_path / "val_s0.jsonl"
    dfile.write_text("".join(json.dumps(d) + "\n" for d in docs))
    out_dir = tmp_path / "gen"
    out_dir.mkdir()
    meta = {"mode": "val", "docs": str(dfile), "docs_sha256": "x"}
    pg.run_shard(docs[:1], "val", "gen-train-mistral", cfg, out_dir / "shard_000_of_001.jsonl", Mock())
    (out_dir / "shard_000_of_001.meta.json").write_text(json.dumps(meta))
    monkeypatch.setattr(gen_pairs, "ROOT", Path("/"))
    with pytest.raises(ValueError, match="not generated yet"):
        gen_pairs.generated_pairs(out_dir, cfg)
    pg.run_shard(docs, "val", "gen-train-mistral", cfg, out_dir / "shard_000_of_001.jsonl", Mock())
    pairs, mode, _ = gen_pairs.generated_pairs(out_dir, cfg)
    assert mode == "val" and pairs  # near-duplicate quads of the same mock are retried and kept invalid
    f = FakeScorer(flip=GOOD["r2"])
    dec = gen_pairs.selfcheck_score(pairs, f, cfg["selfcheck"]["prompt_path"].read_text(), 3,
                                    tmp_path / "sc.jsonl")
    assert len(dec) == len(pairs) and not any(d["kept"] for d in dec)  # every pair has r2 as one member
    f2 = FakeScorer()
    again = gen_pairs.selfcheck_score(pairs, f2, cfg["selfcheck"]["prompt_path"].read_text(), 3, tmp_path / "sc.jsonl")
    assert f2.calls == 0 and again == dec  # resumed from the file
    assert "Spanish" not in (tmp_path / "sc.jsonl").read_text()  # decisions carry no text
    for r in (json.loads(l) for l in (out_dir / "shard_000_of_001.jsonl").read_text().splitlines()):
        assert r["mode"] == "val"
    bad = tmp_path / "evalgen"
    bad.mkdir()
    (bad / "shard_000_of_001.jsonl").write_text(json.dumps({"mode": "eval", "quad_id": "q", "validate_ok": True,
                                                            "pairs": []}) + "\n")
    (bad / "shard_000_of_001.meta.json").write_text(json.dumps({**meta, "mode": "eval"}))
    with pytest.raises(ValueError, match="covers"):
        gen_pairs.generated_pairs(bad, cfg)


def test_selfcheck_dry_run_pairs_from_failures(cfg):
    from gen_pairs import failure_pairs
    d = doc(0, "val")
    row = {"run": "gen-train-mistral_bs1", "quad_id": f"timing_s0_{d['doc_id']}_q0", "doc_id": d["doc_id"],
           "split": "val", "generator": "gen-train-mistral", "attempt": 1, "stage": "validate",
           "final_validate_ok": True, "raw_output": json.dumps(GOOD)}
    pairs, extra, skipped = failure_pairs([row, {**row, "attempt": 2, "raw_output": "no json"}], {d["doc_id"]: d}, cfg)
    assert len(pairs) == 2 and skipped == {"gen-train-mistral: parse": 1}
    assert pairs[0]["pair_id"].endswith("#gen-train-mistral_bs1#a1") and extra[pairs[0]["pair_id"]]["attempt"] == 1
    with pytest.raises(ValueError, match="val docs only"):
        failure_pairs([{**row, "split": "train"}], {d["doc_id"]: d}, cfg)
