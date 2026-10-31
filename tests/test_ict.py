"""ICT (src/models/ict.py) on CPU with tiny random-init backbones and a word-level toy tokenizer (no downloads)."""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoModelForSequenceClassification, DebertaV2Config, PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "models"))
import ict  # noqa: E402

WORDS = ("summarize translate list count email table code report alpha beta gamma delta note please the of "
         "Intent: | Context: what is in this").split()
DEVICE = torch.device("cpu")


def toy_tokenizer():
    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    vocab = {w: i for i, w in enumerate(special + sorted(set(WORDS)))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    t.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                   sep_token="[SEP]")


def tiny_deberta(tok, seed=0, dropout=0.0):
    torch.manual_seed(seed)
    c = DebertaV2Config(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128, relative_attention=True,
                        position_buckets=32, pos_att_type=["p2c", "c2p"], position_biased_input=False,
                        type_vocab_size=0, num_labels=1, pad_token_id=0, pooler_hidden_size=32, initializer_range=0.2,
                        hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout, pooler_dropout=dropout)
    return AutoModelForSequenceClassification.from_config(c)


def tiny_modernbert(tok, seed=0):
    from transformers import ModernBertConfig
    torch.manual_seed(seed)
    c = ModernBertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=128, num_labels=1, pad_token_id=0,
                         cls_token_id=2, sep_token_id=3, bos_token_id=2, eos_token_id=3, local_attention=16,
                         global_attn_every_n_layers=1, attn_implementation="eager", initializer_range=0.2)
    return AutoModelForSequenceClassification.from_config(c)


VERBS = ["summarize", "translate", "list", "count"]


def toy_data(n_docs=8, seed=0, prefix=""):
    """V3-like rows (attack = 'please <verb> the ...' injected, label 1) and crossed-quad-like pairs: the instruction
    'note <verb>' is aligned with the request naming the same verb and subverting for one naming another verb."""
    rng = np.random.default_rng(seed)
    filler = ["alpha", "beta", "gamma", "delta", "the", "of", "report"]
    rows, pairs = [], []
    for d in range(n_docs):
        task = ["email", "table"][d % 2]
        doc = " ".join(rng.choice(filler, 12))
        v1, v2 = rng.choice(VERBS, 2, replace=False)
        for k, (ctx, y) in enumerate([(doc, 0), (f"{doc} please {v1} the {task}", 1)]):
            rows.append({"id": f"{prefix}{task}_{d}_{k}", "doc_id": f"{prefix}{task}_{d}", "split_seed": 0,
                         "task_type": task, "component": f"{prefix}{task}_{d // 4}",  # two docs per thread
                         "label": y, "user_intent": f"what is in this {task}", "text": ctx,
                         "attack_category": "x" if y else "", "attack_source": "", "insertion_mode": "",
                         "clean_type": "" if y else "raw"})
        for q, (vi, vo) in enumerate([(v1, v2), (v2, v1)]):
            pairs.append({"pair_id": f"{prefix}p{d}_{q}", "quad_id": f"{prefix}q{d}", "doc_id": f"{prefix}{task}_{d}",
                          "task": task, "context": f"{doc} note {vi} report", "instruction": f"note {vi} report",
                          "char_offset": len(doc) + 1, "source": "toy",
                          "request_aligned": f"{vi} this {task}", "request_subverting": f"{vo} this {task}"})
    return ict.v3_frame(pd.DataFrame(rows)), pairs


@pytest.fixture(scope="module")
def tok():
    return toy_tokenizer()


# ─── batch invariance ────────────────────────────────────────────────────────

@pytest.mark.parametrize("backbone,windowed,max_length", [("deberta", False, 64), ("deberta", True, 12),
                                                          ("modernbert", True, 12)])
def test_scores_identical_bs1_and_batched(tok, backbone, windowed, max_length):
    model = tiny_deberta(tok) if backbone == "deberta" else tiny_modernbert(tok)
    cfg = ict.ICTConfig(max_length=max_length, windowed=windowed, stride=4)
    rows, pairs = toy_data()
    fr = pd.concat([rows, ict.pairs_frame(pairs)])  # mixed lengths, so batches pad
    enc = ict.make_encoder(tok, model, cfg)
    if windowed:
        assert max(len(enc(q, c)) for q, c in zip(fr["request"], fr["context"])) > 1
    s1 = ict.score(model, enc, fr["request"], fr["context"], DEVICE, batch_size=1)
    s8 = ict.score(model, enc, fr["request"], fr["context"], DEVICE, batch_size=8)
    s_all = ict.score(model, enc, fr["request"], fr["context"], DEVICE, batch_size=len(fr))
    assert np.abs(s1 - s8).max() <= 1e-5 and np.abs(s1 - s_all).max() <= 1e-5


def test_windowed_score_is_max_over_windows(tok):
    model = tiny_deberta(tok)
    cfg = ict.ICTConfig(max_length=12, windowed=True, stride=4)
    enc = ict.make_encoder(tok, model, cfg)
    q, c = "what is in this email", "alpha beta gamma delta the of report alpha beta gamma please list the email"
    wins = enc(q, c)
    assert len(wins) > 1 and all(w["input_ids"][:7] == wins[0]["input_ids"][:7] for w in wins)  # request in each
    per_window = model(**tok.pad(wins, return_tensors="pt")).logits[:, 0]
    assert ict.score(model, enc, [q], [c], DEVICE)[0] == pytest.approx(per_window.max().item(), abs=1e-6)


def test_truncation_keeps_request_and_cuts_context_end(tok):
    cfg = ict.ICTConfig(max_length=10)
    w = ict.make_encoder(tok, tiny_deberta(tok), cfg)("what is in this email", "alpha beta gamma delta the of")
    assert len(w) == 1 and tok.decode(w[0]["input_ids"]) == "[CLS] what is in this email [SEP] alpha beta [SEP]"


# ─── loss ────────────────────────────────────────────────────────────────────

def test_margin_hinge_hand_computed():
    cfg = ict.ICTConfig(margin=1.0, margin_lambda=2.0)
    y = torch.tensor([0, 1, 0, 1])
    s = torch.tensor([0.0, 2.0, 1.0, 0.5])
    total, ce, mg = ict.ict_loss(s, y, [(0, 1), (2, 3)], cfg)
    assert mg.item() == pytest.approx((0.0 + 1.5) / 2)       # gap 2 >= 1 -> 0; gap -0.5 -> 1.5
    assert total.item() == pytest.approx(ce.item() + 2.0 * mg.item())
    total0, ce0, mg0 = ict.ict_loss(s, y, [(0, 1), (2, 3)], replace(cfg, margin_lambda=0.0))
    assert mg0.item() == 0 and total0.item() == pytest.approx(ce0.item())


def test_margin_term_widens_pair_gap(tok):
    """One identical SGD step from the same init: with the margin term the subverting-aligned gap grows more."""
    _, pairs = toy_data()
    gaps = {}
    for lam in (0.0, 10.0):
        model = tiny_deberta(tok, seed=1)
        cfg = ict.ICTConfig(margin_lambda=lam, margin=5.0, request_swap=False)
        enc = ict.make_encoder(tok, model, cfg)
        items = [it for it in ict.epoch_items(cfg, pd.DataFrame(columns=["request", "context", "label"]), pairs, 0)]
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        loss, _, _ = ict.batch_loss(model, enc, items, cfg, DEVICE)
        loss.backward()
        opt.step()
        fr = ict.pairs_frame(pairs)
        s = ict.score(model, enc, fr["request"], fr["context"], DEVICE)
        gaps[lam] = float(np.mean(s[fr["label"].values == 1] - s[fr["label"].values == 0]))
    assert gaps[10.0] > gaps[0.0]


def test_training_loss_decreases(tok, tmp_path):
    rows, pairs = toy_data()
    model = tiny_deberta(tok)
    cfg = ict.ICTConfig(lr=3e-3, epochs=8, batch_size=4, grad_accum=1, max_length=64)
    st = ict.train(cfg, model, tok, rows, pairs, {}, tmp_path, DEVICE, log=lambda m: None)
    first, last = st["history"][0]["train_loss"], st["history"][-1]["train_loss"]
    assert last < 0.7 * first
    assert (tmp_path / "best" / "ict_config.json").exists()


# ─── ablation flags ──────────────────────────────────────────────────────────

def test_presets_and_invalid_combinations():
    assert ict.make_config("a").use_pairs is False and ict.make_config("a").use_agent_data is True
    assert ict.make_config("d").use_agent_data is False and ict.make_config("d").use_pairs
    assert ict.make_config("e").request_input == "none" and not ict.make_config("e").use_pairs
    assert len({tuple(sorted(vars(ict.make_config(k)).items())) for k in ict.ABLATIONS}) == len(ict.ABLATIONS)
    with pytest.raises(ValueError):
        ict.ICTConfig(use_pairs=False)              # margin and swap need pairs
    with pytest.raises(ValueError):
        ict.ICTConfig(request_input="none")         # pair members would be identical inputs
    with pytest.raises(ValueError):
        ict.ICTConfig(request_input="concat")       # the V2 single-sequence variant is dropped (A22)
    assert ict.ICTConfig(backbone="modernbert-large").windowed and not ict.ICTConfig().windowed


def test_each_ablation_removes_exactly_one_factor():
    """A23: (a)-(d) differ from full ICT, and (e) from (a), in one factor. (a)'s factor is pairs+margin; the swap
    is part of it (request_swap needs pairs)."""
    factors = {"pairs_margin": ("use_pairs", "margin_lambda", "request_swap"), "margin": ("margin_lambda",),
               "swap": ("request_swap",), "agent_data": ("use_agent_data",), "request": ("request_input",)}
    expected = {"a": ("full", "pairs_margin"), "b": ("full", "margin"), "c": ("full", "swap"),
                "d": ("full", "agent_data"), "e": ("a", "request")}
    for k, (ref, factor) in expected.items():
        a, r = vars(ict.make_config(k)), vars(ict.make_config(ref))
        assert {f for f in a if a[f] != r[f]} == set(factors[factor]), k


def test_selected_hparams_apply_but_presets_win():
    hp = {"margin_lambda": 2.0, "margin": 0.5, "lr": 1e-5, "grid_index": 9, "selection_score": 0.8}
    full, b = ict.make_config("full", hp), ict.make_config("b", hp)
    assert (full.margin_lambda, full.margin, full.lr) == (2.0, 0.5, 1e-5)
    assert (b.margin_lambda, b.margin, b.lr) == (0.0, 0.5, 1e-5)
    assert ict.make_config("full", hp, lr=3e-5).lr == 3e-5
    assert len(ict.GRID) == 12 and ict.GRID[0] == {"margin_lambda": 0.5, "margin": 0.5, "lr": 1e-5}


def test_every_ablation_changes_items_inputs_or_loss(tok):
    """Each preset differs from full ICT in what it trains on, what the model sees, or the loss value."""
    rows, pairs = toy_data()
    arows, apairs = toy_data(n_docs=4, seed=1, prefix="ag")
    model = tiny_deberta(tok)

    def fingerprint(cfg):
        r, p = ict.training_data(cfg, rows, pairs, arows, apairs)
        items = ict.epoch_items(cfg, r, p, 0)
        enc = ict.make_encoder(tok, model, cfg)
        members = sorted(m for it in items for m in it["members"])
        ids = [tuple(enc(q, c)[0]["input_ids"]) for q, c, _ in members]
        with torch.no_grad():
            loss = ict.batch_loss(model, enc, items, cfg, DEVICE)[0].item()
        return members, ids, round(loss, 6)

    full = fingerprint(ict.make_config("full"))
    for k in "abcde":
        assert fingerprint(ict.make_config(k)) != full, k
    # specific behaviours
    r_a, p_a = ict.training_data(ict.make_config("a"), rows, pairs, arows, apairs)
    assert len(r_a) == len(rows) + len(arows) and p_a == []                            # same data, no pairs
    r_d, p_d = ict.training_data(ict.make_config("d"), rows, pairs, arows, apairs)
    assert len(r_d) == len(rows) and len(p_d) == len(pairs)                            # no agent rows or pairs
    r_f, p_f = ict.training_data(ict.make_config("full"), rows, pairs, arows, apairs)
    assert len(r_f) == len(rows) + len(arows) and len(p_f) == len(pairs) + len(apairs)
    b, c = ict.make_config("b"), ict.make_config("c")
    assert any(len(it["members"]) == 3 for it in ict.epoch_items(b, rows, pairs, 0))      # swap on, no margin
    assert all(len(it["members"]) <= 2 for it in ict.epoch_items(c, rows, pairs, 0))      # margin on, no swap
    enc_e = ict.make_encoder(tok, model, ict.make_config("e"))
    assert enc_e("summarize this email", "alpha beta") == enc_e("list this email", "alpha beta")
    enc_full = ict.make_encoder(tok, model, ict.make_config("full"))
    assert enc_full("summarize this email", "alpha beta") != enc_full("list this email", "alpha beta")


def test_request_swap_draws_same_task_deterministically():
    rows, pairs = toy_data()
    pairs = ict.attach_components(pairs, rows)
    rng = lambda: np.random.default_rng([0, 3])  # noqa: E731
    s1, s2 = ict.swap_requests(pairs, rng()), ict.swap_requests(pairs, rng())
    assert s1 == s2 and all(s is not None for s in s1)
    for p, s in zip(pairs, s1):
        assert s.split()[-1] == p["task"]
    items = ict.epoch_items(ict.ICTConfig(), pd.DataFrame(columns=["request", "context", "label"]), pairs, 0)
    for it in items:
        assert [m[2] for m in it["members"]] == [0, 1, 1] and it["pairs"] == [(0, 1), (0, 2)]


def test_swap_excludes_same_thread_and_shared_content_words():
    rows, pairs = toy_data(n_docs=16)
    pairs = [{**p, "request_aligned": f"{p['request_aligned']} {p['doc_id']}",            # unique per document
              "request_subverting": f"{p['request_subverting']} {p['doc_id']}"} for p in ict.attach_components(pairs, rows)]
    req_group = {r: ict._group(p) for p in pairs for r in (p["request_aligned"], p["request_subverting"])}
    n_swaps = 0
    for seed in range(20):
        for p, s in zip(pairs, ict.swap_requests(pairs, np.random.default_rng(seed))):
            if s is None:
                continue
            n_swaps += 1
            same_thread = [q for q in pairs if ict._group(q) == ict._group(p)]
            assert all(s not in (q["request_aligned"], q["request_subverting"]) for q in same_thread)
            assert not (ict.content_tokens(s) & ict.content_tokens(p["instruction"]))  # e.g. not the same verb
            assert s.split()[0] != p["instruction"].split()[1]
            assert s.split()[-2] == p["task"]
    assert n_swaps > 0 and len(set(req_group.values())) < len({p["doc_id"] for p in pairs})  # threads span docs
    lone = [{"pair_id": "x", "doc_id": "d0", "task": "code", "context": "c", "instruction": "i",
             "request_aligned": "a", "request_subverting": "s"}]
    assert ict.swap_requests(lone, np.random.default_rng(0)) == [None]


def test_swap_audit_in_annotation_format():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_annotation_sheet import sheet_items
    rows, pairs = toy_data(n_docs=40)
    pairs = ict.attach_components(pairs, rows)
    audit = ict.swap_audit(pairs, seed=0)
    assert len(audit) == ict.SWAP_AUDIT_N and audit == ict.swap_audit(pairs, seed=0)
    assert len({a["pair_id"] for a in audit}) == len(audit)
    swaps = dict(zip([p["pair_id"] for p in pairs], ict.swap_requests(pairs, np.random.default_rng([0, 0]))))
    for a in audit:
        orig = a["notes"].split()[-1]
        assert a["request_subverting"] == swaps[orig] and a["source"] == "swap_audit:toy"
    assert len(sheet_items(audit, "ann1", "s")) == len(audit)


def test_select_grid_picks_best_with_tie_break(tmp_path):
    import json
    root = tmp_path / "results"
    scores = [0.5] * 12
    scores[3] = scores[7] = 0.9                                     # tie: the earlier grid point wins
    for i, v in enumerate(scores):
        d = root / "v3" / "ict_grid" / f"2026110{i % 9}-00000{i}_seed0"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"git_dirty": False, "args": {"grid_index": i}}))
        (d / "finished.json").write_text("{}")
        (d / "metrics.json").write_text(json.dumps({"best": {"value": v}}))
    out = tmp_path / "hp.json"
    choice = ict.select_grid(root, out)
    assert choice["grid_index"] == 3 and {k: choice[k] for k in ict.GRID_KEYS} == ict.GRID[3]
    with pytest.raises(FileExistsError):
        ict.select_grid(root, out)
    (next((root / "v3" / "ict_grid").glob("*00000" + "5_seed0")) / "finished.json").unlink()
    with pytest.raises(RuntimeError):
        ict.select_grid(root, tmp_path / "hp2.json")


# ─── resume and save/load ────────────────────────────────────────────────────

def test_resume_matches_uninterrupted(tok, tmp_path):
    rows, pairs = toy_data()
    cfg = ict.ICTConfig(lr=1e-3, epochs=2, batch_size=4, grad_accum=2, max_length=64)

    def fresh():
        return tiny_deberta(tok, seed=2, dropout=0.1)

    torch.manual_seed(5)
    m1 = fresh()
    torch.manual_seed(5)
    ict.train(cfg, m1, tok, rows, pairs, {}, tmp_path / "a", DEVICE, log=lambda m: None)
    torch.manual_seed(5)
    m2 = fresh()
    torch.manual_seed(5)
    st = ict.train(cfg, m2, tok, rows, pairs, {}, tmp_path / "b", DEVICE, log=lambda m: None, stop_after=3)
    assert st["step"] == 3 and st["epoch"] == 0
    m3 = fresh()  # a new process: fresh weights, state comes from state.pt
    ict.train(cfg, m3, tok, rows, pairs, {}, tmp_path / "b", DEVICE, log=lambda m: None)
    for (k, v1), (_, v3) in zip(m1.state_dict().items(), m3.state_dict().items()):
        assert torch.equal(v1, v3), k
    with pytest.raises(ValueError):
        ict.train(replace(cfg, lr=2e-3), fresh(), tok, rows, pairs, {}, tmp_path / "b", DEVICE)


def test_best_checkpoint_selected_on_val_and_reloads(tok, tmp_path):
    rows, pairs = toy_data()
    cfg = ict.ICTConfig(lr=3e-3, epochs=3, batch_size=4, grad_accum=1, max_length=64)
    evals = {"val": ("rows", rows), "val_pairs": ("pairs", ict.pairs_frame(pairs))}
    model = tiny_deberta(tok)
    st = ict.train(cfg, model, tok, rows, pairs, evals, tmp_path, DEVICE, log=lambda m: None)
    sel = [h["select"] for h in st["history"]]
    assert st["best"]["value"] == max(sel) and st["best"]["epoch"] == sel.index(max(sel))
    m2, tok2, cfg2 = ict.load_model(tmp_path / "best")
    assert cfg2 == cfg
    metrics, _ = ict.evaluate(m2, ict.make_encoder(tok2, m2, cfg2), cfg2, evals, DEVICE)
    assert ict.selection_score(metrics) == pytest.approx(st["best"]["value"])
    assert metrics["val"]["task_macro_auroc"] == pytest.approx(
        np.mean([metrics["val"]["by_task"][t]["roc_auc"] for t in ("email", "table")]))
    preds = ict.predictions(evals["val"][1], np.zeros(len(rows)), cfg, "val", "ict_full")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from runlog import validate_predictions
    validate_predictions(preds.assign(run_id="r"))


def test_cli_train_then_score(tok, tmp_path, monkeypatch):
    """main(): train writes a run with val predictions + metrics, resumes nothing, and score reloads best/."""
    import json
    import runlog
    rows, pairs = toy_data()
    split = tmp_path / "split_seed_0"
    split.mkdir()
    raw = rows.rename(columns={"task": "task_type", "request": "user_intent", "context": "text"})
    raw.to_csv(split / "train.csv", index=False)
    raw.to_csv(split / "val.csv", index=False)
    (tmp_path / "pairs.jsonl").write_text("\n".join(json.dumps(p) for p in pairs))
    arows, _ = toy_data(n_docs=4, seed=1, prefix="ag")
    agent = tmp_path / "agent"
    agent.mkdir()
    arows.rename(columns={"task": "task_type", "request": "user_intent", "context": "text"}).to_csv(
        agent / "train.csv", index=False)
    hp = tmp_path / "hp.json"
    monkeypatch.setattr(runlog, "start_run", lambda *a, **k: runlog.Run(*a, **k, results_root=tmp_path / "results"))
    monkeypatch.setattr(ict, "init_model", lambda cfg, device: (tiny_deberta(tok), tok))
    work = tmp_path / "work"
    common = ["--split_dir", str(split), "--agent_dir", str(agent), "--train_pairs", str(tmp_path / "pairs.jsonl"),
              "--val_pairs", str(tmp_path / "pairs.jsonl"), "--hparams", str(hp), "--device", "cpu", "--allow-dirty"]
    with pytest.raises(SystemExit):  # no selected hyperparameters yet
        ict.main(["train", *common, "--work_dir", str(tmp_path / "w0")])
    with pytest.raises(SystemExit):  # the grid runs on seed 0 only
        ict.main(["train", *common, "--grid_index", "0", "--seed", "1", "--work_dir", str(tmp_path / "w1")])
    hp.write_text(json.dumps({"margin_lambda": 2.0, "margin": 0.5, "lr": 3e-3, "grid_index": 11}))
    ict.main(["train", *common, "--work_dir", str(work), "--seed", "0", "--epochs", "2", "--max_length", "64"])
    run = next((tmp_path / "results" / "v3" / "ict_full").iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert set(m) >= {"best", "history", "val", "val_pairs"} and (run / "finished.json").exists()
    cfg = json.loads((run / "config.json").read_text())["args"]["ict_config"]
    assert (cfg["lr"], cfg["margin_lambda"], cfg["margin"], cfg["epochs"]) == (0.003, 2.0, 0.5, 2)
    assert m["val"]["n"] == len(rows)
    p = pd.read_csv(runlog.predictions_file(run))
    assert set(p["eval_set"]) == {"val", "val_pairs"} and len(p) == len(rows) + 2 * len(pairs)
    ict.main(["score", "--ckpt", str(work / "best"), "--eval", f"test={split / 'val.csv'}", "--exp", "ict_full", "--seed", "0",
              "--device", "cpu", "--allow-dirty"])
    s = next((tmp_path / "results" / "v3" / "ict_full_score").iterdir())
    ps = pd.read_csv(runlog.predictions_file(s))
    assert np.allclose(ps["prob"].values, p[p["eval_set"] == "val"]["prob"].values, atol=1e-6)
    with pytest.raises(SystemExit):  # full ICT without the agent-style data
        ict.main(["train", *[c for c in common if c != str(agent) and c != "--agent_dir"],
                  "--work_dir", str(tmp_path / "w2")])
