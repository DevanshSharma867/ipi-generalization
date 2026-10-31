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


def toy_data(n_docs=8, seed=0):
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
            rows.append({"id": f"{task}_{d}_{k}", "doc_id": f"{task}_{d}", "split_seed": 0, "task_type": task,
                         "label": y, "user_intent": f"what is in this {task}", "text": ctx,
                         "attack_category": "x" if y else "", "attack_source": "", "insertion_mode": "",
                         "clean_type": "" if y else "raw"})
        for q, (vi, vo) in enumerate([(v1, v2), (v2, v1)]):
            pairs.append({"pair_id": f"p{d}_{q}", "quad_id": f"q{d}", "doc_id": f"{task}_{d}", "task": task,
                          "context": f"{doc} note {vi} the {task}", "request_aligned": f"{vi} this {task}",
                          "request_subverting": f"{vo} this {task}"})
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
    assert ict.make_config("a").use_pairs is False and ict.make_config("a").margin_lambda == 0
    assert ict.make_config("d").request_input == "none" and ict.make_config("e").request_input == "concat"
    assert len({tuple(sorted(vars(ict.make_config(k)).items())) for k in ict.ABLATIONS}) == len(ict.ABLATIONS)
    with pytest.raises(ValueError):
        ict.ICTConfig(use_pairs=False)              # margin and swap need pairs
    with pytest.raises(ValueError):
        ict.ICTConfig(request_input="none")         # pair members would be identical inputs
    assert ict.ICTConfig(backbone="modernbert-large").windowed and not ict.ICTConfig().windowed


def test_every_ablation_changes_items_inputs_or_loss(tok):
    """Each preset differs from the full recipe in what it trains on, what the model sees, or the loss value."""
    rows, pairs = toy_data()
    model = tiny_deberta(tok)

    def fingerprint(cfg):
        items = ict.epoch_items(cfg, rows, pairs if cfg.use_pairs else [], 0)
        enc = ict.make_encoder(tok, model, cfg)
        members = sorted(m for it in items for m in it["members"])
        ids = [tuple(enc(q, c)[0]["input_ids"]) for q, c, _ in members]
        with torch.no_grad():
            loss = ict.batch_loss(model, enc, items, cfg, DEVICE)[0].item()
        return members, ids, round(loss, 6)

    full = fingerprint(ict.make_config("full"))
    for k in "abcde":
        cfg = ict.make_config(k)
        assert fingerprint(cfg) != full, k
    # specific behaviours
    a_items = ict.epoch_items(ict.make_config("a"), rows, [], 0)
    assert all(len(it["members"]) == 1 for it in a_items) and len(a_items) == len(rows)
    b, c = ict.make_config("b"), ict.make_config("c")
    assert any(len(it["members"]) == 3 for it in ict.epoch_items(b, rows, pairs, 0))      # swap on, no margin
    assert all(len(it["members"]) <= 2 for it in ict.epoch_items(c, rows, pairs, 0))      # margin on, no swap
    enc_d = ict.make_encoder(tok, model, ict.make_config("d"))
    assert enc_d("summarize this email", "alpha beta") == enc_d("list this email", "alpha beta")
    enc_full = ict.make_encoder(tok, model, ict.make_config("full"))
    assert enc_full("summarize this email", "alpha beta") != enc_full("list this email", "alpha beta")
    enc_e = ict.make_encoder(tok, model, ict.make_config("e"))
    assert enc_e("list this email", "alpha beta")[0]["input_ids"] == tok("Intent: list this email | Context: alpha beta")["input_ids"]


def test_request_swap_draws_other_docs_same_task_deterministically():
    _, pairs = toy_data()
    rng = lambda: np.random.default_rng([0, 3])  # noqa: E731
    s1, s2 = ict.swap_requests(pairs, rng()), ict.swap_requests(pairs, rng())
    assert s1 == s2
    own = {p["pair_id"]: {pp["request_aligned"] for pp in pairs if pp["doc_id"] == p["doc_id"]} for p in pairs}
    task_reqs = {t: {r for pp in pairs if pp["task"] == t for r in (pp["request_aligned"], pp["request_subverting"])}
                 for t in ("email", "table")}
    for p, s in zip(pairs, s1):
        assert s in task_reqs[p["task"]] and s.split()[-1] == p["task"]
    items = ict.epoch_items(ict.ICTConfig(), pd.DataFrame(columns=["request", "context", "label"]), pairs, 0)
    for it in items:
        assert [m[2] for m in it["members"]] == [0, 1, 1] and it["pairs"] == [(0, 1), (0, 2)]
    assert own  # the doc check is in swap_requests; the test data has one doc per quad


def test_swap_never_uses_own_document():
    pairs = [{"pair_id": f"p{i}", "doc_id": f"d{i // 2}", "task": "email", "context": "c",
              "request_aligned": f"a{i // 2}", "request_subverting": f"s{i // 2}"} for i in range(6)]
    for seed in range(20):
        for p, s in zip(pairs, ict.swap_requests(pairs, np.random.default_rng(seed))):
            assert s[1:] != p["doc_id"][1:]
    lone = [{"pair_id": "x", "doc_id": "d0", "task": "code", "context": "c", "request_aligned": "a",
             "request_subverting": "s"}]
    assert ict.swap_requests(lone, np.random.default_rng(0)) == [None]


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
    cfg = ict.ICTConfig(lr=3e-3, epochs=3, batch_size=4, grad_accum=1, max_length=64,
                        select_metric="val_pairs:pair_accuracy")
    evals = {"val": ("rows", rows), "val_pairs": ("pairs", ict.pairs_frame(pairs))}
    model = tiny_deberta(tok)
    st = ict.train(cfg, model, tok, rows, pairs, evals, tmp_path, DEVICE, log=lambda m: None)
    sel = [h["select"] for h in st["history"]]
    assert st["best"]["value"] == max(sel) and st["best"]["epoch"] == sel.index(max(sel))
    m2, tok2, cfg2 = ict.load_model(tmp_path / "best")
    assert cfg2 == cfg
    metrics, _ = ict.evaluate(m2, ict.make_encoder(tok2, m2, cfg2), cfg2, evals, DEVICE)
    assert metrics["val_pairs"]["pair_accuracy"] == pytest.approx(st["best"]["value"])
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
    monkeypatch.setattr(runlog, "start_run", lambda *a, **k: runlog.Run(*a, **k, results_root=tmp_path / "results"))
    monkeypatch.setattr(ict, "init_model", lambda cfg, device: (tiny_deberta(tok), tok))
    work = tmp_path / "work"
    ict.main(["train", "--split_dir", str(split), "--train_pairs", str(tmp_path / "pairs.jsonl"),
              "--val_pairs", str(tmp_path / "pairs.jsonl"), "--work_dir", str(work), "--seed", "0", "--epochs", "2",
              "--lr", "0.003", "--max_length", "64", "--device", "cpu", "--allow-dirty"])
    run = next((tmp_path / "results" / "v3" / "ict_full").iterdir())
    m = json.loads((run / "metrics.json").read_text())
    assert set(m) >= {"best", "history", "val", "val_pairs"} and (run / "finished.json").exists()
    cfg = json.loads((run / "config.json").read_text())["args"]["ict_config"]
    assert cfg["lr"] == 0.003 and cfg["epochs"] == 2 and cfg["use_pairs"] is True
    p = pd.read_csv(run / "predictions.csv")
    assert set(p["eval_set"]) == {"val", "val_pairs"} and len(p) == len(rows) + 2 * len(pairs)
    ict.main(["score", "--ckpt", str(work / "best"), "--eval", f"test={split / 'val.csv'}", "--exp", "ict_full", "--seed", "0",
              "--device", "cpu", "--allow-dirty"])
    s = next((tmp_path / "results" / "v3" / "ict_full_score").iterdir())
    ps = pd.read_csv(s / "predictions.csv")
    assert np.allclose(ps["prob"].values, p[p["eval_set"] == "val"]["prob"].values, atol=1e-6)
    with pytest.raises(SystemExit):  # an ablation that uses pairs without --train_pairs
        ict.main(["train", "--split_dir", str(split), "--work_dir", str(tmp_path / "w2"), "--allow-dirty"])
