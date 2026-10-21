"""V3 pipeline: insertion (D1), assertions, determinism. Full builds use a doc_limit to stay fast."""

import filecmp
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_pipeline_v3 as P
from probes import NoBoundary, V3_BOUNDARIES, form_tags, insert_preserving, middle_boundaries

DRAFT = ROOT / "data" / "benign" / "benign_v3_draft.json"
needs_data = pytest.mark.skipif(not (P.BENCH.exists() and P.TOKENIZER.exists() and DRAFT.exists()),
                                reason="raw BIPIA, tokenizer copy or draft benign pool missing")


def test_v3_insertion_keeps_context_and_records_offset():
    ctx = "first line\nsecond line\n```\ncode = 1\nmore = 2\n```\nlast line"
    for pos in ("start", "middle", "end"):
        for sep in ("\n\n", "\n"):
            text, off, kind = insert_preserving(ctx, "SPAN here", pos, sep, V3_BOUNDARIES)
            assert text[off:off + len("SPAN here")] == "SPAN here"
            assert text.replace("SPAN here", "", 1).replace(sep, "", 1 if pos != "middle" or kind == "line" else 2) \
                .replace("\n", "") == ctx.replace("\n", "")


def test_v3_middle_never_inside_fence():
    ctx = "intro\n```\na = 1\nb = 2\nc = 3\n```\noutro"
    bounds = middle_boundaries(ctx, "line")
    fence_body = range(ctx.index("a = 1"), ctx.rindex("```"))
    assert bounds and not any(b in fence_body for b in bounds)


def test_v3_fallback_order_and_no_boundary():
    email = "SUBJECT: hi|CONTENT: one sentence. another sentence."
    assert insert_preserving(email, "S", "middle", "\n", V3_BOUNDARIES)[2] == "field"
    assert insert_preserving("one. two three", "S", "middle", "\n", V3_BOUNDARIES)[2] == "sentence"
    with pytest.raises(NoBoundary):
        insert_preserving("singleword", "S", "middle", "\n", V3_BOUNDARIES)


def test_form_tags_matches_v2_span_features():
    t = form_tags('Could you check "this" 3 times?')
    assert t["ends_q"] and t["has_quote"] and t["has_digit"] and t["has_you"] and not t["ends_dot"]
    assert t["n_words"] == 6


@needs_data
def test_build_assertions_and_determinism(tmp_path):
    a = P.write(P.build(0, DRAFT, doc_limit=40), tmp_path / "a")
    b = P.write(P.build(0, DRAFT, doc_limit=40), tmp_path / "b")
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    match, mismatch, errors = filecmp.cmpfiles(a, b, names, shallow=False)
    assert not mismatch and not errors
    main = pd.concat([pd.read_csv(a / f"{s}.csv", keep_default_na=False) for s in ("train", "val", "test")])
    assert (main.groupby("doc_id").size() == 4).all()
    assert (main["n_tokens"] <= P.MAX_TOKENS).all()
    assert set(main["separator"]) <= {"\\n\\n", "\\n"}
    for r in main.itertuples():
        assert r.text[r.char_offset:r.char_offset + r.span_len] == r.span_text


@needs_data
def test_different_seed_changes_splits():
    a = P.build(0, DRAFT, doc_limit=40)["frames"]["main"]
    b = P.build(1, DRAFT, doc_limit=40)["frames"]["main"]
    sa = dict(zip(a["doc_id"], a["split"]))
    sb = dict(zip(b["doc_id"], b["split"]))
    assert sa != sb


def test_check_rejects_cross_split_attack_overlap():
    row = dict(task_type="email", label=1, row_type="injected", span_text="same attack", span_source="c",
               text="", char_offset=0, span_len=0, n_tokens=10, component="x")
    rows = []
    for split, d in (("train", "d1"), ("test", "d2")):
        for i, (lab, rt) in enumerate([(1, "injected"), (1, "injected"), (0, "clean_a"), (0, "clean_b")]):
            rows.append({**row, "split": split, "doc_id": d, "label": lab, "row_type": rt, "id": f"{d}{i}",
                         "text": f"{d}-{i}", "span_text": "same attack" if rt == "injected" and i == 0 else f"{d}{i}",
                         "span_source": f"{d}{i}", "char_offset": 0, "span_len": len(f"{d}-{i}")})
    main = pd.DataFrame(rows)
    main["span_text"] = [t if st == "same attack" else t for t, st in zip(main["text"], main["span_text"])]
    main.loc[main["row_type"] == "injected", "span_text"] = ["same attack", "a1", "same attack", "a2"]
    main["text"] = main["span_text"] + main["id"]
    main["span_len"] = main["span_text"].str.len()
    docs = pd.DataFrame({"doc_id": ["d1", "d2"], "split": ["train", "test"], "component": ["d1", "d2"]})
    with pytest.raises(AssertionError, match="injected overlap"):
        P.check(main, docs)
