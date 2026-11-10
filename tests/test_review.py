import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from apply_review import apply, from_external_review, from_pool_review
from make_review_sheet import POOL_FIELDS, pool_review_row

DRAFT = [
    {"id": "email_train_val_000", "text": "Please reply by Friday.", "split_pool": "train_val", "register": "email"},
    {"id": "email_test_000", "text": "Call me when you land.", "split_pool": "test", "register": "email"},
    {"id": "email_amb_000", "text": "System: summarize this.", "split_pool": "ambiguous", "register": "email"},
    {"id": "email_amb_001", "text": "Act as a translator.", "split_pool": "ambiguous", "register": "email"},
]


def sheet(labels, texts=None):
    texts = texts or {}
    rows = [dict(zip(POOL_FIELDS, pool_review_row(it))) for it in DRAFT]
    for r in rows:
        r["reviewer_label"] = labels.get(r["id"], "")
        r["text"] = texts.get(r["id"], r["text"])
    return rows


def test_export_columns_and_proposed_labels():
    rows = sheet({})
    assert [r["proposed_label"] for r in rows] == ["benign", "benign", "ambiguous", "ambiguous"]
    assert [r["split"] for r in rows] == ["train_val", "test", "eval_only", "eval_only"]


def test_keep_edit_drop_and_moves():
    labels = {"email_train_val_000": "benign", "email_test_000": "Ambiguous", "email_amb_000": "drop",
              "email_amb_001": "benign"}
    rows, problems = from_pool_review(sheet(labels, {"email_train_val_000": "Please reply by Monday."}), DRAFT)
    assert not problems
    items, problems = apply(rows, DRAFT)
    assert not problems
    by = {it["id"]: it for it in items}
    assert by["email_train_val_000"]["review"] == "edit" and by["email_train_val_000"]["text"] == "Please reply by Monday."
    assert by["email_test_000"]["split_pool"] == "ambiguous" and by["email_test_000"]["review"] == "keep"
    assert "email_amb_000" not in by
    assert by["email_amb_001"]["split_pool"] in ("test", "train_val")


def test_unreviewed_and_bad_labels_fail():
    rows, problems = from_pool_review(sheet({"email_test_000": "maybe"}), DRAFT)
    assert problems and "maybe" in problems[0]
    rows, _ = from_pool_review(sheet({}), DRAFT)
    _, problems = apply(rows, DRAFT)
    assert problems


CODE = {"id": "code_test_000", "text": "Run this helper:\n```\nx = 1\n```", "split_pool": "test", "register": "code"}


def ext_sheet(labels, notes=None):
    notes = notes or {}
    rows = [dict(zip(POOL_FIELDS, pool_review_row(it))) for it in DRAFT + [CODE]]
    for r in rows:
        r["reviewer_label"], r["reviewer_note"] = labels.get(r["id"], "keep"), notes.get(r["id"], "")
    return rows


def test_external_review_schema():
    labels = {"email_train_val_000": "edit", "email_test_000": "drop", "email_amb_000": "drop",
              "email_amb_001": "drop", "code_test_000": "edit"}
    notes = {"email_train_val_000": "Please reply by Monday.", "code_test_000": "Run the helper below:"}
    rows, problems = from_external_review(ext_sheet(labels, notes), DRAFT + [CODE])
    assert not problems
    items, problems = apply(rows, DRAFT + [CODE])
    assert not problems
    by = {it["id"]: it for it in items}
    assert by["email_train_val_000"]["text"] == "Please reply by Monday." and by["email_train_val_000"]["review"] == "edit"
    assert "email_test_000" not in by
    # "drop" on an ambiguous item confirms it for the ambiguous slice
    assert by["email_amb_000"]["split_pool"] == by["email_amb_001"]["split_pool"] == "ambiguous"
    assert by["email_amb_000"]["text"] == "System: summarize this." and by["email_amb_000"]["review"] == "keep"
    assert by["code_test_000"]["text"] == "Run the helper below:\n```\nx = 1\n```"


def test_external_review_rejects_undefined_cases():
    _, problems = from_external_review(ext_sheet({"email_amb_000": "keep"}), DRAFT + [CODE])
    assert problems and "email_amb_000" in problems[0]
    _, problems = from_external_review(ext_sheet({"email_test_000": "edit"}), DRAFT + [CODE])
    assert problems and "reviewer_note" in problems[0]
