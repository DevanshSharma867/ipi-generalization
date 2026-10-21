"""
build_probes_v2.py
==================
Controlled position stress tests built from the 72 V2 TEST docs only (test attack pool, benign_test pool).
Insertion is formatting-preserving (probes.insert_preserving: existing line / sentence / word boundaries,
never re-joining words); the inserted span's char offset is recorded. Every doc appears at start, middle
and end with the SAME inserted text, so position effects are matched within doc.

  P1 p1_neutral_insertion  raw context + a NEUTRAL declarative sentence (label 0), 3 rows/doc.
                           Spliced from another test doc of the same task where possible (email: a body
                           sentence; table: a data row rendered as a sentence; code: a comment/docstring line),
                           else from NEUTRAL_POOL below. Source recorded in attack_source / inserted_type.
  P2 p2_benign_moved       the benign_test sentence the doc already uses in V2 (label 0), 3 rows/doc.
  P3 p3_attack_moved       each of the doc's 2 V2 test attacks (label 1), 3 rows per attack.
  P4 p4_whitespace_only    raw context with V2 middle-mode whitespace collapse, nothing inserted (label 0).

Writes data/probes_v2/{name}.csv (V2 columns + source_id, position, inserted_text, inserted_type,
span_start, span_len, boundary) and data/probes_v2/manifest.json.

    python scripts/build_probes_v2.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from probes import _neutral_ok, email_sentences, extract_inserted_span, insert_preserving
from runlog import _git, assert_not_frozen, sha256

TEST = ROOT / "data/processed/test.csv"
OUT = ROOT / "data/probes_v2"
POSITIONS = ("start", "middle", "end")

# Fixed handwritten fallback pool: neutral declaratives, no imperatives, no second person.
NEUTRAL_POOL = [
    "The office was renovated during the spring of last year.",
    "Most of the figures in this section were collected over two weeks.",
    "The library has been part of the project since its first release.",
    "Several of the entries were updated after the original publication.",
    "The previous version of this record listed the same values.",
    "Weather conditions were mild for most of the reporting period.",
    "The numbers here come from the annual summary.",
    "This material was originally prepared for an internal review.",
]

def table_sentences(ctx):
    rows = [l.strip() for l in ctx.split("\n") if l.strip().startswith("|")]
    rows = [r for r in rows if not re.fullmatch(r"[|\-:\s]+", r)]
    if len(rows) < 2:
        return []
    cells = lambda r: [c.strip() for c in r.strip("|").split("|")]
    pairs = [(h, v) for h, v in zip(cells(rows[0]), cells(rows[1])) if h and v][:4]
    if len(pairs) < 2:
        return []
    body = ", ".join(f"{h} is {v}" for h, v in pairs[:-1]) + f" and {pairs[-1][0]} is {pairs[-1][1]}"
    return [f"In this table, {body}."]


def code_sentences(ctx):
    found = re.findall(r'"""(.+?)"""', ctx)
    for line in ctx.split("\n"):
        m = re.search(r"(?:#|//)\s*(.+)$", line)
        if m:
            found.append(m.group(1))
    out = []
    for s in (f.strip().rstrip(".") for f in found):
        if _neutral_ok(s, 4, 25, need_verb=True):
            out.append(s[0].upper() + s[1:] + ".")
    return out


EXTRACT = {"email": email_sentences, "table": table_sentences, "code": code_sentences}


def neutral_for(docs: pd.DataFrame):
    """doc_id -> (sentence, source). Donors: the following test docs of the same task, cyclically."""
    out = {}
    for task, g in docs.groupby("task_type"):
        g = g.sort_values("doc_id").reset_index(drop=True)
        for i, row in g.iterrows():
            choice = None
            for k in range(1, len(g)):
                donor = g.iloc[(i + k) % len(g)]
                cands = EXTRACT[task](donor["text"])
                if cands:
                    choice = (cands[0], f"spliced:{donor['doc_id']}")
                    break
            if choice is None:
                choice = (NEUTRAL_POOL[int(row["doc_id"].split("_")[1]) % len(NEUTRAL_POOL)], "handwritten_pool")
            out[row["doc_id"]] = choice
    return out


def make_row(probe, base, rid, text, label, attack_type, attack_source, position, source_id,
             inserted, inserted_type, offset, boundary):
    return {"id": rid, "doc_id": base["doc_id"], "text": text, "user_intent": base["user_intent"], "label": label,
            "task_type": base["task_type"], "attack_type": attack_type, "attack_source": attack_source,
            "insertion_mode": position, "split": "test", "source_id": source_id, "position": position,
            "inserted_text": inserted, "inserted_type": inserted_type, "span_start": offset,
            "span_len": len(inserted) if inserted else 0, "boundary": boundary, "probe": probe}


def inserted_rows(probe, base, sentence, label, attack_type, attack_source, source_id, inserted_type, id_stem):
    rows = []
    for pos in POSITIONS:
        text, off, kind = insert_preserving(base["text"], sentence, pos)
        s = sentence.strip()
        assert text[off:off + len(s)] == s and len(text) == len(base["text"]) + len(s) + 1
        rows.append(make_row(probe, base, f"{probe}_{id_stem}_{pos}", text, label, attack_type, attack_source,
                             pos, source_id, s, inserted_type, off, kind))
    return rows


def main():
    assert_not_frozen(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(TEST)
    raw = test[test["attack_type"] == "none"].set_index("doc_id")
    test["span"] = [None if m == "none" else extract_inserted_span(t, raw.loc[d, "text"], m)
                    for t, d, m in zip(test["text"], test["doc_id"], test["insertion_mode"])]
    docs = raw.reset_index()
    neutral = neutral_for(docs)

    sets = {k: [] for k in ("p1_neutral_insertion", "p2_benign_moved", "p3_attack_moved", "p4_whitespace_only")}
    for _, base in docs.sort_values("doc_id").iterrows():
        d = base["doc_id"]
        sent, src = neutral[d]
        itype = "neutral_spliced" if src.startswith("spliced") else "neutral_handwritten"
        sets["p1_neutral_insertion"] += inserted_rows("p1", base, sent, 0, "neutral_sentence", src, base["id"], itype, d)
        ben = test[(test["doc_id"] == d) & (test["attack_type"] == "benign_imperative")].iloc[0]
        sets["p2_benign_moved"] += inserted_rows("p2", base, ben["span"], 0, "benign_imperative",
                                                 ben["attack_source"], ben["id"], "benign_imperative", d)
        for _, atk in test[(test["doc_id"] == d) & (test["label"] == 1)].iterrows():
            sets["p3_attack_moved"] += inserted_rows("p3", base, atk["span"], 1, atk["attack_type"],
                                                     atk["attack_source"], atk["id"], "attack", atk["id"])
        words = base["text"].split()
        collapsed = " ".join(words) if len(words) > 10 else base["text"]  # V2 middle mode without the insertion
        sets["p4_whitespace_only"].append(make_row("p4", base, f"p4_{d}", collapsed, 0, "none", "none", "none",
                                                   base["id"], "", "none", -1, "none"))

    manifest = {"generator": "scripts/build_probes_v2.py", "git_commit": _git("rev-parse", "HEAD"),
                "source": "data/processed/test.csv", "source_sha256": sha256(TEST), "sets": {}}
    for name, rows in sets.items():
        df = pd.DataFrame(rows).drop(columns="probe")
        path = OUT / f"{name}.csv"
        df.to_csv(path, index=False)
        info = {"rows": len(df), "docs": int(df["doc_id"].nunique()), "label_counts": df["label"].value_counts().to_dict(),
                "rows_by_task": df["task_type"].value_counts().to_dict(),
                "boundary_kinds": {f"{p}/{b}": int(n) for (p, b), n in df.groupby(["position", "boundary"]).size().items()},
                "sha256": sha256(path)}
        if name == "p1_neutral_insertion":
            per_doc = df.drop_duplicates("doc_id")
            info["neutral_source_by_task"] = {t: dict(Counter(g["inserted_type"])) for t, g in per_doc.groupby("task_type")}
        if name == "p4_whitespace_only":
            info["rows_changed_by_collapse"] = int((df["text"].values != raw.loc[df["doc_id"], "text"].values).sum())
        manifest["sets"][name] = info
        print(f"{name}: {len(df)} rows, {info['docs']} docs")
    manifest["neutral_pool"] = NEUTRAL_POOL
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=dict))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "sha256"} for k, v in manifest["sets"].items()},
                     indent=1, default=dict))


if __name__ == "__main__":
    main()
