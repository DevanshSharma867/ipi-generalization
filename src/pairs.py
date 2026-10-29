"""
pairs.py
========
Query-counterfactual intent pairs (claim C3; docs/pairs_spec.md). A pair is one context (a document or tool
output with exactly one inserted instruction) and two user requests: under the ALIGNED request following the
instruction serves the request (label 0), under the SUBVERTING request it deviates from it (label 1).

Pairs come in crossed quads: on one document, two requests R1, R2 and two instructions I1 (serves R1, deviates
from R2) and I2 (serves R2, deviates from R1) give the pairs (doc + I1; R1 aligned, R2 subverting) and
(doc + I2; R2 aligned, R1 subverting). Every request string carries both labels once, so a request-only model
has pair accuracy exactly 0.5 on complete quads (paper_plan P3(i)), and the context is byte-identical within a
pair, so a content-only model is exactly 0.5 too (P3(ii)).

Shared by scripts/build_pairs_pilot.py (and the generators to come), scripts/make_annotation_sheet.py and
scripts/agreement.py.
"""

import hashlib
import re
from collections import defaultdict
from typing import Dict, List

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from probes import V3_BOUNDARIES, form_tags, insert_preserving

LABELS = {"aligned": 0, "subverting": 1}
ANNOTATION_LABELS = ("aligned", "subverting", "unclear")
COPY_NGRAM = 5  # a request may not share a word 5-gram with its pair's instruction (spec rule Q3)
# Spec rule E4: markers that make a content-only detector flag the aligned member for its form alone
INJECTION_MARKERS = re.compile(r"\b(ignore|disregard|instead|previous instructions?|above instructions?|system|"
                               r"new task|override)\b|assistant\s*:", re.I)
MAX_INSTRUCTION_WORD_DIFF = 8  # I1 vs I2 of a quad (spec §6)
MAX_REQUEST_LENGTH_RATIO = 2.0  # R1 vs R2, in words (spec §6)
PAIR_FIELDS = ["pair_id", "quad_id", "doc_id", "task", "source", "generator", "context", "instruction",
               "position", "separator", "boundary_kind", "char_offset", "request_aligned", "request_subverting",
               "overlap_aligned", "overlap_subverting", "overlap_class", "form_tags", "doc_sha256", "notes"]

_stem = PorterStemmer().stem
_WORD = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> set:
    """Lower-cased alphanumeric words of >= 2 characters, minus sklearn's English stop words, Porter-stemmed."""
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) >= 2 and w not in ENGLISH_STOP_WORDS}


def overlap(a: str, b: str) -> float:
    """Jaccard similarity of the content-token sets of a and b (0 when both are empty)."""
    x, y = content_tokens(a), content_tokens(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def overlap_class(request_aligned: str, request_subverting: str, instruction: str) -> str:
    """'explicit' if the aligned request shares more content words with the instruction than the subverting
    request does, else 'subtle' (equal or inverted overlap): a word-overlap heuristic cannot solve a subtle pair."""
    return "explicit" if overlap(request_aligned, instruction) > overlap(request_subverting, instruction) else "subtle"


def shared_ngram(a: str, b: str, n: int = COPY_NGRAM) -> bool:
    """True if a and b share a lower-cased word n-gram (a request that copies its instruction)."""
    grams = lambda t: {tuple(w[i:i + n]) for w in [_WORD.findall(t.lower())] for i in range(len(w) - n + 1)}
    return bool(grams(a) & grams(b))


def make_pair(pair_id: str, quad_id: str, doc: dict, instruction: str, request_aligned: str,
              request_subverting: str, position: str, sep: str, source: str, generator: str, notes: str = "") -> dict:
    """One pair record: the instruction inserted into doc['context'] exactly as the V3 builder inserts a span."""
    instruction = instruction.strip()
    context, off, kind = insert_preserving(doc["context"], instruction, position, sep, V3_BOUNDARIES)
    return {"pair_id": pair_id, "quad_id": quad_id, "doc_id": doc["doc_id"], "task": doc["task"], "source": source,
            "generator": generator, "context": context, "instruction": instruction, "position": position,
            "separator": sep.encode("unicode_escape").decode(), "boundary_kind": kind, "char_offset": off,
            "request_aligned": request_aligned.strip(), "request_subverting": request_subverting.strip(),
            "overlap_aligned": round(overlap(request_aligned, instruction), 4),
            "overlap_subverting": round(overlap(request_subverting, instruction), 4),
            "overlap_class": overlap_class(request_aligned, request_subverting, instruction),
            "form_tags": form_tags(instruction),
            "doc_sha256": hashlib.sha256(doc["context"].encode()).hexdigest(), "notes": notes}


def validate(pairs: List[dict]) -> List[str]:
    """Errors (empty list = valid) for the mechanical rules of the spec: fields, instruction placement, Q3 (no
    copying), E4 (no injection markers), §6 lengths, complete crossed quads. E1-E3 and the labels need a human."""
    errors, quads = [], defaultdict(list)
    ids = [p["pair_id"] for p in pairs]
    if len(ids) != len(set(ids)):
        errors.append("duplicate pair_id")
    for p in pairs:
        pid = p["pair_id"]
        missing = [f for f in PAIR_FIELDS if f not in p]
        if missing:
            errors.append(f"{pid}: missing {missing}")
            continue
        quads[p["quad_id"]].append(p)
        ins, ctx, off = p["instruction"], p["context"], p["char_offset"]
        if ctx[off:off + len(ins)] != ins:
            errors.append(f"{pid}: instruction not at char_offset {off}")
        if ctx.count(ins) != 1:
            errors.append(f"{pid}: instruction occurs {ctx.count(ins)} times in the context")
        ra, rs = p["request_aligned"], p["request_subverting"]
        if not ra or not rs or ra == rs:
            errors.append(f"{pid}: requests empty or identical")
        for role, r in (("aligned", ra), ("subverting", rs)):
            if shared_ngram(r, ins):
                errors.append(f"{pid}: {role} request copies a {COPY_NGRAM}-gram of the instruction")
        if p["overlap_class"] != overlap_class(ra, rs, ins):
            errors.append(f"{pid}: overlap_class is stale")
        if INJECTION_MARKERS.search(ins):
            errors.append(f"{pid}: instruction contains an injection marker ({INJECTION_MARKERS.search(ins).group(0)!r}; rule E4)")
        na, ns = len(ra.split()), len(rs.split())
        if na and ns and max(na, ns) / min(na, ns) > MAX_REQUEST_LENGTH_RATIO:
            errors.append(f"{pid}: request lengths {na} and {ns} words differ by more than {MAX_REQUEST_LENGTH_RATIO}x")
    for q, ps in quads.items():
        if len(ps) != 2:
            errors.append(f"quad {q}: {len(ps)} pairs (a quad has exactly 2)")
            continue
        a, b = ps
        if a["doc_id"] != b["doc_id"] or a["doc_sha256"] != b["doc_sha256"]:
            errors.append(f"quad {q}: pairs on different documents")
        if (a["request_aligned"], a["request_subverting"]) != (b["request_subverting"], b["request_aligned"]):
            errors.append(f"quad {q}: requests are not crossed (R1/R2 must swap roles)")
        if a["instruction"] == b["instruction"]:
            errors.append(f"quad {q}: both pairs use the same instruction")
        if abs(len(a["instruction"].split()) - len(b["instruction"].split())) > MAX_INSTRUCTION_WORD_DIFF:
            errors.append(f"quad {q}: instructions differ by more than {MAX_INSTRUCTION_WORD_DIFF} words")
    return errors


def members(pairs: List[dict]) -> List[dict]:
    """The two members of every pair: pair_id, role, request, label (0 aligned, 1 subverting)."""
    return [{"pair_id": p["pair_id"], "quad_id": p["quad_id"], "task": p["task"], "overlap_class": p["overlap_class"],
             "role": role, "request": p[f"request_{role}"], "label": LABELS[role]}
            for p in pairs for role in ("aligned", "subverting")]


def complete_quads(pair_ids_ok: set, pairs: List[dict]) -> Dict[str, List[str]]:
    """quad_id -> its pair ids, for quads whose both pairs are in pair_ids_ok. Dropping half a quad would give
    its requests a single label, so a quad is kept or dropped whole (spec §6)."""
    by_quad = defaultdict(list)
    for p in pairs:
        by_quad[p["quad_id"]].append(p["pair_id"])
    return {q: ids for q, ids in by_quad.items() if len(ids) == 2 and set(ids) <= pair_ids_ok}
