"""
probes.py
=========
Insertion helpers for controlled probe sets.

- extract_inserted_span: recover the exact sentence V2's data_pipeline.insert_sentence inserted into a
  row, given the doc's raw context and the row's insertion_mode (inverse of insert_sentence).
- insert_preserving: formatting-PRESERVING insertion at an existing boundary (never re-joins words):
  start / end attach with the doc's own line separator; middle uses the line start (outside ``` code
  fences) closest to the character midpoint, else the closest sentence boundary, else the closest
  word boundary. Returns the new text, the span's char offset and the boundary kind.
"""

import re
from typing import Tuple


def extract_inserted_span(text: str, context: str, mode: str) -> str:
    """Inverse of data_pipeline.insert_sentence for one V2 row. Raises if the row does not match."""
    if mode == "start":
        suffix = "\n\n" + context
        if text.endswith(suffix):
            return text[: -len(suffix)]
    elif mode in ("middle", "end"):
        words = context.split()
        if mode == "middle" and len(words) > 10:
            mid = len(words) // 2
            head, tail = " ".join(words[:mid]) + "\n\n", "\n\n" + " ".join(words[mid:])
            if text.startswith(head) and text.endswith(tail) and len(text) >= len(head) + len(tail):
                return text[len(head): len(text) - len(tail)]
        else:
            prefix = context + "\n\n"
            if text.startswith(prefix):
                return text[len(prefix):]
    raise ValueError(f"row does not match insert_sentence(mode={mode!r})")


def _line_starts_outside_fences(ctx: str):
    starts, pos, in_fence = [], 0, False
    for line in ctx.split("\n"):
        if pos > 0 and not in_fence:
            starts.append(pos)
        if line.strip().startswith("```"):
            in_fence = not in_fence
        pos += len(line) + 1
    return [s for s in starts if 0 < s < len(ctx)]


def insert_preserving(ctx: str, sentence: str, position: str) -> Tuple[str, int, str]:
    sentence = sentence.strip()
    sep = "\n" if "\n" in ctx else " "
    if position == "start":
        return sentence + sep + ctx, 0, "start"
    if position == "end":
        return ctx + sep + sentence, len(ctx) + len(sep), "end"
    if position != "middle":
        raise ValueError(position)
    mid = len(ctx) / 2
    candidates = [("line", _line_starts_outside_fences(ctx), "\n"),
                  ("sentence", [m.end() for m in re.finditer(r"[.!?] +", ctx) if 0 < m.end() < len(ctx)], " "),
                  ("word", [m.end() for m in re.finditer(r"\s+", ctx) if 0 < m.end() < len(ctx)], " ")]
    for kind, bounds, joiner in candidates:
        if bounds:
            b = min(bounds, key=lambda x: (abs(x - mid), x))
            return ctx[:b] + sentence + joiner + ctx[b:], b, kind
    # No internal boundary at all (single token): fall back to the end
    return ctx + sep + sentence, len(ctx) + len(sep), "end_fallback"
