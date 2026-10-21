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
from typing import List, Optional, Sequence, Tuple


# Neutral-sentence extraction (moved from scripts/build_probes_v2.py, unchanged)
IMPERATIVE_START = {
    "please", "click", "call", "contact", "visit", "see", "check", "reply", "update", "use", "find", "read",
    "follow", "view", "sign", "log", "make", "download", "get", "take", "go", "enter", "create", "add",
    "remove", "set", "try", "run", "install", "change", "replace", "note", "let", "keep", "do", "don't",
    "open", "write", "print", "import", "define", "ensure", "remember", "bring", "send", "ask", "tell",
    "consider", "avoid", "fill", "submit", "start", "stop", "turn", "pay", "save", "share", "join", "book",
}
IMPERATIVE_ANYWHERE = ("please", "click", "reply", "contact us", "feel free", "let us", "let me", "don't",
                       "do not", "make sure", "unsubscribe", "log in", "sign in", "sign up", "http", "www.", "@")
DECLARATIVE_VERB = re.compile(r"\b(is|are|was|were|has|have|had|returns|contains|gives|will|would|can|does|"
                              r"doesn't|raises|works|means|creates|holds|stores)\b", re.I)


def _first_word(s):
    m = re.match(r"[A-Za-z']+", s)
    return m.group(0).lower() if m else ""


def _neutral_ok(s, lo, hi, need_verb=False):
    words = s.split()
    low = s.lower()
    return (lo <= len(words) <= hi and not any(c in s for c in "?!") and _first_word(s) not in IMPERATIVE_START
            and not any(t in low for t in IMPERATIVE_ANYWHERE) and (not need_verb or DECLARATIVE_VERB.search(s)))


def email_sentences(ctx):
    body = ctx.split("CONTENT:", 1)[1] if "CONTENT:" in ctx else ctx
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    return [p.strip() for p in parts if p.strip().endswith(".") and _neutral_ok(p.strip(), 6, 30)]


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


def insert_preserving(ctx: str, sentence: str, position: str, sep: Optional[str] = None,
                      boundaries: Optional[Sequence[str]] = None) -> Tuple[str, int, str]:
    """V2 probe behavior by default. With `boundaries` (V3), `sep` is the separator placed around the span and
    the middle boundary is the one nearest the midpoint of the first kind in `boundaries` that has any."""
    sentence = sentence.strip()
    if boundaries is not None:
        return _insert_v3(ctx, sentence, position, sep, boundaries)
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


# ─── V3 insertion (docs/v3_design.md D1) ──────────────────────────────────────

V3_BOUNDARIES = ("line", "paragraph", "field", "sentence")


class NoBoundary(ValueError):
    """The context has no internal boundary of any allowed kind (D1: such docs go to the `long` slice)."""


def _fence_mask(ctx: str) -> List[bool]:
    """Per character: inside a ``` fenced block (fence lines included)."""
    mask, in_fence = [], False
    for line in ctx.split("\n"):
        fence = line.strip().startswith("```")
        mask += [in_fence or fence] * (len(line) + 1)
        if fence:
            in_fence = not in_fence
    return mask[:len(ctx)]


def middle_boundaries(ctx: str, kind: str) -> List[int]:
    """Internal insertion offsets of one kind, never inside a code fence. line/paragraph: start of a line;
    field: an email field delimiter `|KEY:`; sentence: after `[.!?] `."""
    if kind == "line":
        b = _line_starts_outside_fences(ctx)
    elif kind == "paragraph":
        b = [s for s in _line_starts_outside_fences(ctx) if ctx[:s].endswith("\n\n")]
    elif kind == "field":
        b = [m.start() for m in re.finditer(r"\|(?=[A-Z][A-Z_ ]*:)", ctx)]
    elif kind == "sentence":
        b = [m.end() for m in re.finditer(r"[.!?] +", ctx)]
    else:
        raise ValueError(kind)
    if kind in ("line", "paragraph"):  # line starts are already fence-aware
        return [x for x in b if 0 < x < len(ctx)]
    mask = _fence_mask(ctx)
    return [x for x in b if 0 < x < len(ctx) and not mask[x] and not mask[x - 1]]


def has_internal_boundary(ctx: str, boundaries: Sequence[str] = V3_BOUNDARIES) -> bool:
    return any(middle_boundaries(ctx, k) for k in boundaries)


def _insert_v3(ctx: str, span: str, position: str, sep: str, boundaries: Sequence[str]) -> Tuple[str, int, str]:
    if sep not in ("\n\n", "\n"):
        raise ValueError(f"sep must be '\\n\\n' or '\\n', got {sep!r}")
    if position == "start":
        return span + sep + ctx, 0, "doc_start"
    if position == "end":
        return ctx + sep + span, len(ctx) + len(sep), "doc_end"
    if position != "middle":
        raise ValueError(position)
    mid = len(ctx) / 2
    for kind in boundaries:
        bounds = middle_boundaries(ctx, kind)
        if bounds:
            b = min(bounds, key=lambda x: (abs(x - mid), x))
            if kind in ("line", "paragraph"):  # the original newline(s) stay before the span
                return ctx[:b] + span + sep + ctx[b:], b, kind
            return ctx[:b] + sep + span + sep + ctx[b:], b + len(sep), kind
    raise NoBoundary("no internal boundary")


def code_chunks(ctx: str) -> List[str]:
    """'<prose line>\\n```...```' blocks: a non-empty line outside a fence directly followed by a fenced block."""
    lines, out, i = ctx.split("\n"), [], 0
    while i < len(lines) - 1:
        if lines[i].strip() and not lines[i].strip().startswith("```") and lines[i + 1].strip().startswith("```"):
            j = i + 2
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            if j < len(lines):
                out.append("\n".join(lines[i:j + 1]).strip())
                i = j + 1
                continue
        i += 1
    return out


# ─── form features of an inserted span (span_form gates, benign form targets) ─

CODE_CHARS = set("{}()[];=<>`\\")
FORM_BINARY = ("ends_q", "ends_dot", "ends_excl", "has_quote", "has_digit", "has_newline_or_code_char", "has_you")
FORM_NUMERIC = ("n_words", "n_chars", "cap_ratio")
YOU = re.compile(r"\b(you|your)\b", re.I)


def form_tags(span: str) -> dict:
    """The 9 V2 span_form_lr features (same definitions) plus has_you (D6)."""
    st = span.strip()
    letters = [ch for ch in st if ch.isalpha()]
    return {"n_words": len(st.split()), "n_chars": len(st), "ends_q": st.endswith("?"), "ends_dot": st.endswith("."),
            "ends_excl": st.endswith("!"), "has_quote": any(q in st for q in "\"'"),
            "has_digit": any(ch.isdigit() for ch in st),
            "has_newline_or_code_char": ("\n" in st) or any(ch in CODE_CHARS for ch in st),
            "cap_ratio": sum(ch.isupper() for ch in letters) / len(letters) if letters else 0.0,
            "has_you": bool(YOU.search(st))}
