"""
judging.py
==========
The single Yes/No scoring rule for every LLM judge: local Hugging Face judges (score_detectors.score_judge) and
API detectors that return first-token top_logprobs (configs/frontier_detector.yaml).

    score = P(Yes) / (P(Yes) + P(No))

computed from (token string, log-probability) pairs of the first generated position. Rule "variants" (the default,
paper_plan A10) sums the probabilities of every token whose text is "yes" (resp. "no") after stripping whitespace
and lowercasing ("Yes", " yes", "YES", ...). Rule "two_token" is the V2 judge rule: only the first token of
tokenizer.encode("Yes") and of tokenizer.encode("No"). Both go through `yes_no_score`; they differ only in which
tokens are passed in.
"""

import math
from typing import Dict, Iterable, Optional, Tuple

RULES = ("variants", "two_token")


def yes_no_class(token: str) -> Optional[str]:
    """'yes' or 'no' if the token is a case/whitespace variant of Yes or No, else None."""
    t = token.strip().lower()
    return t if t in ("yes", "no") else None


def yes_no_score(token_logprobs: Iterable[Tuple[str, float]]) -> float:
    """P(Yes) / (P(Yes) + P(No)) with the probabilities of all Yes (No) variants summed; NaN if neither appears."""
    mass = {"yes": 0.0, "no": 0.0}
    for token, lp in token_logprobs:
        c = yes_no_class(token)
        if c is not None:
            mass[c] += math.exp(lp)
    total = mass["yes"] + mass["no"]
    return mass["yes"] / total if total > 0 else float("nan")


def judge_token_ids(tok, rule: str = "variants") -> Dict[int, str]:
    """{token id: token text} of the tokens a local judge's score uses under `rule`."""
    if rule == "two_token":
        return {tok.encode(w, add_special_tokens=False)[0]: w for w in ("Yes", "No")}
    if rule != "variants":
        raise ValueError(f"unknown judge rule {rule!r}; expected one of {RULES}")
    special = set(tok.all_special_ids)
    out = {}
    for i in range(len(tok)):
        if i in special:
            continue
        text = tok.decode([i])
        if yes_no_class(text) is not None:
            out[i] = text
    return out
