import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from judging import judge_token_ids, yes_no_class, yes_no_score


class FakeTok:
    """Minimal tokenizer: id -> text, with BPE-style space-prefixed variants and a special token."""
    vocab = ["<eos>", "Yes", "No", " Yes", " no", "yes", "YES", "Yesterday", "Nope", " the", "\nNo"]
    all_special_ids = [0]

    def __len__(self):
        return len(self.vocab)

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.vocab.index(text)]


def test_classes():
    assert yes_no_class(" Yes") == "yes" and yes_no_class("NO\n") == "no"
    assert yes_no_class("Yesterday") is None and yes_no_class("Nope") is None and yes_no_class("") is None


def test_variants_are_summed():
    lp = [("Yes", math.log(0.30)), (" yes", math.log(0.10)), ("No", math.log(0.20)), (" NO", math.log(0.20)),
          ("Maybe", math.log(0.20))]
    assert yes_no_score(lp) == pytest.approx(0.40 / 0.80)


def test_missing_classes():
    assert math.isnan(yes_no_score([("Maybe", 0.0)]))
    assert yes_no_score([("yes", math.log(0.5))]) == 1.0
    assert yes_no_score([(" no", math.log(0.5))]) == 0.0


def test_token_ids_per_rule():
    tok = FakeTok()
    assert judge_token_ids(tok, "two_token") == {1: "Yes", 2: "No"}
    assert set(judge_token_ids(tok, "variants")) == {1, 2, 3, 4, 5, 6, 10}
    with pytest.raises(ValueError):
        judge_token_ids(tok, "verbalized")


def test_two_token_matches_v2_softmax():
    """two_token through yes_no_score equals the V2 judge's softmax over the two logits."""
    rng = np.random.RandomState(0)
    tok = FakeTok()
    ids = judge_token_ids(tok, "two_token")
    for _ in range(20):
        logits = rng.randn(len(tok)) * 4
        logprobs = logits - np.log(np.exp(logits).sum())
        v2 = np.exp(logits[1]) / (np.exp(logits[1]) + np.exp(logits[2]))
        assert yes_no_score(zip(ids.values(), logprobs[list(ids)])) == pytest.approx(v2, rel=1e-12)
