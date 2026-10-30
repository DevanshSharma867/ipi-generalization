"""Transfer-harness detector port (eval/transfer/detectors.py) against Liu's score_many.py and hf_judge.py."""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval" / "transfer"))
detectors = pytest.importorskip("detectors")
LIU_SCRIPTS = ROOT / "data" / "raw" / "benign-instruction-bench" / "scripts"
needs_liu = pytest.mark.skipif(not LIU_SCRIPTS.exists(), reason="Liu's repo is not cloned")


@needs_liu
def test_registry_matches_liu_detector_table():
    src = (LIU_SCRIPTS / "score_many.py").read_text()
    liu = ast.literal_eval(re.search(r"DETECTORS = (\{.*?\n\})", src, re.S).group(1))
    reg = detectors.load_registry()
    for name, (repo, kind, max_len, idx, remote) in liu.items():
        if name == "injecguard":  # alias of piguard, scored once
            continue
        e = reg[name]
        assert (e["repo"], e["max_len"], e["inj_index"], e["trust_remote_code"]) == (repo, max_len, idx, remote), name
        assert e["kind"] == ("sheltron" if kind.startswith("sheltron") else kind), name
        assert name not in detectors.SHELTRON or detectors.SHELTRON[name] == \
            {"sheltron": "plain", "sheltron-task": "task", "sheltron-generic": "generic"}[kind]
    # Liu pins only Horizon; ours must agree with his pin
    pins = ast.literal_eval(re.search(r"REVISIONS = (\{.*?\})", src).group(1))
    assert all(reg[n]["revision"] == r for n, r in pins.items())


@needs_liu
def test_judge_prompt_is_liu_llm_judge_sys():
    p = detectors.judge_prompt()
    assert p.startswith("You audit data returned by tools") and "Yes" in p


class FakeTok:
    """get_vocab / convert_tokens_to_string as a SentencePiece-style tokenizer would answer them."""
    VOCAB = {"Yes": 0, "▁yes": 1, "YES.": 2, "->Yes": 3, "No": 4, "▁no,": 5, "Nope": 6, "yess": 7, "Ġ": 8, "No!": 9}
    all_special_ids = []

    def get_vocab(self):
        return dict(self.VOCAB)

    def convert_tokens_to_string(self, toks):
        return "".join(toks).replace("▁", " ").replace("Ġ", " ")

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids):
        inv = {i: t for t, i in self.VOCAB.items()}
        return self.convert_tokens_to_string([inv[i] for i in ids])


def test_liu_rule_strips_punctuation_ours_does_not():
    yes, no = detectors.judge_token_ids(FakeTok(), "liu")
    assert sorted(yes) == [0, 1, 2, 3] and sorted(no) == [4, 5]
    yes, no = detectors.judge_token_ids(FakeTok(), "variants")
    assert sorted(yes) == [0, 1] and sorted(no) == [4]
