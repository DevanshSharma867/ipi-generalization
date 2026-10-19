"""
tokenizer_check.py
==================
Diagnose the "incorrect regex pattern" warning seen when deberta.py loads the DeBERTa-v3 tokenizer.

Loads the tokenizer exactly as src/models/deberta.py does (resolve_model_name_or_path + AutoTokenizer),
captures any warning, then compares its tokenization of 20 V2 rows (email/table/code; raw, benign and
start/middle attack rows; texts containing newlines) against reference tokenizers:
  - use_fast=False (sentencepiece slow tokenizer), if this transformers version provides it
  - fix_mistral_regex=True, the fix the warning recommends, if accepted
and checks whether "\\n" and "\\n\\n" survive as distinct tokens.
Results are logged through runlog to results/v3/tokenizer_check/<run_id>/metrics.json.

    python scripts/tokenizer_check.py
"""

import difflib
import io
import logging
import sys
import warnings
from contextlib import redirect_stderr
from pathlib import Path

import pandas as pd
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "models")]
from runlog import start_run
import deberta  # sets HF offline env exactly as training does
from preprocess import build_combined_text
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

DATA = ROOT / "data/processed/dataset.csv"


def load(path, **kw):
    """Load a tokenizer, returning (tokenizer or None, captured warnings/logs, error)."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    hf_logger = hf_logging.get_logger()
    hf_logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as w, redirect_stderr(buf):
            warnings.simplefilter("always")
            tok = AutoTokenizer.from_pretrained(path, **kw)
        msgs = buf.getvalue() + "".join(str(x.message) + "\n" for x in w)
        return tok, msgs, None
    except Exception as e:  # report, do not crash: some kwargs may not exist in this version
        return None, buf.getvalue(), f"{type(e).__name__}: {e}"
    finally:
        hf_logger.removeHandler(handler)


def pick_rows(df):
    """7 email, 7 table, 6 code rows: raw x2 (with newlines), benign x1, start attack x2, middle attack x2."""
    rows = []
    for task, n in (("email", 7), ("table", 7), ("code", 6)):
        t = df[df.task_type == task]
        raw = t[(t.attack_type == "none") & t.text.str.contains("\n")].head(2)
        ben = t[t.attack_type == "benign_imperative"].head(1)
        start = t[(t.label == 1) & (t.insertion_mode == "start")].head(2)
        mid = t[(t.label == 1) & (t.insertion_mode == "middle")].head(2)
        rows.append(pd.concat([raw, ben, start, mid]).head(n))
    return pd.concat(rows).reset_index(drop=True)


def token_diff(a, b) -> int:
    """Tokens not covered by matching blocks (max over the two sequences)."""
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    same = sum(bl.size for bl in sm.get_matching_blocks())
    return max(len(a), len(b)) - same


def newline_probe(tok):
    out = {}
    for name, s in (("space", "alpha beta"), ("nl", "alpha\nbeta"), ("nlnl", "alpha\n\nbeta")):
        out[name] = tok.convert_ids_to_tokens(tok(s, add_special_tokens=False)["input_ids"])
    out["nl_distinct_from_space"] = out["nl"] != out["space"]
    out["nlnl_distinct_from_nl"] = out["nlnl"] != out["nl"]
    return out


def main():
    path = deberta.resolve_model_name_or_path("microsoft/deberta-v3-base")
    run = start_run(version="v3", exp="tokenizer_check", seed=0,
                    args={"model_path": path, "rows": "20 fixed V2 rows (see pick_rows)"},
                    data_version="v2", data_files=[DATA])
    df = pick_rows(pd.read_csv(DATA))
    texts = [build_combined_text(r) for _, r in df.iterrows()]
    run.log(f"rows: {df.task_type.value_counts().to_dict()}, with newline: {sum(chr(10) in t for t in texts)}")

    variants = {"as_deberta_py": {}, "slow_use_fast_false": {"use_fast": False}, "fix_mistral_regex": {"fix_mistral_regex": True}}
    loaded, report = {}, {"model_path": path, "tokenizers": {}}
    for name, kw in variants.items():
        tok, msgs, err = load(path, **kw)
        info = {"kwargs": kw, "loaded": tok is not None, "class": type(tok).__name__ if tok else None,
                "error": err, "regex_warning": "incorrect regex pattern" in msgs}
        if tok is not None:
            loaded[name] = tok
            info["newline_probe"] = newline_probe(tok)
        report["tokenizers"][name] = info
        run.log(f"{name}: loaded={info['loaded']} class={info['class']} regex_warning={info['regex_warning']} error={err}")

    base = loaded["as_deberta_py"]
    ids = {n: [t(x, add_special_tokens=True)["input_ids"] for x in texts] for n, t in loaded.items()}
    toks = {n: [t.convert_ids_to_tokens(i) for i in ids[n]] for n, t in loaded.items()}
    report["comparisons"] = {}
    for name in loaded:
        if name == "as_deberta_py":
            continue
        per_row = [token_diff(toks["as_deberta_py"][i], toks[name][i]) for i in range(len(texts))]
        report["comparisons"][f"as_deberta_py_vs_{name}"] = {
            "rows_differing": sum(d > 0 for d in per_row), "rows": len(texts),
            "differing_tokens_total": sum(per_row), "differing_tokens_per_row": per_row,
            "base_tokens_total": sum(len(x) for x in toks["as_deberta_py"]),
            "same_ids_rows": sum(a == b for a, b in zip(ids["as_deberta_py"], ids[name])),
        }
        run.log(f"as_deberta_py vs {name}: {report['comparisons'][f'as_deberta_py_vs_{name}']['rows_differing']}/20 rows "
                f"differ, {sum(per_row)} differing tokens")
    # Do newlines change the token ids of real rows? Compare each text with its newlines replaced by spaces.
    report["rows_where_newlines_change_ids"] = {
        n: sum(t(x)["input_ids"] != t(x.replace("\n", " "))["input_ids"] for x in texts if "\n" in x)
        for n, t in loaded.items()}
    report["rows_with_newline"] = sum("\n" in x for x in texts)
    # Ground truth: the original DeBERTa-v3 tokenizer is the raw sentencepiece model shipped with the checkpoint
    sp = spm.SentencePieceProcessor(model_file=str(Path(path) / "spm.model"))
    sp_toks = [sp.encode(x, out_type=str) for x in texts]
    report["vs_raw_sentencepiece"] = {
        name: {"rows_identical": sum(t.tokenize(x) == s for x, s in zip(texts, sp_toks)), "rows": len(texts),
               "differing_tokens_total": sum(token_diff(t.tokenize(x), s) for x, s in zip(texts, sp_toks))}
        for name, t in loaded.items()}
    report["raw_sentencepiece_newline_probe"] = {s: sp.encode(s, out_type=str)
                                                 for s in ("alpha beta", "alpha\nbeta", "alpha\n\nbeta")}
    run.log(f"vs raw sentencepiece: {report['vs_raw_sentencepiece']}")
    report["row_ids"] = df["id"].tolist()
    run.save_metrics(report, headline=("rows_differing_vs_slow",
                                       report["comparisons"].get("as_deberta_py_vs_slow_use_fast_false", {}).get("rows_differing", float("nan"))))
    run.log(f"newline probe (as_deberta_py): {report['tokenizers']['as_deberta_py']['newline_probe']}")
    run.log(f"rows where newlines change ids: {report['rows_where_newlines_change_ids']} of {report['rows_with_newline']}")
    run.finish()
    print(f"wrote {run.dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
