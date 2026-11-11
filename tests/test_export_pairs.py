"""scripts/export_pairs.py: kept TRAIN-side pairs -> ICT pair files (A32)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "src" / "models"), str(ROOT / "tests")]
import pairs_gen as pg  # noqa: E402
import export_pairs as ex  # noqa: E402
from gen_pairs import selfcheck_score  # noqa: E402
from test_pairs_gen import GOOD, Mock, doc, FakeScorer  # noqa: E402

OTHER = {"r1": "Who received the payment?", "r2": "Write a short poem about this email.",
         "i1": "Name the recipient of the transfer.", "i2": "Compose four rhyming lines about the message.",
         "why_i1_serves_r1": "a", "why_i1_deviates_r2": "b", "why_i2_serves_r2": "c", "why_i2_deviates_r1": "d"}


class TwoOutputs(Mock):
    """GOOD for the first call, OTHER afterwards (so a doc's second quad is not a near-duplicate)."""
    def __call__(self, prompts, seeds):
        out = super().__call__(prompts, seeds)
        text = json.dumps(GOOD if len(self.calls) == 1 else OTHER)
        return [(text, s) for _, s in out]


@pytest.fixture(scope="module")
def cfg():
    return pg.load_config()


def gen_dir(tmp_path, cfg, mode="val", split="val", n_docs=1, selfcheck=True, flip=None):
    docs = [doc(i, split) for i in range(n_docs)]
    dfile = tmp_path / f"{mode}_s0.jsonl"
    dfile.write_text("".join(json.dumps(d) + "\n" for d in docs))
    d = tmp_path / f"{mode}_gen" / "v5_s0_abc"
    d.mkdir(parents=True)
    gen_mode = mode if mode != "eval" else "eval"
    pg.run_shard(docs, gen_mode, "gen-train-mistral" if mode != "eval" else "gen-eval-gemma", cfg,
                 d / "shard_000_of_001.jsonl", TwoOutputs(), batch_size=1)
    (d / "shard_000_of_001.meta.json").write_text(json.dumps({"mode": mode, "docs": str(dfile), "docs_sha256": "x"}))
    if selfcheck:
        from gen_pairs import generated_pairs
        pairs, _, _ = generated_pairs(d, cfg)
        selfcheck_score(pairs, FakeScorer(flip=flip), cfg["selfcheck"]["prompt_path"].read_text(), 4,
                        d / "selfcheck.jsonl")
        (d / "selfcheck.meta.json").write_text("{}")
    return d.parent


def test_export_keeps_only_selfcheck_kept_pairs(cfg, tmp_path):
    import ict
    root = gen_dir(tmp_path, cfg, flip=OTHER["r1"])  # pairs whose members include OTHER's r1 are labelled wrong
    out = tmp_path / "ict_s0" / "val_pairs.jsonl"
    m = ex.export(root, out, cfg)
    pairs = ict.load_pairs(out)  # the format ICT reads
    assert m["valid_pairs"] > m["kept_pairs"] == len(pairs) > 0
    assert all(OTHER["r1"] not in (p["request_aligned"], p["request_subverting"]) for p in pairs)
    assert m["sha256"] == pg.sha256_file(out) and m["mode"] == "val"
    assert ex.export(root, out, cfg)["sha256"] == m["sha256"]  # identical rerun: no-op
    out.write_text(out.read_text() + "\n{}\n")
    with pytest.raises(FileExistsError):
        ex.export(root, out, cfg)


def test_export_refuses_incomplete_selfcheck(cfg, tmp_path):
    root = gen_dir(tmp_path, cfg)
    sc = root / "v5_s0_abc" / "selfcheck.jsonl"
    sc.write_text("".join(sc.read_text().splitlines(keepends=True)[1:]))
    with pytest.raises(ValueError, match="self-check misses 1"):
        ex.export(root, tmp_path / "o.jsonl", cfg)
    (root / "v5_s0_abc" / "selfcheck.jsonl").unlink()
    with pytest.raises(ValueError, match="misses"):
        ex.export(root, tmp_path / "o.jsonl", cfg)


def test_export_refuses_wrong_split_and_two_dirs(cfg, tmp_path):
    root = gen_dir(tmp_path, cfg)
    p = root / "v5_s0_abc" / "shard_000_of_001.jsonl"
    p.write_text(p.read_text().replace("v3_val@split_seed0", "v3_val@split_seed1"))
    with pytest.raises(ValueError, match="not from v3_val@split_seed0"):
        ex.export(root, tmp_path / "o.jsonl", cfg)
    (root / "v5_s0_def").mkdir()
    with pytest.raises(ValueError, match="expected one generate dir"):
        ex.export(root, tmp_path / "o.jsonl", cfg)
