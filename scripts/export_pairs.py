"""
export_pairs.py
===============
Kept TRAIN-side pairs -> the pair files ICT reads (`ict.py train --train_pairs / --val_pairs`; paper_plan A22, A32;
docs/pairs_spec.md v3 §8).

Input: one finished generate dir of mode train or val (scripts/gen_pairs.py generate; every quad of its docs file
present) and its self-check decisions (gen_pairs.py selfcheck: <dir>/selfcheck.jsonl, one decision for every valid
pair). Output: the valid pairs that the self-check kept, in generation order, one JSON record per line
(pairs.make_pair format), plus <out>.manifest.json (counts, sha256 of every input and of the output, the commit).

Refuses: a mode other than train/val (EVAL pairs are never exported for training), pairs whose doc is not from the
mode's split of split seed 0, a self-check that misses or adds pairs, and overwriting an existing file with different
content (an identical rerun is a no-op).

  python scripts/export_pairs.py --in_root data/pairs/train_gen --out data/pairs/ict_s0/train_pairs.jsonl
  python scripts/export_pairs.py --in_root data/pairs/val_gen   --out data/pairs/ict_s0/val_pairs.jsonl
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import pairs_gen as pg  # noqa: E402
from gen_pairs import generated_pairs  # noqa: E402

SPLIT_OF = {"train": "train", "val": "val"}
SPLIT_SEED = 0  # A18 Q1: every model compared on pairs is trained on split seed 0


def find_gen_dir(in_root: Path) -> Path:
    dirs = sorted(d for d in in_root.glob("v*_s0_*") if d.is_dir())
    if len(dirs) != 1:
        raise ValueError(f"expected one generate dir under {in_root}, found {[str(d) for d in dirs]}")
    return dirs[0]


def kept_pairs(gen_dir: Path, cfg: dict) -> tuple:
    """(kept pairs in generation order, mode, counts) for a finished, fully self-checked generate dir."""
    pairs, mode, docs_sha = generated_pairs(gen_dir, cfg)
    if mode not in SPLIT_OF:
        raise ValueError(f"{gen_dir}: mode {mode}; only train and val pairs are exported for ICT")
    want = f"v3_{SPLIT_OF[mode]}@split_seed{SPLIT_SEED}"
    bad = [p["pair_id"] for p in pairs if p.get("source") != want]
    if bad:
        raise ValueError(f"{gen_dir}: {len(bad)} pairs not from {want}: {bad[:5]}")
    dec = {d["pair_id"]: d for d in pg.read_shard(gen_dir / "selfcheck.jsonl")}
    ids = [p["pair_id"] for p in pairs]
    missing, extra = set(ids) - set(dec), set(dec) - set(ids)
    if missing or extra:
        raise ValueError(f"{gen_dir}: self-check misses {len(missing)} pairs and has {len(extra)} unknown ones")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{gen_dir}: duplicate pair ids")
    kept = [p for p in pairs if dec[p["pair_id"]]["kept"]]
    counts = {"valid_pairs": len(pairs), "kept_pairs": len(kept), "docs_sha256": docs_sha,
              "kept_by_task": {t: sum(p["task"] == t for p in kept) for t in sorted({p["task"] for p in pairs})}}
    return kept, mode, counts


def export(in_root: Path, out: Path, cfg: dict) -> dict:
    gen_dir = find_gen_dir(in_root)
    kept, mode, counts = kept_pairs(gen_dir, cfg)
    text = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in kept)
    if out.exists():
        if out.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"{out} exists with different content; not overwritten")
        print(f"{out}: identical, unchanged")
        return json.loads(out.with_suffix(".manifest.json").read_text())
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    tmp.write_text(text, encoding="utf-8")
    inputs = sorted(gen_dir.glob("shard_*.jsonl")) + [gen_dir / "selfcheck.jsonl", gen_dir / "selfcheck.meta.json"]
    manifest = {"file": out.name, "sha256": pg.sha256_file(tmp), "mode": mode, "split_seed": SPLIT_SEED,
                "source_dir": str(gen_dir.relative_to(ROOT) if gen_dir.is_relative_to(ROOT) else gen_dir),
                "inputs_sha256": {p.name: pg.sha256_file(p) for p in inputs if p.exists()},
                "config_sha256": cfg["sha256"], "selfcheck_prompt_sha256": cfg["selfcheck"]["prompt_sha256"],
                "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip(), **counts}
    tmp.replace(out)
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_root", type=Path, required=True, help="data/pairs/train_gen or data/pairs/val_gen")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=pg.CONFIG, help="config with a selfcheck section (v5 or later)")
    a = ap.parse_args()
    export(a.in_root, a.out, pg.load_config(a.config))


if __name__ == "__main__":
    main()
