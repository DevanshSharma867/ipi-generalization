"""
gate2.py
========
Gate 2 of the transfer harness: reproduce Liu (2026)'s scores (paper_plan A17, protocol A19).

Detectors: PIGuard, Prompt Guard 2 86M, Horizon-Labs (rev af3c52e), Qwen2.5-7B judge (4-bit, Liu's Yes/No rule).
Sets: AD-Matched with AD-Clean (its negatives in Liu's table), TB-Matched, BIPIA.

  score    (GPU; resumable, existing score files are skipped) writes <work>/run{1,2,3}/<detector>__<set>.json:
             run1  Liu's unmodified score_many.py / hf_judge.py in his env, through a throwaway HF cache whose
                   refs/main point at our pinned revisions
             run2  our port (eval/transfer/detectors.py) in his env
             run3  our port in our env
  compare  (CPU) checks the metric port against numbers_many.json on Liu's released files, then applies A19:
             PASS per run if every AUROC is within ±0.005 of numbers_many.json and every TPR@1%FPR is inside Liu's
             bootstrap CI (numbers.json, same metric definition) or within ±0.01 where he gives none. Per-row max
             |Δprob| vs his released files is a diagnostic. run2 must equal run1 within 1e-6 (port check).
           Logged through runlog to results/transfer/liu_repro_gate2/<run_id>/ with every score file and both envs.

    python eval/transfer/gate2.py score   --work <dir> --liu_py <python> --our_py <python>
    python eval/transfer/gate2.py compare --work <dir> --liu_py <python> --our_py <python>
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval" / "transfer"))

LIU = ROOT / "data" / "raw" / "benign-instruction-bench"
SETS_DIR = ROOT / "data" / "transfer" / "sets"
ENCODERS = ["piguard", "promptguard2-86m", "horizon-base"]
JUDGE = "judge-qwen"
DETECTORS = ENCODERS + [JUDGE]
SETS = ["adclean", "admatched", "bipia", "tbeval"]
# Liu's released files: results/many/<detector>__<set>.json; judge files by set
JUDGE_FILES = {"adclean": "agentdojo_clean", "admatched": "agentdojo_pairs", "bipia": "bipia_pairs",
               "tbeval": "taubench_eval"}
# numbers.json (compute_all.py) bootstrap CIs: detector -> his display name; set -> his section
CI_NAMES = {"piguard": "PIGuard", JUDGE: "Judge (Qwen2.5-7B)"}
CI_SECTIONS = {"admatched_tpr1_matchedneg": "AgentDojo", "bipia_tpr1": "BIPIA"}
AUROC_TOL, TPR_TOL, PORT_TOL = 0.005, 0.01, 1e-6
DIAG_BOUND = {"encoder": 1e-3, "judge": 0.02}
GPU_MEM = "18GiB"


def load(path):
    return [json.loads(line) for line in open(path)]


# ─── Liu's metric definitions (analysis/compute_many.py) ─────────────────────

def tpr_at(y, p, t=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= t
    return float(tpr[ok].max()) if ok.any() else 0.0


def liu_metrics(scores, bip_keep, tb_y):
    """compute_many.py's AUROC and TPR@1%FPR keys for one detector. scores: {set: (probs, labels)}."""
    row = {}
    for s in ["admatched", "bipia"]:
        y, p = np.array(scores[s][1]), np.array(scores[s][0])
        if s == "bipia":
            y, p = y[bip_keep], p[bip_keep]
        else:  # AD-Matched against AD-Clean negatives
            row[f"{s}_tpr1_matchedneg"] = tpr_at(y, p)
            neg = np.array(scores["adclean"][0])
            p = np.r_[p[y == 1], neg]
            y = np.r_[np.ones(int((y == 1).sum())), np.zeros(len(neg))]
        row[f"{s}_auroc"] = float(roc_auc_score(y, p))
        row[f"{s}_tpr1"] = tpr_at(y, p)
    p = np.array(scores["tbeval"][0])
    assert len(p) == len(tb_y)
    row["tbeval_auroc"] = float(roc_auc_score(tb_y, p))
    row["tbeval_tpr1"] = tpr_at(tb_y, p)
    return row


def set_masks():
    bip = load(SETS_DIR / "bipia_pairs.jsonl")
    bip_keep = np.array([r["label"] == 0 or r.get("attack_split") == "test" for r in bip])
    tb_y = np.array([r["label"] for r in load(SETS_DIR / "taubench_eval.jsonl")])
    return bip_keep, tb_y


def released(det, s):
    """(probs, labels) of Liu's released score file."""
    if det == JUDGE:
        d = json.load(open(LIU / "results" / f"hfjudge_Qwen2.5-7B-Instruct_{JUDGE_FILES[s]}.json"))
    else:
        d = json.load(open(LIU / "results" / "many" / f"{det}__{s}.json"))
    return d["probs"], d["labels"]


def ours(run_dir, det, s):
    d = json.load(open(run_dir / f"{det}__{s}.json"))
    return d["probs"], d["labels"]


# ─── score ───────────────────────────────────────────────────────────────────

def registry():
    from detectors import load_registry
    return load_registry()


def snapshot(hf_home, repo, rev):
    return Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}" / "snapshots" / rev


def throwaway_cache(work, hf_home, reg):
    """HF cache whose refs/main is our pin for each model, so Liu's unpinned from_pretrained(repo) loads it."""
    cache = work / "hf_run1"
    pins = {}
    for det in DETECTORS:
        repo, rev = reg[det]["repo"], reg[det]["revision"]
        real = Path(hf_home) / "hub" / f"models--{repo.replace('/', '--')}"
        if not (real / "snapshots" / rev).is_dir():
            sys.exit(f"{repo}@{rev} is not in {hf_home}; run scripts/prefetch_models.sh first")
        d = cache / "hub" / real.name
        (d / "refs").mkdir(parents=True, exist_ok=True)
        for sub in ("blobs", "snapshots"):
            if not (d / sub).exists():
                (d / sub).symlink_to(real / sub)
        (d / "refs" / "main").write_text(rev)
        pins[det] = {"repo": repo, "refs_main": rev}
    json.dump(pins, open(work / "run1_refs_main.json", "w"), indent=1)
    return cache


def sh(cmd, env=None, cwd=None, ok_codes=(0,)):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    rc = subprocess.run(list(map(str, cmd)), env=env, cwd=cwd).returncode
    if rc not in ok_codes:
        sys.exit(f"command failed with exit {rc}")
    return rc


def cmd_score(a):
    reg = registry()
    work = Path(a.work).resolve()
    hf_home = os.environ["HF_HOME"]
    base = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", GPU_MEM=GPU_MEM, PYTHONHASHSEED="0")

    # run 1: Liu's scripts, unmodified, run from a directory holding the rebuilt sets (they read relative paths)
    r1 = work / "run1"
    (r1 / "results" / "many").mkdir(parents=True, exist_ok=True)
    from detectors import SETS as FILES
    for f in FILES.values():
        if not (r1 / f).exists():
            (r1 / f).symlink_to(SETS_DIR / f)
    env1 = dict(base, HF_HOME=str(throwaway_cache(work, hf_home, reg)), DEVICE="cuda")
    for det in ENCODERS:
        want = [r1 / "results" / "many" / f"{det}__{s}.json" for s in SETS]
        if all(w.exists() for w in want):
            continue
        # score_many.py loops over all of its sets and stops with FileNotFoundError at llmail_eval.jsonl, which
        # is not part of this study; every set before it (ours included) is written first
        sh([a.liu_py, LIU / "scripts" / "score_many.py", det], env=env1, cwd=r1, ok_codes=(0, 1))
        missing = [w.name for w in want if not w.exists()]
        if missing:
            sys.exit(f"run1 {det}: missing {missing}")
    jsnap = snapshot(hf_home, reg[JUDGE]["repo"], reg[JUDGE]["revision"])
    for s in SETS:
        out = r1 / f"{JUDGE}__{s}.json"
        if not out.exists():
            sh([a.liu_py, LIU / "scripts" / "hf_judge.py", jsnap, FILES[s], out], env=env1, cwd=r1)
    for det in ENCODERS:  # one naming scheme for every run
        for s in SETS:
            dst = r1 / f"{det}__{s}.json"
            if not dst.exists():
                shutil.copy(r1 / "results" / "many" / f"{det}__{s}.json", dst)

    # runs 2 and 3: the port, in his env and in ours
    for run, py in (("run2", a.liu_py), ("run3", a.our_py)):
        sh([py, ROOT / "eval" / "transfer" / "detectors.py", "--detectors", *DETECTORS, "--sets", *SETS,
            "--sets_dir", SETS_DIR, "--out", work / run, "--judge_rule", "liu", "--gpu_mem", GPU_MEM],
           env=dict(base, HF_HOME=hf_home))


# ─── compare ─────────────────────────────────────────────────────────────────

def reference_check(bip_keep, tb_y):
    """compute_many.py metrics on Liu's released files must equal numbers_many.json exactly."""
    table = json.load(open(LIU / "analysis" / "numbers_many.json"))["table"]
    bad = []
    for det in DETECTORS:
        m = liu_metrics({s: released(det, s) for s in SETS}, bip_keep, tb_y)
        bad += [f"{det} {k}: {v} vs {table[det][k]}" for k, v in m.items() if v != table[det][k]]
    return bad, table


def ci_ref():
    """{(detector, metric): [lo, hi]} from numbers.json, only where its point value equals numbers_many.json's."""
    nj = json.load(open(LIU / "analysis" / "numbers.json"))
    table = json.load(open(LIU / "analysis" / "numbers_many.json"))["table"]
    out = {}
    for det, pretty in CI_NAMES.items():
        for metric, section in CI_SECTIONS.items():
            e = nj.get(section, {}).get(pretty)
            if e and abs(e["tpr_at_1fpr"] - table[det][metric]) < 1e-12:
                out[(det, metric)] = e["tpr_at_1fpr_ci"]
    return out


def check_run(run_dir, table, cis, bip_keep, tb_y):
    checks, ok = {}, True
    for det in DETECTORS:
        m = liu_metrics({s: ours(run_dir, det, s) for s in SETS}, bip_keep, tb_y)
        for k, v in m.items():
            ref = table[det][k]
            if k.endswith("_auroc"):
                rule, passed = f"±{AUROC_TOL}", abs(v - ref) <= AUROC_TOL
            elif (det, k) in cis:
                lo, hi = cis[(det, k)]
                rule, passed = f"Liu CI [{lo:.4f}, {hi:.4f}]", lo <= v <= hi
            else:
                rule, passed = f"±{TPR_TOL}", abs(v - ref) <= TPR_TOL
            checks[f"{det}/{k}"] = dict(ours=v, liu=ref, diff=v - ref, rule=rule, pass_=bool(passed))
            ok &= bool(passed)
    return {"pass": ok, "checks": checks}


def row_diffs(run_dir, ref_fn):
    out = {}
    for det in DETECTORS:
        for s in SETS:
            a, la = ours(run_dir, det, s)
            b, lb = ref_fn(det, s)
            # hf_judge.py writes all-zero labels for AD-Clean (no label field); the port writes None
            assert la is None or lb is None or la == lb, f"{run_dir.name} {det} {s}: label sequence differs"
            d = np.abs(np.array(a) - np.array(b))
            out[f"{det}/{s}"] = dict(n=len(d), max_abs=float(d.max()), mean_abs=float(d.mean()),
                                     flips_at_05=int(((np.array(a) >= 0.5) != (np.array(b) >= 0.5)).sum()))
    return out


def env_versions(py):
    code = ("import importlib.metadata as m, json, sys; "
            "print(json.dumps({'python': sys.version.split()[0], "
            "**{d.metadata['Name'].lower(): d.version for d in m.distributions()}}))")
    return json.loads(subprocess.run([py, "-c", code], capture_output=True, text=True, check=True).stdout)


def cmd_compare(a):
    from runlog import start_run
    work = Path(a.work).resolve()
    manifest = json.load(open(SETS_DIR / "manifest.json"))
    reg = registry()
    run = start_run(version="transfer", exp="liu_repro_gate2", seed=0,
                    args={"work": "<scratch>/" + work.name, "detectors": DETECTORS, "sets": SETS,
                          "revisions": {d: reg[d]["revision"] for d in DETECTORS}, "judge_rule": "liu",
                          "gpu_mem": GPU_MEM, "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"),
                          "upstream_commit": manifest["upstream"]["commit"],
                          "tolerances": {"auroc": AUROC_TOL, "tpr_no_ci": TPR_TOL, "port": PORT_TOL,
                                         "row_diag": DIAG_BOUND}},
                    data_version=manifest["data_version"],
                    data_files=[SETS_DIR / "manifest.json"], allow_dirty=a.allow_dirty)
    bip_keep, tb_y = set_masks()
    bad, table = reference_check(bip_keep, tb_y)
    if bad:
        run.log("metric port does not reproduce numbers_many.json on Liu's released files:\n  " + "\n  ".join(bad))
        run.save_metrics({"reference_check": bad, "pass": False}, headline=("gate2_pass", 0.0))
        run.finish()
        sys.exit(1)
    run.log("reference check: compute_many metrics on Liu's released files equal numbers_many.json")
    cis = ci_ref()
    res = {"reference_check": "match", "ci_from_numbers_json": {f"{d}/{k}": v for (d, k), v in cis.items()},
           "runs": {}, "row_diffs_vs_liu": {}}
    for name in ("run1", "run2", "run3"):
        rd = work / name
        res["runs"][name] = check_run(rd, table, cis, bip_keep, tb_y)
        res["row_diffs_vs_liu"][name] = row_diffs(rd, released)
        for k, d in res["row_diffs_vs_liu"][name].items():
            kind = "judge" if k.startswith(JUDGE) else "encoder"
            d["exceeds_expected_bound"] = d["max_abs"] > DIAG_BOUND[kind]
        run.log(f"{name}: {'PASS' if res['runs'][name]['pass'] else 'FAIL'}")
        for k, c in res["runs"][name]["checks"].items():
            run.log(f"  {k:45s} ours {c['ours']:.4f} liu {c['liu']:.4f} ({c['rule']}) {'ok' if c['pass_'] else 'FAIL'}")
        for k, d in res["row_diffs_vs_liu"][name].items():
            run.log(f"  rows {k:28s} max|Δ| {d['max_abs']:.2e} flips@0.5 {d['flips_at_05']}"
                    + ("  EXCEEDS expected bound" if d["exceeds_expected_bound"] else ""))
    port = row_diffs(work / "run2", lambda det, s: ours(work / "run1", det, s))
    res["port_check"] = {"pass": all(d["max_abs"] <= PORT_TOL for d in port.values()), "rows": port}
    run.log(f"port check (run2 vs run1, ≤ {PORT_TOL}): {'PASS' if res['port_check']['pass'] else 'FAIL'}")
    env3 = row_diffs(work / "run3", lambda det, s: ours(work / "run2", det, s))
    res["env_effect_run3_vs_run2"] = env3
    res["envs"] = {"liu": env_versions(a.liu_py), "ours": env_versions(a.our_py)}
    res["run1_refs_main"] = json.load(open(work / "run1_refs_main.json"))
    res["pass"] = res["runs"]["run1"]["pass"] and res["runs"]["run3"]["pass"] and res["port_check"]["pass"]
    for name in ("run1", "run2", "run3"):
        dst = run.dir / "scores" / name
        dst.mkdir(parents=True, exist_ok=True)
        for det in DETECTORS:
            for s in SETS:
                shutil.copy(work / name / f"{det}__{s}.json", dst)
    run.log(f"gate2: {'PASS' if res['pass'] else 'FAIL'}")
    run.save_metrics(res, headline=("gate2_pass", float(res["pass"])))
    run.finish()
    sys.exit(0 if res["pass"] else 1)


def main():
    from runlog import add_run_args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "compare", "refcheck"])
    ap.add_argument("--work", default=".")
    ap.add_argument("--liu_py")
    ap.add_argument("--our_py", default=sys.executable)
    a = add_run_args(ap).parse_args()
    if a.cmd == "score":
        cmd_score(a)
    elif a.cmd == "compare":
        cmd_compare(a)
    else:  # metric port vs numbers_many.json only (CPU, seconds; no run record)
        bad, _ = reference_check(*set_masks())
        print("\n".join(bad) if bad else "reference check: match")
        print("CIs from numbers.json:", ci_ref())
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
