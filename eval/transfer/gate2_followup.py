"""
gate2_followup.py
=================
Gate 2 follow-up (docs/deviations.md, 2026-10-30; paper_plan A19, A20). Three investigations after the Gate 2 FAIL
(results/transfer/liu_repro_gate2/20261030-160800_seed0):

1. AD-Matched / AD-Shift pairing. Liu's collect_agentdojo_pairs.py picks each user task's injection task with
   hash(task_id), and his released scores do not match our PYTHONHASHSEED=0 build. His script is run unmodified with
   `hash` shadowed in its globals: once per candidate (every user task gets injection-task index k), then once with
   the recovered map. Per (suite, user task) the candidate whose Prompt Guard 2 86M scores match his released file
   within 1e-4 is kept. Validation: PIGuard and Horizon on the recovered sets must match his files within 1e-3.
   Some tool outputs carry a wall-clock timestamp (AgentDojo's last_modified), so they differ between any two builds:
   these "time-dependent" rows are found by rebuilding the seeded pairing (W/timecheck) and comparing with the seeded
   set, and are left out of matching and of the validation bound (reported separately).
2. Environment effect: PIGuard, Horizon and PG2 on a fixed 500-row BIPIA subset at batch size 1 and 16, in his env and
   ours; our V2 DeBERTa checkpoints at batch size 1 and 16 in our env.
3. Judge: a repeat of the 4-bit judge on BIPIA (run-to-run noise), one run at GPU_MEM=6GiB, and bf16 vs 4-bit AUROC
   on AD-Matched (A20).

    gen      (AgentDojo env) --set S --out_dir D (--cands | --map M | --map_from_rows F)   rebuild, shadowed hash()
    match    --set S --work W                                         PG2 candidates vs Liu -> W/map_<set>.json
    subset   --work W                                                 W/bipia500/bipia_pairs.jsonl (+ indices)
    deberta_bs --ckpts C... --data CSV --out D                        bs 1 vs 16 probabilities per checkpoint
    report   --part {batch,pairing,judge} --work W --gate2 G --liu_py P --our_py P
                                                     run record: results/transfer/gate2_followup_<part>/
    envfix   --work W --device D           our env with tokenizer.json / local RoPE theta vs Liu's env (BIPIA subset)
    residual --work W --pairing_run R      PIGuard rows over the validation bound vs AgentDojo's set-ordered
                                           participant lists (calendar_client: list(set(participants)), hash-seed
                                           dependent) -> results/transfer/gate2_followup_residual/
"""

import argparse
import json
import os
import runpy
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIU = ROOT / "data" / "raw" / "benign-instruction-bench"
SUITES = ["workspace", "travel", "banking", "slack"]
# collect_agentdojo_pairs.py arguments as in scripts/build_transfer_sets.sh
ATTACKS = {"admatched": "important_instructions,ignore_previous,direct",
           "adshift": "tool_knowledge,injecagent,system_message,important_instructions_no_names,"
                      "important_instructions_wrong_model_name"}
FILES = {"admatched": "agentdojo_pairs.jsonl", "adshift": "agentdojo_pairs_shift.jsonl"}
MATCH_TOL = 1e-4      # PG2 vs Liu per row, for a candidate to match a user task
VALID_TOL = 1e-3      # PIGuard / Horizon vs Liu per row on the recovered sets (A19 encoder bound)
ENCODERS = ["piguard", "promptguard2-86m", "horizon-base"]
SUBSET_N, SUBSET_SEED = 500, 0


def load(path):
    return [json.loads(line) for line in open(path)]


# ─── gen (AgentDojo env; stdlib + agentdojo only) ────────────────────────────

def run_builder(set_name, out, choose):
    """Liu's collect_agentdojo_pairs.py, unmodified, with hash(task_id) -> choose(suite, task_id)."""
    def fake_hash(task_id):
        return choose(sys._getframe(1).f_globals["suite_name"], task_id)
    script = str(LIU / "scripts" / "collect_agentdojo_pairs.py")
    argv, sys.argv = sys.argv, [script, ATTACKS[set_name], str(out)]
    try:
        g = runpy.run_path(script, init_globals={"hash": fake_hash}, run_name="__main__")
    finally:
        sys.argv = argv
    g["out"].close()  # the script never closes its output file


def cmd_gen(a):
    from agentdojo.task_suite import get_suite
    out_dir = Path(a.out_dir)
    if a.map_from_rows:  # the pairing of an existing build (time-dependence check)
        order = {s: sorted(get_suite("v1.2", s).injection_tasks) for s in SUITES}
        m = {}
        for r in load(a.map_from_rows):
            m.setdefault(r["suite"], {})[r["task"]] = {"index": order[r["suite"]].index(r["inj"])}
        out_dir.mkdir(parents=True, exist_ok=True)
        run_builder(a.set, out_dir / FILES[a.set], lambda s, t: m.get(s, {}).get(t, {}).get("index", 0))
        return
    if a.map:
        m = json.load(open(a.map))["map"]
        out_dir.mkdir(parents=True, exist_ok=True)
        # user tasks with no rows in Liu's file get index 0: no row depends on their choice
        run_builder(a.set, out_dir / FILES[a.set], lambda s, t: m.get(s, {}).get(t, {}).get("index", 0))
        return
    n = max(len(get_suite("v1.2", s).injection_tasks) for s in SUITES)
    for k in (a.k if a.k is not None else range(n)):  # candidate k: each user task gets injection task k mod count
        d = out_dir / f"c{k:02d}"
        if (d / FILES[a.set]).exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (FILES[a.set] + ".tmp")
        run_builder(a.set, tmp, lambda s, t, k=k: k)
        tmp.replace(d / FILES[a.set])


# ─── match ───────────────────────────────────────────────────────────────────

def liu_file(det, s):
    return json.load(open(LIU / "results" / "many" / f"{det}__{s}.json"))


def time_dependent(work, s):
    """Row indices whose text differs between two builds with the same pairing (wall-clock content)."""
    a = load(ROOT / "data" / "transfer" / "sets" / FILES[s])
    b = load(Path(work) / "timecheck" / FILES[s])
    assert len(a) == len(b) and all((x["suite"], x["task"], x["inj"], x["step"]) == (y["suite"], y["task"], y["inj"],
                                                                                    y["step"]) for x, y in zip(a, b))
    return sorted(i for i, (x, y) in enumerate(zip(a, b)) if x["text"] != y["text"])


def cmd_match(a):
    import numpy as np
    work = Path(a.work)
    ref = liu_file("promptguard2-86m", a.set)
    P, Ly = np.array(ref["probs"]), ref["labels"]
    cdirs = sorted((work / "cands" / a.set).glob("c*"))
    groups, cand, aligned = None, {}, {}
    for d in cdirs:
        rows = load(d / FILES[a.set])
        ok = len(rows) == len(P) and [r["label"] for r in rows] == Ly
        aligned[d.name] = ok
        if not ok:
            continue
        g = defaultdict(list)
        for i, r in enumerate(rows):
            g[(r["suite"], r["task"])].append(i)
        keys = [(r["suite"], r["task"], r["attack"], r["step"]) for r in rows]
        if groups is None:
            groups, keys0 = g, keys
        assert keys == keys0, f"{d.name}: row keys differ from {cdirs[0].name}"
        probs = np.array(json.load(open(d / "scores" / f"promptguard2-86m__{a.set}.json"))["probs"])
        inj = {k: rows[ix[0]]["inj"] for k, ix in g.items()}
        cand[d.name] = (probs, inj)
    tdep = set(time_dependent(work, a.set))
    res, unmatched, ambiguous, time_only = defaultdict(dict), [], 0, []
    for (suite, task), all_ix in sorted(groups.items()):
        ix = [i for i in all_ix if i not in tdep]
        dist = {c: float(np.abs(p[ix or all_ix] - P[ix or all_ix]).max()) for c, (p, _) in cand.items()}
        if not ix:  # every row carries a timestamp: closest candidate, flagged
            time_only.append(f"{suite}/{task}")
            best = min(dist, key=dist.get)
            res[suite][task] = dict(inj=cand[best][1][(suite, task)], index=int(best[1:]), max_abs=dist[best],
                                    matching_injs=[], n_rows=len(all_ix), time_dependent_only=True)
            continue
        hits = sorted((d, c) for c, d in dist.items() if d <= MATCH_TOL)
        injs = sorted({cand[c][1][(suite, task)] for _, c in hits})
        if not hits:
            unmatched.append(f"{suite}/{task} (closest {min(dist.values()):.2e})")
            continue
        ambiguous += len(injs) > 1
        best = hits[0][1]
        inj_id = cand[best][1][(suite, task)]
        from_ids = sorted({cand[c][1][(suite, task)] for c in cand})
        res[suite][task] = dict(inj=inj_id, index=int(best[1:]), max_abs=hits[0][0], matching_injs=injs,
                                n_rows=len(all_ix), n_rows_used=len(ix), n_candidates=len(from_ids))
    out = dict(set=a.set, tol=MATCH_TOL, candidates=len(cdirs), aligned=aligned, groups=len(groups),
               time_dependent_rows=len(tdep), time_dependent_only_groups=time_only,
               matched=sum(len(v) for v in res.values()) - len(time_only), unmatched=unmatched,
               ambiguous=ambiguous, map=res)
    json.dump(out, open(work / f"map_{a.set}.json", "w"), indent=1)
    print(f"{a.set}: {out['matched']}/{out['groups']} user tasks matched, {len(unmatched)} unmatched, "
          f"{ambiguous} ambiguous (several injection tasks match), {len(time_only)} time-dependent only; "
          f"{len(tdep)} time-dependent rows", flush=True)


# ─── subset, deberta_bs ──────────────────────────────────────────────────────

def cmd_subset(a):
    import numpy as np
    rows = load(ROOT / "data" / "transfer" / "sets" / "bipia_pairs.jsonl")
    idx = sorted(np.random.default_rng(SUBSET_SEED).choice(len(rows), SUBSET_N, replace=False).tolist())
    d = Path(a.work) / "bipia500"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "bipia_pairs.jsonl", "w") as f:
        for i in idx:
            f.write(json.dumps(rows[i]) + "\n")
    json.dump(idx, open(d / "indices.json", "w"))


def cmd_deberta_bs(a):
    """Our V2 DeBERTa checkpoints, scored as src/models/deberta.py evaluates (max_len 512, dynamic padding)."""
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    sys.path.insert(0, str(ROOT / "src"))
    from preprocess import build_combined_text
    df = pd.read_csv(a.data)
    texts = df.fillna({"user_intent": ""}).apply(build_combined_text, axis=1).tolist()
    os.makedirs(a.out, exist_ok=True)
    for ck in a.ckpts:
        out = Path(a.out) / f"{Path(ck).name}.json"
        if out.exists():
            continue
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForSequenceClassification.from_pretrained(ck, dtype=torch.float32).to(a.device).eval()
        rec = {"checkpoint": Path(ck).name, "labels": df["label"].tolist()}
        for bs in (1, 16):
            p = []
            with torch.no_grad():
                for i in range(0, len(texts), bs):
                    enc = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True,
                              return_tensors="pt").to(a.device)
                    p += torch.softmax(model(**enc).logits.float(), -1)[:, 1].cpu().tolist()
            rec[f"bs{bs}"] = p
        json.dump(rec, open(out, "w"))
        print(f"deberta {Path(ck).name}: scored {len(texts)} rows at bs 1 and 16", flush=True)


# ─── report ──────────────────────────────────────────────────────────────────

def diff(a, b):
    import numpy as np
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    d = np.abs(a - b)
    return dict(n=len(d), max_abs=float(d.max()), mean_abs=float(d.mean()), n_gt_1e3=int((d > 1e-3).sum()),
                flips_at_05=int(((a >= 0.5) != (b >= 0.5)).sum()))


def probs(path):
    return json.load(open(path))["probs"]


PARTS = ("batch", "pairing", "judge")


def cmd_report(a):
    """One run record per part (each job writes its own): exp gate2_followup_<part>."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval" / "transfer"))
    from gate2 import (JUDGE, check_run, ci_ref, env_versions, liu_metrics, ours, reference_check, released,
                       set_masks, tpr_at)
    from runlog import start_run
    work, g2 = Path(a.work).resolve(), Path(a.gate2).resolve()
    sets_dir = ROOT / "data" / "transfer" / "sets"
    manifest = json.load(open(sets_dir / "manifest.json"))
    run = start_run(version="transfer", exp=f"gate2_followup_{a.part}" + ("_cpu" if a.device == "cpu" else ""),
                    seed=0, args={"part": a.part, "device": a.device, "work": "<scratch>/" + work.name,
                                  "gate2_work": "<scratch>/" + g2.name,
                                  "match_tol": MATCH_TOL, "valid_tol": VALID_TOL, "subset": [SUBSET_N, SUBSET_SEED],
                                  "torchdynamo_disable": os.environ.get("TORCHDYNAMO_DISABLE"),
                                  "upstream_commit": manifest["upstream"]["commit"]},
                    data_version=manifest["data_version"], data_files=[sets_dir / "manifest.json"],
                    allow_dirty=a.allow_dirty)
    res = {}
    bip_keep, tb_y = set_masks()

    def ad_auroc(adm, clean):
        d, neg = json.load(open(adm)), probs(clean)
        y, p = np.array(d["labels"]), np.array(d["probs"])
        pos = p[y == 1]
        return dict(admatched_auroc=float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                                                        np.r_[pos, neg])),
                    admatched_auroc_matchedneg=float(roc_auc_score(y, p)))

    def judge_gap(setname, nf4_adm, nf4_clean, bf16_dir):
        n = ad_auroc(nf4_adm, nf4_clean)
        b = ad_auroc(bf16_dir / f"{JUDGE}__admatched.json", bf16_dir / f"{JUDGE}__adclean.json")
        run.log(f"judge AD-Matched ({setname}) AUROC nf4 {n['admatched_auroc']:.4f} bf16 {b['admatched_auroc']:.4f} "
                f"gap {b['admatched_auroc'] - n['admatched_auroc']:+.4f}")
        return {"nf4": n, "bf16": b, "gap_bf16_minus_nf4": {k: b[k] - n[k] for k in n}}

    {"pairing": report_pairing, "batch": report_batch, "judge": report_judge}[a.part](**locals())
    res["envs"] = {"liu": env_versions(a.liu_py), "ours": env_versions(a.our_py)}
    for sub in SUBDIRS[a.part]:
        if (work / sub).exists():
            shutil.copytree(work / sub, run.dir / "scores" / sub, dirs_exist_ok=True)
    head = {"pairing": "recovery_pass", "batch": "batch_independent_ours_deberta", "judge": "judge_repeat_identical"}
    run.save_metrics(res, headline=(head[a.part], float(res[head[a.part]])))
    run.finish()


SUBDIRS = {"pairing": ("recovered", "judge_nf4_rec", "judge_bf16_rec"), "batch": ("bs", "deberta_bs", "bipia500"),
           "judge": ("judge_repeat", "judge_6gib", "judge_bf16_seeded")}


def report_pairing(work, g2, sets_dir, run, res, bip_keep, tb_y, check_run, ci_ref, reference_check, released,
                   JUDGE, judge_gap, **_):
    # 1. pairing recovery and validation
    rec_dir = work / "recovered"
    res["recovery"] = {s: {k: v for k, v in json.load(open(work / f"map_{s}.json")).items() if k != "map"}
                       for s in FILES}
    val, val_t = {}, {}
    for s in FILES:
        td = set(time_dependent(work, s))
        for det in ENCODERS:
            p, q = probs(rec_dir / "scores" / f"{det}__{s}.json"), liu_file(det, s)["probs"]
            keep = [i for i in range(len(q)) if i not in td]
            val[f"{det}/{s}"] = diff([p[i] for i in keep], [q[i] for i in keep])
            if td:
                val_t[f"{det}/{s}"] = diff([p[i] for i in sorted(td)], [q[i] for i in sorted(td)])
    res["validation_time_dependent_rows_vs_liu"] = val_t
    ok = all(res["recovery"][s]["matched"] + len(res["recovery"][s]["time_dependent_only_groups"])
             == res["recovery"][s]["groups"] for s in FILES)
    ok &= all(v["max_abs"] <= (MATCH_TOL if k.startswith("promptguard2") else VALID_TOL) for k, v in val.items())
    res["validation_rows_vs_liu"] = val
    seeded = {s: load(sets_dir / FILES[s]) for s in FILES}
    recov = {s: load(rec_dir / FILES[s]) for s in FILES}
    res["recovered_vs_seeded"] = {s: dict(rows=len(recov[s]), text_differs=sum(x["text"] != y["text"] for x, y in
                                                                              zip(recov[s], seeded[s])),
                                          labels_equal=[x["label"] for x in recov[s]] == [y["label"] for y in
                                                                                          seeded[s]])
                                  for s in FILES}
    res["recovery_pass"] = bool(ok)
    run.log(f"pairing recovery: {'PASS' if ok else 'FAIL'}")
    for s in FILES:
        r = res["recovery"][s]
        run.log(f"  {s}: {r['matched']}/{r['groups']} user tasks matched (PG2 ≤ {MATCH_TOL}), "
                f"{len(r['unmatched'])} unmatched, {r['ambiguous']} ambiguous; "
                f"{res['recovered_vs_seeded'][s]['text_differs']} rows differ from the seeded build")
    for k, v in val.items():
        run.log(f"  rows {k:28s} max|Δ| {v['max_abs']:.2e} >1e-3 {v['n_gt_1e3']} flips@0.5 {v['flips_at_05']}")
    for k, v in val_t.items():
        run.log(f"  time-dependent rows {k:28s} n {v['n']} max|Δ| {v['max_abs']:.2e} flips@0.5 {v['flips_at_05']}")

    # A19 metrics on the recovered AD-Matched (other sets: Gate 2 run 2, the port in Liu's env)
    bad, table = reference_check(bip_keep, tb_y)
    assert not bad, bad
    cis = ci_ref()
    tmp = work / "a19_recovered"
    tmp.mkdir(exist_ok=True)
    for det in ENCODERS + [JUDGE]:
        for s in ("adclean", "bipia", "tbeval"):
            shutil.copy(g2 / "run2" / f"{det}__{s}.json", tmp)
        src = (work / "judge_nf4_rec" if det == JUDGE else rec_dir / "scores") / f"{det}__admatched.json"
        shutil.copy(src, tmp)
    res["a19_with_recovered_admatched"] = check_run(tmp, table, cis, bip_keep, tb_y)
    run.log(f"A19 metrics, recovered AD-Matched + run 2 other sets: "
            f"{'PASS' if res['a19_with_recovered_admatched']['pass'] else 'FAIL'}")
    for k, c in res["a19_with_recovered_admatched"]["checks"].items():
        if "admatched" in k or not c["pass_"]:
            run.log(f"  {k:45s} ours {c['ours']:.4f} liu {c['liu']:.4f} ({c['rule']}) {'ok' if c['pass_'] else 'FAIL'}")
    res["judge_nf4_recovered_rows_vs_liu"] = diff(probs(work / "judge_nf4_rec" / f"{JUDGE}__admatched.json"),
                                                  released(JUDGE, "admatched")[0])

    res["judge_bf16_vs_nf4_admatched_recovered"] = judge_gap(
        "recovered", work / "judge_nf4_rec" / f"{JUDGE}__admatched.json", g2 / "run2" / f"{JUDGE}__adclean.json",
        work / "judge_bf16_rec")
    for s in FILES:
        shutil.copy(work / f"map_{s}.json", run.dir)


def report_batch(work, run, res, np, **_):
    # 2. batch size and environment, 500-row BIPIA subset
    idx = json.load(open(work / "bipia500" / "indices.json"))
    bs = {}
    for det in ENCODERS:
        p = {f"{e}_bs{b}": probs(work / "bs" / f"{e}_bs{b}" / f"{det}__bipia.json")
             for e in ("liu", "ours") for b in (1, 16)}
        lp = np.array(liu_file(det, "bipia")["probs"])[idx]
        bs[det] = {"liu_bs1_vs_bs16": diff(p["liu_bs1"], p["liu_bs16"]),
                   "ours_bs1_vs_bs16": diff(p["ours_bs1"], p["ours_bs16"]),
                   "bs1_ours_vs_liu_env": diff(p["ours_bs1"], p["liu_bs1"]),
                   "bs16_ours_vs_liu_env": diff(p["ours_bs16"], p["liu_bs16"]),
                   **{f"{k}_vs_released": diff(v, lp) for k, v in p.items()}}
    res["batch_env_bipia500"] = bs
    run.log("batch size / environment, BIPIA 500-row subset (max|Δ|, rows > 1e-3):")
    for det, d in bs.items():
        for k, v in d.items():
            run.log(f"  {det:18s} {k:28s} {v['max_abs']:.2e} {v['n_gt_1e3']:4d} flips {v['flips_at_05']}")
    deb = {}
    for f in sorted((work / "deberta_bs").glob("*.json")):
        r = json.load(open(f))
        deb[r["checkpoint"]] = diff(r["bs1"], r["bs16"])
        run.log(f"  deberta {r['checkpoint']:14s} bs1 vs bs16 max|Δ| {deb[r['checkpoint']]['max_abs']:.2e} "
                f"rows > 1e-3 {deb[r['checkpoint']]['n_gt_1e3']} flips {deb[r['checkpoint']]['flips_at_05']}")
    res["deberta_v2_bs1_vs_bs16"] = deb
    res["batch_independent_ours_deberta"] = bool(deb) and all(v["max_abs"] <= 1e-5 for v in deb.values())
    run.log(f"our DeBERTa batch-independent (max|Δ| ≤ 1e-5): {res['batch_independent_ours_deberta']}")


def report_judge(work, g2, run, res, bip_keep, np, roc_auc_score, released, tpr_at, JUDGE, judge_gap, **_):
    # 3. judge: run-to-run noise, 6 GiB, bf16 vs 4-bit
    def bipia_m(p):
        y, p = np.array(released(JUDGE, "bipia")[1])[bip_keep], np.array(p)[bip_keep]
        return dict(bipia_auroc=float(roc_auc_score(y, p)), bipia_tpr1=tpr_at(y, p))
    base = probs(g2 / "run2" / f"{JUDGE}__bipia.json")
    j = {"run2": bipia_m(base), "liu_released": bipia_m(released(JUDGE, "bipia")[0])}
    for name in ("judge_repeat", "judge_6gib"):
        f = work / name / f"{JUDGE}__bipia.json"
        if f.exists():
            d = json.load(open(f))
            j[name] = dict(**bipia_m(d["probs"]), rows_vs_run2=diff(d["probs"], base),
                           rows_vs_liu=diff(d["probs"], released(JUDGE, "bipia")[0]),
                           placement=d.get("placement"), gpu_mem=d.get("gpu_mem"))
    res["judge_bipia"] = j
    run.log(f"judge BIPIA: run2 {j['run2']}, Liu {j['liu_released']}")
    for name in ("judge_repeat", "judge_6gib"):
        if name in j:
            run.log(f"  {name}: {({k: j[name][k] for k in ('bipia_auroc', 'bipia_tpr1', 'placement')})}; "
                    f"vs run2 max|Δ| {j[name]['rows_vs_run2']['max_abs']:.2e}, "
                    f"vs Liu max|Δ| {j[name]['rows_vs_liu']['max_abs']:.2e}")

    rep = j.get("judge_repeat", {}).get("rows_vs_run2")
    res["judge_repeat_identical"] = bool(rep) and rep["max_abs"] == 0
    res["judge_bf16_vs_nf4_admatched_seeded"] = judge_gap(
        "seeded", g2 / "run2" / f"{JUDGE}__admatched.json", g2 / "run2" / f"{JUDGE}__adclean.json",
        work / "judge_bf16_seeded")


def multi_participant(text):
    """create/get calendar event output with 2+ participants: their order follows set iteration (hash seed)."""
    block = text.split("participants:\n", 1)
    return len(block) == 2 and block[1].split("\n- ")[0].startswith("- ") and block[1].count("@") >= 2


def cmd_residual(a):
    import numpy as np
    sys.path.insert(0, str(ROOT / "src"))
    from runlog import start_run
    rec = Path(a.pairing_run).resolve()
    manifest = ROOT / "data" / "transfer" / "sets" / "manifest.json"
    run = start_run(version="transfer", exp="gate2_followup_residual", seed=0,
                    args={"pairing_run": str(rec.relative_to(ROOT)), "work": "<scratch>/" + Path(a.work).name,
                          "valid_tol": VALID_TOL}, data_version=json.load(open(manifest))["data_version"],
                    data_files=[manifest], allow_dirty=a.allow_dirty)
    res = {}
    for s, f in FILES.items():
        rows = load(rec / "scores" / "recovered" / f)
        td = set(time_dependent(a.work, s))
        keep = [i for i in range(len(rows)) if i not in td]
        multi = [i for i in keep if multi_participant(rows[i]["text"])]
        other = [i for i in keep if i not in set(multi)]
        res[s] = {}
        for det in ENCODERS:
            d = np.abs(np.array(probs(rec / "scores" / "recovered" / "scores" / f"{det}__{s}.json"))
                       - np.array(liu_file(det, s)["probs"]))
            over = [i for i in keep if d[i] > VALID_TOL]
            res[s][det] = dict(rows_over_bound=len(over),
                               over_bound_multi_participant=sum(i in set(multi) for i in over),
                               multi_participant_rows=len(multi), max_abs_multi_participant=float(d[multi].max()),
                               max_abs_other=float(d[other].max()), n_other=len(other))
            run.log(f"{s:9s} {det:18s} rows > {VALID_TOL}: {len(over)} ({res[s][det]['over_bound_multi_participant']} "
                    f"multi-participant); multi-participant rows {len(multi)} max|Δ| "
                    f"{res[s][det]['max_abs_multi_participant']:.2e}; other {len(other)} rows max|Δ| "
                    f"{res[s][det]['max_abs_other']:.2e}")
    res["explained"] = all(r["piguard"]["rows_over_bound"] == r["piguard"]["over_bound_multi_participant"]
                           and r[d]["max_abs_other"] <= VALID_TOL for r in res.values() for d in ENCODERS)
    run.log(f"every row over the bound is a multi-participant row: {res['explained']}")
    run.save_metrics(res, headline=("explained", float(res["explained"])))
    run.finish()


def cmd_envfix(a):
    """Our env (transformers 5) with the two loader differences undone, on the BIPIA subset, vs Liu's env at bs 1:
    PIGuard and PG2 tokenized with the repo's tokenizer.json (transformers 5 rebuilds the DeBERTa-v2 tokenizer from
    spm.model / its own converter); Horizon's sliding-window layers given ModernBERT's local RoPE theta
    (transformers 5 gives them the global theta). Also scores the unmodified loaders in the same process as a control."""
    import torch
    import transformers
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizerFast
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "eval" / "transfer"))
    from detectors import load_registry
    from runlog import start_run
    work = Path(a.work).resolve()
    manifest = ROOT / "data" / "transfer" / "sets" / "manifest.json"
    run = start_run(version="transfer", exp="gate2_followup_envfix", seed=0,
                    args={"work": "<scratch>/" + work.name, "device": a.device, "bs": 16,
                          "transformers": transformers.__version__, "torch": torch.__version__},
                    data_version=json.load(open(manifest))["data_version"], data_files=[manifest],
                    allow_dirty=a.allow_dirty)
    import detectors as D
    reg, rows = load_registry(), load(work / "bipia500" / "bipia_pairs.jsonl")
    texts = [r["text"] for r in rows]
    D.BS = 16
    res = {}
    for det in ENCODERS:
        e = reg[det]
        repo, rev = e["repo"], e["revision"]
        if det == "horizon-base":
            raw = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
            cfg = AutoConfig.from_pretrained(repo, revision=rev)
            # its config.json has no local_rope_theta: 4.57 falls back to ModernBertConfig's default 10000,
            # 5.x gives the sliding layers the global theta instead
            cfg.rope_parameters["sliding_attention"]["rope_theta"] = float(raw.get("local_rope_theta", 10000.0))
            fixed = {"model_kw": {"config": cfg}}
        else:
            auto = AutoTokenizer.from_pretrained(repo, revision=rev)
            fixed = {"tok": PreTrainedTokenizerFast(
                tokenizer_file=hf_hub_download(repo, "tokenizer.json", revision=rev),
                **{k: v for k, v in auto.special_tokens_map.items() if isinstance(v, str)})}
        ref = probs(work / "bs" / "liu_bs1" / f"{det}__bipia.json")
        res[det] = {}
        for name, kw in (("default", {}), ("fixed", fixed)):
            if e["kind"] == "seg":
                p, _ = D.run_segmented(repo, e["max_len"], texts, rev, a.device, **kw)
            else:
                p, _ = D.run_cls(repo, e["max_len"], e["inj_index"], e.get("trust_remote_code", False), texts, rev,
                                 a.device, **kw)
            res[det][name] = diff(p, ref)
            run.log(f"{det:18s} {name:8s} vs Liu env bs1: max|Δ| {res[det][name]['max_abs']:.2e} "
                    f"rows > 1e-3 {res[det][name]['n_gt_1e3']} flips {res[det][name]['flips_at_05']}")
    res["explained"] = all(r["fixed"]["max_abs"] <= 1e-4 for r in res.values())
    run.log(f"shift removed by the two fixes (max|Δ| ≤ 1e-4 for every detector): {res['explained']}")
    run.save_metrics(res, headline=("explained", float(res["explained"])))
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["gen", "match", "subset", "deberta_bs", "report", "residual", "envfix"])
    ap.add_argument("--set", choices=list(FILES))
    ap.add_argument("--out_dir")
    ap.add_argument("--cands", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", help="with --cands: only these candidates")
    ap.add_argument("--map")
    ap.add_argument("--map_from_rows")
    ap.add_argument("--work")
    ap.add_argument("--gate2")
    ap.add_argument("--ckpts", nargs="+")
    ap.add_argument("--data")
    ap.add_argument("--out")
    ap.add_argument("--liu_py")
    ap.add_argument("--pairing_run")
    ap.add_argument("--our_py", default=sys.executable)
    ap.add_argument("--part", choices=PARTS)
    ap.add_argument("--device", default="cuda", help="deberta_bs; report: cpu marks a CPU diagnostic run")
    ap.add_argument("--allow-dirty", dest="allow_dirty", action="store_true")
    a = ap.parse_args()
    {"gen": cmd_gen, "match": cmd_match, "subset": cmd_subset, "deberta_bs": cmd_deberta_bs,
     "report": cmd_report, "residual": cmd_residual, "envfix": cmd_envfix}[a.cmd](a)


if __name__ == "__main__":
    main()
