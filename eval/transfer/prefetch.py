"""
prefetch.py
===========
Download every model in configs/detectors.yaml into HF_HOME at its pinned revision, and report access to gated
ones. Network I/O only, so it runs on the cluster login node (scripts/prefetch_models.sh sets HF_HOME and
HF_TOKEN_PATH); the token is read by huggingface_hub from HF_TOKEN_PATH and never printed or logged here.

    python eval/transfer/prefetch.py pin                   # laptop: pin revision: null entries to the head of main
    python eval/transfer/prefetch.py fetch [--only NAME..]  # download (resumable; rerun after an interruption)
    python eval/transfer/prefetch.py status                # access + local completeness of every model, logged via
                                                           # runlog to results/transfer/prefetch/<run_id>/

Access per model: granted, pending (request awaiting the repo authors), rejected, not_requested, or not_found.
Pending vs not requested is read from the Hub's 403 message, which is the only place the Hub states it.
Only the files needed to load the model are downloaded: no ONNX/TF/Flax/GGUF exports, no `original/` checkpoints,
and no .bin/.pt weights when safetensors exist.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, filter_repo_objects

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "detectors.yaml"
EXPORTS = ["*.onnx", "onnx/*", "*.h5", "*.msgpack", "*.ot", "tf_model*", "flax_model*", "*.tflite", "openvino/*",
           "original/*", "*.gguf", "coreml/*"]


def load_registry():
    """Models with their own weights (alias_of entries reuse another entry's download)."""
    return [m for m in yaml.safe_load(REGISTRY.read_text())["models"] if "alias_of" not in m]


def ignore_patterns(files):
    if any(f.endswith(".safetensors") for f in files):
        return EXPORTS + ["*.bin", "*.pt", "*.pth"]
    return EXPORTS


def access(api, repo):
    try:
        api.auth_check(repo)
        return "granted", ""
    except GatedRepoError as e:
        msg = str(getattr(e, "server_message", "") or e)
        if re.search(r"awaiting a review|pending", msg, re.I):
            return "pending", ""
        if re.search(r"rejected", msg, re.I):
            return "rejected", ""
        return "not_requested", ""
    except RepositoryNotFoundError:
        return "not_found", ""
    except Exception as e:  # network or auth failure: reported, never treated as granted
        return "error", f"{type(e).__name__}: {e}"[:300]


def expected_files(api, m):
    files = [s.rfilename for s in api.model_info(m["repo"], revision=m["revision"]).siblings]
    return sorted(filter_repo_objects(files, ignore_patterns=ignore_patterns(files))), ignore_patterns(files)


def local_state(api, m):
    """(complete, missing files) for the pinned revision in the local HF cache."""
    want, ign = expected_files(api, m)
    try:
        snap = Path(snapshot_download(m["repo"], revision=m["revision"], ignore_patterns=ign, local_files_only=True))
    except Exception:
        return False, want
    missing = [f for f in want if not (snap / f).exists()]
    return not missing, missing


def cmd_pin(api, _):
    text, lines = REGISTRY.read_text(), []
    for line in text.splitlines():
        m = re.search(r"repo: ([^,}]+),.*revision: null", line)
        if m:
            info = api.model_info(m.group(1).strip())
            line = line.replace("revision: null", f"revision: {info.sha}, last_modified: {info.last_modified.date()}")
            print(f"pinned {m.group(1).strip():55s} {info.sha} ({info.last_modified.date()})")
        lines.append(line)
    REGISTRY.write_text("\n".join(lines) + "\n")


def cmd_fetch(api, a):
    models = [m for m in load_registry() if not a.only or m["name"] in a.only]
    for m in models:
        state, err = access(api, m["repo"])
        if state != "granted":
            print(f"{m['name']:20s} {state:14s} skipped {err}", flush=True)
            continue
        _, ign = expected_files(api, m)
        snapshot_download(m["repo"], revision=m["revision"], ignore_patterns=ign)
        ok, missing = local_state(api, m)
        print(f"{m['name']:20s} granted        {'complete' if ok else f'INCOMPLETE, missing {missing[:3]}'}", flush=True)


def cmd_status(api, a):
    sys.path.insert(0, str(ROOT / "src"))
    from runlog import start_run
    try:
        api.whoami()
        token = "valid"
    except Exception:
        token = "missing_or_invalid"
    run = start_run(version="transfer", exp="prefetch", seed=0, args={"registry": "configs/detectors.yaml"},
                    data_version="detectors-registry", data_files=[REGISTRY], allow_dirty=a.allow_dirty)
    run.log(f"hf token: {token}")
    out = {}
    for m in load_registry():
        state, err = access(api, m["repo"])
        info = api.model_info(m["repo"], revision=m["revision"]) if state != "not_found" else None
        ok, missing = local_state(api, m) if state == "granted" else (False, [])
        out[m["name"]] = {"repo": m["repo"], "revision": m["revision"], "role": m["role"],
                          "gated": info.gated if info else None, "access": state, "complete": ok,
                          "missing": missing[:10], "error": err}
        run.log(f"{m['name']:20s} {m['repo']:55s} gated={str(out[m['name']]['gated']):6s} {state:14s} "
                f"{'complete' if ok else ('INCOMPLETE' if state == 'granted' else '-')}")
    blocked = sorted({v["repo"] for v in out.values() if v["access"] != "granted"})
    run.log("still blocked: " + (", ".join(blocked) or "none"))
    ready = sum(v["complete"] for v in out.values())
    run.save_metrics({"hf_token": token, "models": out, "blocked": blocked}, headline=("models_ready", float(ready)))
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pin")
    f = sub.add_parser("fetch")
    f.add_argument("--only", nargs="+", default=[])
    s = sub.add_parser("status")
    s.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    {"pin": cmd_pin, "fetch": cmd_fetch, "status": cmd_status}[a.cmd](HfApi(), a)


if __name__ == "__main__":
    main()
