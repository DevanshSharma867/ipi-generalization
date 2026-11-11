# done_run VERSION EXP SEED: exit 0 if a finished, clean run of results/VERSION/EXP with this seed exists.
done_run() {
  python - "$@" <<'PY'
import json, sys
from pathlib import Path
version, exp, seed = sys.argv[1:]
for d in Path("results", version, exp).glob(f"*_seed{seed}"):
    if (d / "finished.json").exists() and not json.loads((d / "config.json").read_text())["git_dirty"]:
        sys.exit(0)
sys.exit(1)
PY
}
