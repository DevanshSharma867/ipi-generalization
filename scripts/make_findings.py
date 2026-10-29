"""
make_findings.py
================
Render docs/findings.md from docs/findings.src.md, filling every number from paper/numbers.tex (entries with
full provenance) or paper/legacy_numbers.tex (every other entry; never \\input by the paper).

Each entry is a "## F<nnn>. <title>" section that opens with a fenced ```finding block (YAML) holding its
provenance; scripts/make_tables.py validates these blocks and decides which file each macro goes to.
In the prose, {{Name}} becomes "`\\Name` = <value>" and {{Name|ci}} becomes "`\\Name` = <value> [<lo>, <hi>]"
with the CI taken from \\NameCiLo / \\NameCiHi ("n/a (...)" values have no CI). An entry may only use macros it
owns (its `macros:` globs, minus any "!glob" exclusions), and a missing macro is an error, so no number can be typed
by hand.

    python scripts/make_findings.py      # run after scripts/make_tables.py
"""

import fnmatch
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "docs" / "findings.src.md", ROOT / "docs" / "findings.md"
NUMBERS, LEGACY = ROOT / "paper" / "numbers.tex", ROOT / "paper" / "legacy_numbers.tex"

# full: produced by a clean committed run. recomputed-match: recomputed by committed code from saved frozen scores
# and equal to the frozen value. Only these two reach paper/numbers.tex.
STATUSES = ("full", "recomputed-match", "linked-dirty", "legacy-unverified", "contaminated", "pending")
NUMBERS_OK = ("full", "recomputed-match")
REQUIRED = ("id", "status", "script", "commit", "seeds", "data_version", "eval_set", "threshold_protocol", "date",
            "source", "macros")
BLOCK = re.compile(r"```finding\n(.*?)\n```\n", re.S)
HEAD = re.compile(r"^## (F\d{3})\. ", re.M)
MACRO = re.compile(r"\{\{(\w+)(\|ci)?\}\}")


def parse_entries(text=None):
    """[(entry dict, prose)] in file order, plus the preamble before the first entry."""
    text = SRC.read_text() if text is None else text
    heads = list(HEAD.finditer(text))
    preamble = text[:heads[0].start()] if heads else text
    entries = []
    for i, h in enumerate(heads):
        section = text[h.start():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        m = BLOCK.search(section)
        if not m:
            raise ValueError(f"{h.group(1)}: no ```finding block")
        e = yaml.safe_load(m.group(1))
        if e.get("id") != h.group(1):
            raise ValueError(f"{h.group(1)}: block id {e.get('id')!r} does not match the heading")
        entries.append((e, section, m))
    return preamble, entries


def owns(entry, name):
    """name matches one of the entry's `macros:` globs and none of its "!glob" exclusions."""
    keep = [g for g in entry["macros"] if not g.startswith("!")]
    drop = [g[1:] for g in entry["macros"] if g.startswith("!")]
    return any(fnmatch.fnmatchcase(name, g) for g in keep) and not any(fnmatch.fnmatchcase(name, g) for g in drop)


def load_macros(path):
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", path.read_text())) if path.exists() else {}


def provenance_line(e):
    seeds = e["seeds"] if isinstance(e["seeds"], str) else ", ".join(map(str, e["seeds"]))
    src = e["source"] if isinstance(e["source"], list) else [e["source"]]
    proto = e["threshold_protocol"] if isinstance(e["threshold_protocol"], str) else "; ".join(e["threshold_protocol"])
    lines = [f"> **{e['id']}** · status **`{e['status']}`** · {e['date']} · data `{e['data_version']}` · "
             f"eval `{e['eval_set']}` · threshold: {proto} · seeds: {seeds}  ",
             f"> script `{e['script']}` @ `{e['commit']}` · source: " + ", ".join(f"`{s}`" for s in src)]
    if e.get("cluster"):
        lines[-1] += f" · bootstrap clusters: {e['cluster']}"
    if e["status"] not in NUMBERS_OK:
        lines.append(f"> **Not a headline number ({e['status']}):** {e.get('reason', '')}")
    if e.get("preliminary"):
        lines.append(f"> **PRELIMINARY:** {e['preliminary']}")
    return "\n".join(lines) + "\n"


def main():
    macros = {**load_macros(LEGACY), **load_macros(NUMBERS)}
    preamble, entries = parse_entries()
    missing, foreign = [], []

    def render(text, entry):
        def fill(m):
            name, want_ci = m.group(1), m.group(2)
            if entry is None or not owns(entry, name):
                foreign.append(f"{entry['id'] if entry else 'preamble'} uses {name}")
                return m.group(0)
            if name not in macros:
                missing.append(name)
                return m.group(0)
            value = macros[name].replace(r"\_", "_")
            if want_ci and not value.startswith("n/a"):
                lo, hi = macros.get(name + "CiLo"), macros.get(name + "CiHi")
                if lo is None or hi is None:
                    missing.append(name + "CiLo/CiHi")
                    return m.group(0)
                return f"`\\{name}` = {value} [{lo}, {hi}]"
            return f"`\\{name}` = {value}"
        return MACRO.sub(fill, text)

    counts = {s: 0 for s in STATUSES}
    out = [render(preamble, None), "@@SUMMARY@@"]
    for e, section, block in entries:
        counts[e["status"]] += 1
        out.append(render(section[:block.start()] + provenance_line(e) + section[block.end():], e))
    if missing or foreign:
        sys.exit("\n".join(["missing macros: " + ", ".join(sorted(set(missing)))] * bool(missing)
                           + ["macros used outside their owning entry: " + "; ".join(foreign)] * bool(foreign)))
    summary = "Entries by status: " + ", ".join(f"{s} {n}" for s, n in counts.items()) + ".\n\n"
    OUT.write_text("<!-- Generated by scripts/make_findings.py from docs/findings.src.md, paper/numbers.tex and "
                   "paper/legacy_numbers.tex; edit the .src.md -->\n" + "".join(out).replace("@@SUMMARY@@", summary, 1))
    print(f"wrote {OUT.relative_to(ROOT)} ({summary.strip()})")


if __name__ == "__main__":
    main()
