"""
make_annotation_sheet.py
========================
Offline annotation page for intent pairs (docs/pairs_annotation_guide.md): one self-contained HTML file per
annotator. It shows one pair at a time (the context with the inserted instruction highlighted, and the two
requests as "Request A" / "Request B"), records label (aligned | subverting | unclear), confidence (1-3) and a
note for each request, autosaves in the browser, and exports a CSV. No server, no network.

The page never contains the intended label, the generator or the overlap class. Pair order and the A/B order
are drawn per annotator from a seeded hash, so the two annotators see different orders. The exported CSV
identifies a request by pair_id + member (A/B) + its text; scripts/agreement.py maps it back to its role.

    python scripts/make_annotation_sheet.py data/pairs/pilot_v1.jsonl --annotator ann1
    # -> annotation/pilot_v1/ann1.html; the annotator exports annotation/pilot_v1/ann1.csv
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV_FIELDS = ["annotator", "pair_id", "member", "request", "label", "confidence", "note", "saved_at"]


def ab_order(seed: str, annotator: str, pair_id: str) -> bool:
    """True if Request A is the aligned request for this annotator and pair."""
    return hashlib.sha256(f"{seed}:{annotator}:{pair_id}".encode()).digest()[0] % 2 == 0


def sheet_items(pairs: list, annotator: str, seed: str) -> list:
    items = []
    for p in pairs:
        reqs = [p["request_aligned"], p["request_subverting"]]
        if not ab_order(seed, annotator, p["pair_id"]):
            reqs.reverse()
        off = p["char_offset"]
        items.append({"pair_id": p["pair_id"], "task": p["task"], "before": p["context"][:off],
                      "instruction": p["instruction"], "after": p["context"][off + len(p["instruction"]):],
                      "A": reqs[0], "B": reqs[1]})
    random.Random(f"{seed}:{annotator}:order").shuffle(items)
    return items


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intent Pair Annotation</title>
<style>
:root { --bg:#fbfaf7; --fg:#1f1e1c; --muted:#6b6862; --line:#e2dfd8; --card:#ffffff; --accent:#2a63c4;
        --mark:#fff1b8; --ok:#1d8a4e; }
@media (prefers-color-scheme: dark) { :root { --bg:#191817; --fg:#ecebe8; --muted:#a19e97; --line:#34322f;
        --card:#22211f; --accent:#7aa7f0; --mark:#5a4a12; --ok:#4cc38a; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, -apple-system, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; }
h1 { font-size:16px; margin:0 8px 0 0; }
button, textarea, select { font:inherit; color:inherit; }
button, select { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 10px; cursor:pointer; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.stats { color:var(--muted); font-variant-numeric:tabular-nums; }
main { padding:12px 16px 48px; max-width:980px; margin:0 auto; }
.q { color:var(--muted); margin:4px 0 12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin:12px 0; }
.meta { color:var(--muted); font-size:12px; margin-bottom:6px; }
pre { white-space:pre-wrap; word-break:break-word; margin:0; font:13px/1.45 ui-monospace, Menlo, monospace;
      max-height:50vh; overflow:auto; }
mark { background:var(--mark); color:inherit; outline:1px solid var(--line); }
.req { font-weight:600; margin:2px 0 8px; }
.choices { display:flex; flex-wrap:wrap; gap:6px 16px; margin:6px 0; }
.choices label { cursor:pointer; }
textarea { width:100%; min-height:2.4em; background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:6px; }
.done { color:var(--ok); }
</style>
</head>
<body>
<header>
  <h1>Intent pairs · <span id="who"></span></h1>
  <button id="prev">← Prev</button><span class="stats" id="pos"></span><button id="next">Next →</button>
  <button id="todo">Next unlabelled</button>
  <span class="stats" id="stats"></span>
  <button class="primary" id="export">Export CSV</button>
</header>
<main>
  <p class="q">For each request: if the assistant did what the <mark>highlighted instruction</mark> says while answering
  that request, would the user get what they asked for (<b>aligned</b>) or something else (<b>subverting</b>)? Use
  <b>unclear</b> when you cannot tell. See docs/pairs_annotation_guide.md.</p>
  <div class="card"><div class="meta" id="meta"></div><pre id="ctx"></pre></div>
  <div id="members"></div>
</main>
<script>
const ITEMS = __ITEMS__, ANNOTATOR = __ANNOTATOR__, KEY = "intent_pairs:" + __SHEET__ + ":" + ANNOTATOR;
let state = {}, i = 0;
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const done = it => ["A", "B"].every(m => (state[it.pair_id + "|" + m] || {}).label);
document.getElementById("who").textContent = ANNOTATOR;

function render() {
  const it = ITEMS[i];
  document.getElementById("pos").textContent = `${i + 1} / ${ITEMS.length}`;
  document.getElementById("meta").innerHTML = `${esc(it.pair_id)} · ${esc(it.task)}` + (done(it) ? ' · <span class="done">labelled</span>' : "");
  document.getElementById("ctx").innerHTML = esc(it.before) + "<mark>" + esc(it.instruction) + "</mark>" + esc(it.after);
  const box = document.getElementById("members");
  box.innerHTML = "";
  for (const m of ["A", "B"]) {
    const k = it.pair_id + "|" + m, s = state[k] || {};
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `<div class="meta">Request ${m}</div><div class="req">${esc(it[m])}</div>
      <div class="choices">${["aligned", "subverting", "unclear"].map(l =>
        `<label><input type="radio" name="l${m}" value="${l}" ${s.label === l ? "checked" : ""}> ${l}</label>`).join("")}</div>
      <div class="choices">confidence ${[1, 2, 3].map(c =>
        `<label><input type="radio" name="c${m}" value="${c}" ${String(s.confidence) === String(c) ? "checked" : ""}> ${["low", "medium", "high"][c - 1]}</label>`).join("")}</div>
      <textarea placeholder="note (optional)">${esc(s.note || "")}</textarea>`;
    const upd = patch => { state[k] = {...(state[k] || {}), ...patch, saved_at: new Date().toISOString()}; save(); stats();
                           document.getElementById("meta").innerHTML = `${esc(it.pair_id)} · ${esc(it.task)}` + (done(it) ? ' · <span class="done">labelled</span>' : ""); };
    div.querySelectorAll(`input[name=l${m}]`).forEach(r => r.addEventListener("change", () => upd({label: r.value})));
    div.querySelectorAll(`input[name=c${m}]`).forEach(r => r.addEventListener("change", () => upd({confidence: r.value})));
    div.querySelector("textarea").addEventListener("input", ev => upd({note: ev.target.value}));
    box.appendChild(div);
  }
  stats();
}
function stats() {
  const n = ITEMS.filter(done).length;
  document.getElementById("stats").textContent = `${n}/${ITEMS.length} pairs labelled`;
}
const go = j => { i = Math.max(0, Math.min(ITEMS.length - 1, j)); render(); window.scrollTo(0, 0); };
document.getElementById("prev").addEventListener("click", () => go(i - 1));
document.getElementById("next").addEventListener("click", () => go(i + 1));
document.getElementById("todo").addEventListener("click", () => {
  for (let d = 1; d <= ITEMS.length; d++) { const j = (i + d) % ITEMS.length; if (!done(ITEMS[j])) return go(j); }
});
const q = v => '"' + String(v ?? "").replace(/"/g, '""') + '"';
document.getElementById("export").addEventListener("click", () => {
  const rows = [__FIELDS__.join(",")];
  for (const it of ITEMS) for (const m of ["A", "B"]) {
    const s = state[it.pair_id + "|" + m] || {};
    rows.push([ANNOTATOR, it.pair_id, m, it[m], s.label, s.confidence, s.note, s.saved_at].map(q).join(","));
  }
  const blob = new Blob([rows.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = ANNOTATOR + ".csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
render();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pairs", type=Path, help="pairs JSONL (src/pairs.PAIR_FIELDS)")
    ap.add_argument("--annotator", required=True, help="short id, e.g. ann1 (used in the CSV and the page order)")
    ap.add_argument("--seed", default="sheet_v1", help="order seed; keep it fixed for a set")
    ap.add_argument("--out", type=Path, default=None, help="default: annotation/<pairs stem>/<annotator>.html")
    a = ap.parse_args()
    pairs = [json.loads(line) for line in open(a.pairs, encoding="utf-8")]
    out = a.out or ROOT / "annotation" / a.pairs.stem / f"{a.annotator}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet = hashlib.sha256(a.pairs.read_bytes()).hexdigest()[:16]
    dump = lambda x: json.dumps(x, ensure_ascii=False).replace("</", "<\\/")
    page = (PAGE.replace("__ITEMS__", dump(sheet_items(pairs, a.annotator, a.seed)))
                .replace("__ANNOTATOR__", dump(a.annotator)).replace("__SHEET__", dump(sheet))
                .replace("__FIELDS__", dump(CSV_FIELDS)))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(pairs)} pairs; export saves {a.annotator}.csv)")


if __name__ == "__main__":
    main()
