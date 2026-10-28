"""
make_review_sheet.py
====================
Review sheet for the draft benign pool: data/benign/review_sheet.csv and an offline review page
data/benign/review.html (mark keep/edit/drop, edit inline, export CSV; no server, no network), plus the
hand-review sheet review/benign_pool_review.csv (benign pool + ambiguous slice, one row per item):

  split           train_val | test (benign pool) or eval_only (ambiguous slice)
  proposed_label  benign (human-addressed; label 0 when inserted) or ambiguous (model-directed; eval-only slice)
  reviewer_label  fill in: benign | ambiguous | drop  (an empty cell = not reviewed)
  reviewer_note   free text
  text            edit in place to change an item; apply_review.py records it as an edit

scripts/apply_review.py turns either exported CSV into data/benign/benign_v3.json.

    python scripts/make_review_sheet.py [--draft data/benign/benign_v3_draft.json]
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benign"
FIELDS = ["id", "register", "split_pool", "text", "form_tags", "keep_edit_drop", "edited_text"]
POOL_REVIEW = ROOT / "review" / "benign_pool_review.csv"
POOL_FIELDS = ["id", "split", "register", "text", "proposed_label", "reviewer_label", "reviewer_note"]


def pool_review_row(it: dict) -> list:
    amb = it["split_pool"] == "ambiguous"
    return [it["id"], "eval_only" if amb else it["split_pool"], it["register"], it["text"],
            "ambiguous" if amb else "benign", "", ""]

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Benign Pool Review</title>
<style>
:root { --bg:#fbfaf7; --fg:#1f1e1c; --muted:#6b6862; --line:#e2dfd8; --card:#ffffff; --accent:#2a63c4;
        --keep:#1d8a4e; --edit:#b7791f; --drop:#c0392b; }
@media (prefers-color-scheme: dark) { :root { --bg:#191817; --fg:#ecebe8; --muted:#a19e97; --line:#34322f;
        --card:#22211f; --accent:#7aa7f0; --keep:#4cc38a; --edit:#e0a64a; --drop:#ec6a5c; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, -apple-system, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--bg); border-bottom:1px solid var(--line);
         padding:12px 16px; display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; }
h1 { font-size:16px; margin:0 12px 0 0; }
select, button, textarea { font:inherit; color:inherit; }
select, button { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:4px 8px; }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
.stats { color:var(--muted); font-variant-numeric: tabular-nums; }
main { padding:12px 16px 48px; max-width:1100px; margin:0 auto; }
.item { background:var(--card); border:1px solid var(--line); border-left:4px solid var(--line); border-radius:8px;
        padding:10px 12px; margin:10px 0; }
.item[data-d=keep] { border-left-color:var(--keep); } .item[data-d=edit] { border-left-color:var(--edit); }
.item[data-d=drop] { border-left-color:var(--drop); opacity:.65; }
.meta { color:var(--muted); font-size:12px; display:flex; flex-wrap:wrap; gap:4px 12px; margin-bottom:6px; }
pre { white-space:pre-wrap; word-break:break-word; margin:6px 0; font:13px/1.4 ui-monospace, Menlo, monospace; }
.choices { display:flex; gap:12px; margin:6px 0; flex-wrap:wrap; }
.choices label { cursor:pointer; }
textarea { width:100%; min-height:3.2em; background:var(--bg); border:1px solid var(--line); border-radius:6px;
           padding:6px; font:13px/1.4 ui-monospace, Menlo, monospace; display:none; }
.item[data-d=edit] textarea { display:block; }
.tags { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Benign pool review</h1>
  <label>Register <select id="fr"><option value="">all</option><option>email</option><option>table</option><option>code</option></select></label>
  <label>Pool <select id="fp"><option value="">all</option><option>train_val</option><option>test</option><option>ambiguous</option></select></label>
  <label>Show <select id="fd"><option value="">all</option><option value="none">unreviewed</option><option>keep</option><option>edit</option><option>drop</option></select></label>
  <span class="stats" id="stats"></span>
  <button id="allkeep">Keep all unreviewed (shown)</button>
  <button class="primary" id="export">Export CSV</button>
</header>
<main id="list"></main>
<script>
const ITEMS = __ITEMS__;
const KEY = "benign_review_v3:" + __DRAFT_SHA__;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} };
const esc = s => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const fmtTags = t => Object.entries(t).map(([k, v]) => k + "=" + (typeof v === "number" && !Number.isInteger(v) ? v.toFixed(3) : v)).join("  ");

function render() {
  const fr = document.getElementById("fr").value, fp = document.getElementById("fp").value,
        fd = document.getElementById("fd").value;
  const list = document.getElementById("list");
  list.innerHTML = "";
  for (const it of ITEMS) {
    const s = state[it.id] || {d: "", e: ""};
    if ((fr && it.register !== fr) || (fp && it.split_pool !== fp) || (fd && (fd === "none" ? s.d : s.d !== fd))) continue;
    const div = document.createElement("div");
    div.className = "item"; div.dataset.d = s.d || "";
    div.innerHTML = `<div class="meta"><b>${it.id}</b><span>${it.register}</span><span>${it.split_pool}</span></div>
      <pre>${esc(it.text)}</pre>
      <div class="choices">${["keep", "edit", "drop"].map(d =>
        `<label><input type="radio" name="${it.id}" value="${d}" ${s.d === d ? "checked" : ""}> ${d}</label>`).join("")}</div>
      <textarea placeholder="edited text">${esc(s.e || it.text)}</textarea>
      <div class="tags">${esc(fmtTags(it.form_tags))}</div>`;
    div.querySelectorAll("input").forEach(r => r.addEventListener("change", () => {
      state[it.id] = {d: r.value, e: (state[it.id] || {}).e || ""};
      if (r.value === "edit" && !state[it.id].e) state[it.id].e = it.text;
      div.dataset.d = r.value; save(); stats();
    }));
    div.querySelector("textarea").addEventListener("input", ev => {
      state[it.id] = {d: "edit", e: ev.target.value}; save(); stats();
    });
    list.appendChild(div);
  }
  stats();
}
function stats() {
  const c = {keep: 0, edit: 0, drop: 0};
  for (const it of ITEMS) { const d = (state[it.id] || {}).d; if (d) c[d]++; }
  const n = ITEMS.length, done = c.keep + c.edit + c.drop;
  document.getElementById("stats").textContent =
    `${done}/${n} reviewed · keep ${c.keep} · edit ${c.edit} · drop ${c.drop}`;
}
const q = v => '"' + String(v).replace(/"/g, '""') + '"';
document.getElementById("export").addEventListener("click", () => {
  const head = ["id", "register", "split_pool", "text", "form_tags", "keep_edit_drop", "edited_text"];
  const rows = [head.join(",")];
  for (const it of ITEMS) {
    const s = state[it.id] || {d: "", e: ""};
    rows.push([it.id, it.register, it.split_pool, it.text, JSON.stringify(it.form_tags), s.d || "",
               s.d === "edit" ? s.e : ""].map(q).join(","));
  }
  const blob = new Blob([rows.join("\r\n") + "\r\n"], {type: "text/csv;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "review_sheet_reviewed.csv"; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
document.getElementById("allkeep").addEventListener("click", () => {
  document.querySelectorAll(".item").forEach(div => {
    const id = div.querySelector("b").textContent;
    if (!(state[id] || {}).d) { state[id] = {d: "keep", e: ""}; }
  });
  save(); render();
});
["fr", "fp", "fd"].forEach(id => document.getElementById(id).addEventListener("change", render));
render();
</script>
</body>
</html>
"""


def main():
    import hashlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", type=Path, default=OUT / "benign_v3_draft.json")
    a = ap.parse_args()
    items = json.load(open(a.draft))["items"]
    with open(OUT / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for it in items:
            w.writerow([it["id"], it["register"], it["split_pool"], it["text"],
                        json.dumps(it["form_tags"], sort_keys=True), "", ""])
    POOL_REVIEW.parent.mkdir(exist_ok=True)
    with open(POOL_REVIEW, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(POOL_FIELDS)
        w.writerows(pool_review_row(it) for it in items)
    sha = hashlib.sha256(a.draft.read_bytes()).hexdigest()[:16]
    page = PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/")) \
               .replace("__DRAFT_SHA__", json.dumps(sha))
    (OUT / "review.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'review_sheet.csv'}, {OUT / 'review.html'} and {POOL_REVIEW} ({len(items)} items)")


if __name__ == "__main__":
    main()
