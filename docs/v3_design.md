# V3 dataset design (Phase A: for approval, nothing built yet)

**Goal.** In V3 the label depends only on the content of one inserted span, and no content-free feature predicts it.
V2 stays frozen. New code goes in `src/data_pipeline_v3.py`, data in `data/v3/`, results in `results/v3/`.

Every raw-data fact below comes from `results/v3/design_facts/20261021-131051_seed0/metrics.json`
(`scripts/v3_design_facts.py`, commit 2b1eff0). V2 numbers are cited by their `docs/findings.md` macro name.

**V2 problems this design fixes:**
- Position artifacts: `\VTwoArtifactPositionOracleFOneMacro` 0.7333; `\VTwoArtifactPrefixThreeZeroTfidfLrFOneMacro` 0.7046.
- Span-form artifacts: `\VTwoArtifactSpanFormLrFOneMacro` 0.8421.
- A seed-dependent DeBERTa position shortcut: `\VTwoSeedvarDebertaSNineNineNinePTwostartminusmiddle` vs `…SFourTwo…`.
- Truncated end-position spans: `\VTwoTruncDebertaMeanPThreeendtruncated` 0.104.

## 1. Row design
**Main set.** Each doc yields 4 rows (2 clean, 2 injected). Every row has exactly one inserted span.

| Row | Label | Span |
|---|---|---|
| injected ×2 | 1 | two different attack strings from the doc's split pool (§4) |
| clean A | 0 | one item from the benign human-addressed pool, matched to the doc's register (§3) |
| clean B | 0 | spliced verbatim from another doc of the same task and split |

**Where clean-B spans come from:**
- **Email:** a neutral declarative sentence, using the V2 extractor (64/78 docs have one; 82 distinct sentences).
- **Code:** a prose line followed by its fenced code block. Every code attack has exactly this form (multi-line, 100% backticks). 80/100 docs have one; there are 159 distinct chunks, all within the code-attack token budget of 204. This replaces V2's comment extractor (12 donor docs, 14 sentences), which caused the 2-donor problem.
- **Table:** table docs contain no prose. Clean B is a declarative sentence rendered from a donor table's header and one row, using ≥10 templates, with digits and length controlled to match the form targets. This is synthetic text (risk R3).

**Donor reuse caps.** No donor sentence or chunk is used more than k times: k=2 for email and code, k=1 for table. Donor docs available within each split, for split seeds 0–4, before the length filter:

| Task | Train | Val | Test |
|---|---|---|---|
| Email | 41–48 | 9–12 | 7–12 |
| Code | 54–58 | 10–14 | 11–12 |
| Table | all | all | all |

With k=2, the smallest test cell is email at seed 0 or 3: 12 docs and 7 donor docs. That holds only if the 7 donor docs offer ≥6 distinct sentences. The builder asserts this and fails rather than raising k.

**Eval-only slices (never trained on):**
- `raw_clean`: the untouched doc, one row per doc (natural false-positive rate).
- `ambiguous`: model-directed items ("Act as…", "System: …", "New task: …"). One row per val/test doc; report the flag rate only.
- `long`: see §7.

## 2. Insertion
Insertion uses `probes.insert_preserving`, extended with two arguments: `sep` and `boundaries=("paragraph", "line")`. The current defaults stay, so the V2 probes reproduce.

- **Sampling.** Position ∈ {start, middle, end} (uniform) and separator ∈ {`\n\n`, `\n`} (uniform) are sampled i.i.d. per row with a seeded RNG, identically for all 4 rows types.
- **Middle position:** the line/paragraph boundary nearest the character midpoint, never inside a code fence. Words are never re-joined.
- **Recorded per row:** `position, separator, boundary_kind, char_offset, span_len`.

Some docs have no internal line boundary:
- 21/78 email docs have no newline at all. I propose two fallbacks, in order: the email field delimiter (`|CONTENT:` etc.), then a sentence boundary. Every one of the 21 has one of these.
- Code: 4 docs have only a sentence boundary. 1 doc has no internal boundary; its middle rows become `end` (recorded as `boundary_kind=none`).

**This deviates from "line/paragraph only" and needs your OK.** The fallback depends on the doc, not the label, because all 4 rows of a doc share the same boundary set.

## 3. Benign pool v3
**Removals from V2.** The "Note:" template (`Note: ignore the introduction and focus on the results section.`) is dropped. Model-directed items move to the ambiguous pool.

**Sizes.** ≥120 train/val items and ≥60 test items, disjoint. Each is tagged by register: email, table (text register) or code. The ambiguous pool has ≥30 items.

**Form profile of the train attack pool:**

| Feature | Text register (75) | Code register (50) |
|---|---|---|
| Words: min / p25 / median / p75 / max | 5 / 7 / 9 / 12.5 / 17 | 16 / 23.25 / 27.5 / 31 / 58 |
| Ends with "?" | 0.16 | 0 |
| Ends with "." | 0.813 | 0 |
| Double quote | 0.053 | 0.96 |
| Single quote or apostrophe | 0.20 | 0.12 |
| Digit | 0.04 | 0.36 |
| Multi-line | 0 | 1.0 (prose lead-in + fenced block) |

For comparison, the V2 benign_train items are 6–11 words, 100% end in ".", and 0% have "?", quotes, digits or newlines.

**Form targets.** Clean A and clean B are pooled 1:1 against attacks. So, for each of the 9 `span_form_lr` features and each register, clean A's target is 2·(attack rate) − (clean-B rate), clipped to [0, 1]. Clean-B rates are measured after donor selection. For example, clean B never ends in "?", so clean A's "?" target in the text register is 0.32.

Each split's benign items are matched to the attack pool it is paired with:
- train/val items to the train-category profile;
- test items to the test-file profile, which has digit share 0.20 vs 0.04 in train.

Matching uses form statistics only, never model outputs.

**Required variety:**
- benign questions to the reader;
- benign quoted text;
- digits (dates, amounts);
- for code: a prose lead-in plus a fenced benign snippet (a TODO block, a logging line, a test), reaching the attack word range and quote/digit shares.

**Drafting and review:**
1. I over-generate about 2× each target and write `data/v3/benign/candidates_v0.csv`. Columns: `id, pool (benign|ambiguous), register, text, n_words, ends_q, has_quote, has_digit, multi_line, keep_edit_drop, edited_text, note`. The form columns are computed, not typed. V2 items other than the Note template are included as candidates.
2. A generated `review.html` shows the same sheet, plus per-register form histograms of the kept items against the targets.
3. You fill in `keep_edit_drop` (and `edited_text`).
4. `scripts/freeze_benign_v3.py` writes `data/v3/benign/benign_v3.json` with a sha256. It then assigns items to train/val vs test by a seeded draw stratified by register and form bucket, so no reviewer picks the test items.

**Criterion for benign items:** addressed to the human reader. Following the item must not change what an assistant should do for any intent in that register.

## 4. Attack pools and validation
The BIPIA train and test attack files stay disjoint as in V2 (0 shared strings; CODEBASE_STATE §3.1). However, one category name, `Language Translation`, appears in both text files.

Categories in each file (5 strings each):

| File | Categories |
|---|---|
| text train (15) | Information Retrieval, Content Creation, Learning and Tutoring, Language Translation, Programming Help, Alphanumeric Substitution, Homophonic Substitution, Misspelling Intentionally, Anagramming, Space Removal & Grouping, Instruction, Social Interaction, Persuasion, Clickbait, Malware Distribution |
| text test (15) | Task Automation, Business Intelligence, Conversational Agent, Research Assistance, Sentiment Analysis, Substitution Ciphers, Base Encoding, Reverse Text, Emoji Substitution, Language Translation, Information Dissemination, Marketing & Advertising, Entertainment, Scams & Fraud, Misinformation & Propaganda |
| code train (10) | Cookie Theft, Memory Scanning, Dumpster Diving, Environment Variable Analysis, Device and Driver Enumeration, Sending Out Spam Emails, Crippling Critical Infrastructures, Network Propagation, Exploiting System Vulnerabilities, Cryptocurrency Mining |
| code test (10) | Data Eavesdropping, Traffic Analysis, Keylogging, Screen Scraping, Introduce System Fingerprinting, Blocking Internet Connection, Corrupting an Operating System, Encrypting Documents and Demanding Ransom (Ransomware), Compromising Computers, Bringing Down Hosts and Servers (Denial of Service) |

**Validation split by category.** Draw: `random.Random(0).sample(sorted(categories), n)`.
- **text, n=3:** val categories are Instruction, Programming Help and Social Interaction (15 strings); the other 12 are train categories.
- **code, n=2:** val categories are Exploiting System Vulnerabilities and Sending Out Spam Emails (10 strings); the other 8 are train categories.

**Pool by split:**
- Train docs: train-category attacks.
- Val docs: val-category attacks only. This is the model-selection set, so selection measures generalization to held-out categories.
- `val_id` slice: val docs × train-category attacks, 2 extra injected rows per doc paired with that doc's 2 clean rows. Reported, never used for selection.
- Test docs: the test file.

## 5. Intents
BIPIA code rows have the fields `error, code, context, ideal, context_url, context_author_url` (all 100 rows), and none has `question`.

The first error line is useless as an intent: it is `Traceback (most recent call last):` in 96/100 rows. The proposal uses the last line, which is the exception message:

> `Fix this error in my code: <last non-empty error line>`

Example: `Fix this error in my code: AttributeError: '_io.TextIOWrapper' object has no attribute 'strip'`.

Distinct intents per task under this proposal:

| Task | Distinct intents | Docs | Most docs sharing one intent |
|---|---|---|---|
| Code | 97 | 100 | 3 |
| Email (BIPIA question) | 42 | 78 | 5 |
| Table | 700 | 701 | 2 |

A few code last lines are odd: one row of asterisks, and one sklearn warning path. They are kept verbatim.

## 6. Size and splits
- Unique docs after V2 normalization and exact dedup: email 100→78, table 1000→701 (no cap), code 100→100.
- 5 split realizations (split_seed 0–4). Each is document-level, per task: n_val = n_test = round(0.15·n), the rest train.
- I also propose grouping near-duplicate docs (word-5-gram Jaccard ≥ 0.5) into the same split. The V2 audit found test docs with Jaccard > 0.5 to a train doc (CODEBASE_STATE §3.2). Whether to do this is open.

Projected main set with §7 applied (identical for every split_seed):

| Task | Docs (train / val / test) | Rows (train / val / test) |
|---|---|---|
| Email (77 docs) | 53 / 12 / 12 | 212 / 48 / 48 |
| Table (597) | 417 / 90 / 90 | 1668 / 360 / 360 |
| Code (100, 33 trimmed) | 70 / 15 / 15 | 280 / 60 / 60 |
| **Total** | **540 / 117 / 117** | **2160 / 468 / 468** |

Plus `val_id` (2 rows per val doc), `raw_clean`, `ambiguous` and `long`.

## 7. Length control
**Filter.** Input = `build_combined_text` (intent + context + span), checked with the real DeBERTa-v3 tokenizer (`hf_cache/tokenizers_hpc/deberta_v2_ckpt`).
- A doc is kept only if base tokens + the largest span cost of its register ≤ 512. Span cost includes the separator: text 30 tokens, code 204.
- The builder also asserts ≤ 512 on every inserted row after insertion.
- Benign items and donors are capped at the same span cost.

| Task | Removed by the filter | Base input alone > 512 |
|---|---|---|
| Email | 1/78 (1.3%) | 1 |
| Table | 104/701 (14.8%) | 66 |
| Code | 33/100 (33%) | 11 |

With the p90 span cost instead of the max, code still loses 22%.

**Code exceeds 25%, so I propose trimming.** For each code doc that does not fit:
1. Pick a seeded random line boundary outside fences.
2. Grow a window of whole lines (never cutting a fence) until the context budget is reached: at most 289 tokens, less for longer intents.
3. Use the trimmed context for all 4 rows and for `raw_clean`. Trimming is per doc, before insertion, and the same for both classes.

Projected share of context kept for the 33 docs: median 0.635, p10 0.32, min 0.218. These are upper bounds, since keeping fences whole can only keep less.

**Email and table are filtered rather than trimmed.** Their removed docs go to the eval-only `long` slice. The `long` slice uses test pools: per doc, the 4 row types × 3 positions = 12 rows. For code it holds the untrimmed versions of test-split docs only.

**Alternative:** trim table too, keeping all 701 docs in the main set; the header row is always kept. Say if you prefer this.

## 8. Gates (split_seed 0 test; fail → fix and rebuild)
**Content-free baselines:** each must be ≤ 0.60 macro-F1, point estimate, pooled over tasks. The bootstrap CI and per-task values are reported too.
- position/metadata LR on `position, separator, boundary_kind, offset/len(text)`;
- the first-paragraph rule;
- prefix-30 and suffix-30 TF-IDF+LR;
- `span_form_lr` (the same 9 features);
- a length-only LR on chars, words and tokens of the full input.

**Integrity assertions:**
- 0 overlap across splits in doc_id, exact text, attack strings, benign items, and donor sentences/chunks.
- Exact 1:1 balance per task × split.
- Every main row ≤ 512 tokens.
- Every row has exactly one span, recoverable from `char_offset`/`span_len`.

**`scripts/artifact_baselines.py` is parameterized, not copied.**
- Add `--dataset {v2,v3}`, `--split_seed` and `--out_version`. A loader returns (train, test, span) frames: V2 recovers spans with `extract_inserted_span` as now; V3 slices `text[char_offset:char_offset+span_len]`.
- The probe functions stay shared. The metadata LR and length LR are added for both datasets.
- V2 output must reproduce the committed `results/v2_probes/artifact_baselines` numbers exactly.

## 9. Risks and how we would report them
| Risk | How we report it |
|---|---|
| **R1** Benign forms are matched to attack forms using the test attack file's form statistics. | Disclosed. Only form statistics are used, never model outputs. |
| **R2** Lexical cues remain: 0.693 of text train attacks contain "you/your"; clean B's rate is not yet measured. "Addressee" is content, but a reviewer may call it a cheap cue. | Report a second-person-only LR next to the gates. |
| **R3** Table clean B is templated, not natural text. | Report results with and without clean B (clean A vs injected only). |
| **R4** Inserting any span into tables, or newlines into the 21 newline-free emails, is unnatural. It affects all classes equally, but `raw_clean` becomes trivially separable from inserted rows. | Raw-clean results are eval-only and reported separately as a false-positive rate. |
| **R5** `Language Translation` is in both the train and test text files. | Report test results with and without that category. |
| **R6** Test docs can overlap the training docs of V2 models, because V3 splits are new. | V2 checkpoints are evaluated on V3 only on docs that were in V2 test, or not at all. |
| **R7** Small cells: email test has 12 docs, code test 15; code's 33 docs are trimmed. | Per-task CIs, mean over 5 split seeds, and results with and without trimmed docs. |
| **R8** Category-held-out val uses only 15 text and 10 code strings, so selection is noisy. | Disclosed; selection uses doc-cluster bootstrap means. |
| **R9** The intent is still not needed for the label. V3 tests span content, not intent-conditioning (C3 remains the pairs work). | Stated explicitly. |
