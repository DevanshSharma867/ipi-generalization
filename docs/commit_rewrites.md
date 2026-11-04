# Commit rewrites

## 2026-11-04: commit-message trailer removed; trees identical

Reason: commit-message trailer removed; trees identical. Every commit on `main` from the root (`5dae025`, the earliest
commit carrying the trailer) to `e18e1d6` was rewritten with `git filter-branch --msg-filter`, which deleted the
`Co-Authored-By` line (53 commits carried it); the other 79 commits changed hash only because their parents did.
Each new commit has the same tree hash as its original (checked for all 132). Author, committer and dates are unchanged.
Tag `v2-frozen` was rewritten consistently: `a4a74a1` → `5b3adcb` (same tree `b450593`, same tagger, date and message).

Run manifests (`results/**/config.json`, findings provenance blocks, `docs/deviations.md`, `docs/paper_plan.md`) keep
the hashes they recorded; resolve them with this table. The pre-rewrite history is kept in the local branch
`backup/pre-rewrite-2026-11-04` (not pushed).

| old | new | subject |
|---|---|---|
| `e18e1d6a5ae4adce67027c88990aafbadc3e00a6` | `0e8e530ad88b8f360968aa72da44abe7dff2b3d9` | Pilot report: author labels match 39/40 drafted members; 1.03 min/pair (5.15 h per 300 pairs); STATUS |
| `44b88720174efaefbdd08fa9e1f9c30cbb71a0a7` | `9dbc4bf51e3675c0339ad556a3927ab797683f98` | Pilot labels (author, 20 pilot pairs) and pilot_report.py: one annotator vs intended labels, time per pair from saved_at |
| `1a90b8c3801ed55ae908d1c76861a0183ef1fd45` | `20ab1956c1aba2b6917cb9f96d54a8004a45ab97` | STATUS: gpunode8 drained by admins for the ECC fault; probe 432581 queued; GPU guard in every job |
| `b87548d68c8390625713194d81c947e28608f6a0` | `0af0bf393e29edfbd98710a9c661c04d5e4fd5bb` | GPU jobs fail fast on a bad card: 1-tensor CUDA test and known-bad UUID check (gpunode8 ECC card) before any work; runlog records the GPU UUID; short probe job |
| `5eba0b977dea51de49885c139887b87164bf739a` | `b29d544534365cf4b8a933065f1f32e7b3de2a45` | F021 on bf16 judges only (A20; 4-bit stays with the Liu reproduction) plus task_block_rate; F022: calibration-pool thresholds miss their FPR target on AgentDojo, task-disjoint halves as the in-domain contrast; STATUS |
| `a70c7db57f671568623c926bc2aaff4b1e524aeb` | `896f480d4042b2289700db5893a6c761001f9b26` | make_tables: halves macros loop over the two protocols only |
| `ae2502c06fa28a00aa577906b40bdb471241ba58` | `2e9e9f16f9eec5ec0252abdcab5dba97611fc6f0` | Transfer metrics: task_block_rate on AD-Clean/TB-Clean and the secondary task-disjoint-half protocol (A3), with the calibration-pool threshold on the same evaluation half for the calibration-transfer contrast |
| `cf947bd03b7f0558cc81b73b56c062c46f4b1a0b` | `36d051aa7a8c6e0734f55234f89468cc2543c5f9` | Transfer scoring: add protectai-small-v2 (HF access granted 2026-11-02) |
| `c2f8313d9b7def975c7d49601d588aef7f7dc981` | `56ccbdb439e6989f7c21ad1846f6733a83d9a0bf` | Transfer baselines: every published detector on AD-Matched, AD-Shift, TB-Matched (calibrated and oracle protocols, realized FPR, user-task CIs); F021 and paper table |
| `640f907d12e7fdd27bcc12a431f6b81a2a6ae75b` | `e32b02d274bd0878c9a20b7f407e688db4c3f475` | record_scores: judges have no truncation fraction |
| `ee59c134daf92828f470f95fbe5efdaef43e4fe8` | `fa6a4c4b138276665e7cc86f793cc7c732660f93` | Accept agent_v2 for ICT training despite the span_form_you_lr gate failure (deviation logged); report \|AUROC - 0.5\| per gate probe as a transparency metric |
| `3e3ce0c09cdfcc3ac5f35ced84018f29a66a6876` | `9e61deb5b1ffe9488ac990b88ce758ee1cb2f5fa` | F020: agent_v2 removes the within-train span length/form shortcut (CV <= 0.599) but fails the span_form_you_lr gate (up to 0.601 pooled, 0.633 per tool); not cleared for training |
| `5c97b168072597e642a77efadb960bc34a2c89e2` | `bb3fb05b27c943753cced66cd2d6606ba0570ff1` | Transfer metrics: calibrated (calibration-pool thresholds, A3/A16 gates) and oracle TPR@1/5%FPR, realized FPR, AUROC, FPR@0.5, user-task cluster CIs |
| `12a141919211679dfde346793d61f7dada98503e` | `9cdb464ccd1f24b2f3f35d07ff1a745aa99c3f3b` | Transfer scoring: calibration-pool rows carry the tool name the judge prompt needs (new data version); work dir per data version; jobs refuse a stale sets dir |
| `9b1675cdccedccf4d10f02f5ad3cfe9f00fe0ae1` | `646992d4060573b11e751363f13581077ee5e7a0` | Gate 2 PASS with the recovered AD pairing; transformers-5 shift explained (DeBERTa-v2 tokenizer rebuild, ModernBERT sliding-window RoPE theta): F017, F018 updated |
| `54d129553273c03fe8be1dbec6f630239cb59fd5` | `3d350c51d1d4c502acd9fed837f98ee539129717` | Adopt the recovered AD pairing (deviation 2026-10-31); transfer scoring jobs for every published detector on AD-Matched, AD-Shift, TB-Matched, AD-Clean, TB-Clean and the calibration pool, with run records |
| `08be6d6d60ace3634661e0e2467cd1a5e41f5a07` | `bd93cc7968120892b51168509c60730d0bb51bcf` | agent_overlap_audit: pool path separate from the training-data version (pool stays in agent_v1) |
| `88400e354ba3b763d8cf800cd89f52c0ed2814dd` | `d5183ca508b509b3644d38096d2c6f9d16c9b999` | agent_v2: match clean spans to each split's attacks (length, form), since agent_v1 train had a span length/form shortcut (within-train CV 0.61-0.66 macro-F1) |
| `aaa3c8e854b882e37d5dcfe75bd44d4ca452d960` | `434ed21501bf5f566c551a074175b3c41da95ff9` | Gate 2 envfix: test whether the transformers-5 score shift is the DeBERTa-v2 tokenizer rebuild (PIGuard, PG2) and ModernBERT's sliding-window RoPE theta (Horizon) |
| `518cacc6823735570ca092488dd13326cb3072ff` | `f086302b86946977c80af15abacbf1ed4f84a6c3` | Within-train CV diagnostic for span length/form shortcuts in agent_v1 (F019 follow-up) |
| `a370b0eb31cb35b3660e0a381719296587f09e57` | `996351b2a83ec4710096546366ae09d90ba534e8` | F019: agent_v1 calibration pool recorded (1,140 rows, sha256), overlap audit and content-free gates pass on split seeds 0-4 |
| `387b6fc79d36672637a11611c8fa4629806bc6b9` | `78ac75a0b7778c4613eebd7d205abe193f18404c` | Gate 2 batch/env test on CPU (diagnostic, agrees with the GPU reference); neutral paths in the residual run record |
| `cb7d881f920de16e776ba71863cf355a6104b9a2` | `82a5f2855e225691e03a1e95fce07ca627682776` | Agent-style synthetic tool outputs (A3): 19 tool types, calibration pool of 1,140 clean outputs, training data in V3's row design |
| `da5939d7b8448cc0f9d2cd434982c137cefabc0b` | `f36d005b38fc4add6a0117af0cf10c5d293b897c` | Gate 2 follow-up results: batch-independent scores, transformers-5 shift not padding, AD pairing recovered (A19 PASS; PIGuard row rule fails on hash-ordered participant lists), judge deterministic; findings F017, F018 |
| `dad273088bd3c3f3a82fb6ff548a2f474572e0d6` | `4db696a40506ea6216cbffd7c81f33d7d4741ffc` | gate2_followup residual: record the transfer data version |
| `16f089907d158ab1480f4ec10f949b78ac24788b` | `dc8929ca1b4f31d7fa565679fe82d6e8a6f5bad1` | gate2_followup residual: PIGuard rows over the validation bound vs AgentDojo's hash-ordered participant lists |
| `f87e25ec12f7a0ed0d086f72824e231112cbfeb6` | `ef2f1a57052b3c07114cad43a98ac8a28aec8664` | STATUS: generator download job 425712 |
| `43725c350fdff98cd09ceffca635ef267b940822` | `84ae4685367d1aa86af06071ffc6a4c5b1d63563` | Generator download as a resumable sbatch job (standing rule 10: no long login-node work) |
| `855f5edd27a8f7ee977fe10f8a888b4aac4f6cd7` | `408dceee3bdb498f8364945a42778262bb1fc9fb` | EVAL pair generator moves to open-weight Gemma 3 27B on the cluster (A24); external annotators in two teams (A25) |
| `e5354ff7ff87c9bccce8a4a26feaa63b461e2f50` | `63624f2f4865bad43ed33bf3ee5f124a083bf184` | A23: redefine ICT ablations (a) and (e) so each removes exactly one factor |
| `9ee9fec0d87a15dfc99e7cdf863bb41fd314773c` | `0d1cbe01d635d03bb3ba16963b745fb3f37d46ee` | Pre-register ICT ablations, selection rule, swap safety and batch tolerance (A22) and implement them |
| `bfe987219b446ca6d26f9594f7c648aa7b31e0e1` | `bafcb5bcf202591759090b1e0da30af2ede8b4e5` | Drop the row-level leakage claim (P4) from C1; add the 2027 venue scan |
| `57420ec7acebd7561d4d443754d55a86afade485` | `970ea8cc373709263aa709e7fb760cebfcad8df7` | Course demo: Gradio GUI scoring every cached detector side by side, pairs v0 tab, numbers.tex tab and a tiny CPU live-training run; drop the CLI paths that wrote frozen results/ and data/ |
| `733ffb22fc15f70bb26039cd6d1405d37efcdcea` | `74c0fd9254d4216b2d4607f95ca3fedf65ccecac` | Add ICT (C4): request/content cross-encoder with pair margin loss, request-swap augmentation and ablation presets |
| `f250fb326c0d6a03e517da6e67aa450425a82c4c` | `12ac50bd8e78808fc6c3589c5bef7f162d2012cd` | Start the paper draft (ACL template): intro, related work, threat model, C1 artifacts, V3 and pair construction, limitations |
| `5d31e39e46e7cd197ea60947eab26dd29c7ea679` | `16ca734b9fb28f232272031290376b182df4726c` | gate2_followup: --device for deberta_bs and report (a CPU run is recorded as gate2_followup_<part>_cpu, a diagnostic) |
| `ed4568018c2b37623113ee4cabd79a20cd3b61ba` | `718d7e704f5bae842f248443d8c680894244f29d` | Withdraw the frontier-API comparator (A21) to keep the project at about zero cost; add the pairs workstream to STATUS |
| `007c5a5b81b1540610b07557e917193c8edae1dc` | `bc2f183e9ee87dfd06ad1d291d6fe92ed5d8e8f4` | Gate 2 follow-up as three short jobs (batch test, pairing recovery, judge) so backfill can place them; each writes its own run record; candidate builds run 4 at a time |
| `7b56ebfd513a1fbde811fca8f4fffbc68f193f0f` | `855dd1fddeceb1e2c0c9329213cfc86ac557a462` | Gate 2 follow-up: leave wall-clock-timestamped AgentDojo rows out of pairing match and validation (reported separately) |
| `1680e3990b30cf652d663fb5166465ba114f2e90` | `556e16a78dba89ac21e5c44145b02728a6a8f611` | Gate 2 follow-up: AD pairing recovery, batch/env diagnosis, judge noise and bf16 (deviations logged; A20 judge precision before any H3 score) |
| `2f9f96c834aa5edff5a09be6a63efb69426c4a64` | `23530614a950fddcc1fedd8d7c6f38f732652b2c` | Gate 2 run record (FAIL: AD-Matched text differs from Liu's; large transformers-5 env effect; port check passes) |
| `7f0c3209b8ed2c1ff40f74598dbbf87dffe42c51` | `d0693dff8c4848171c4884e6b582865765b4aec6` | Gate 2 job: run with TORCHDYNAMO_DISABLE=1 (ModernBERT's torch.compile needs nvcc, absent on compute nodes) |
| `9bf92e15a2ccc8338bb3fa5d39914ad8ad2ad030` | `ef87a075015c0806215aba826ca481b8b2eb6430` | Transfer harness step b (detector wrappers) and Gate 2 job (Liu reproduction, A19) |
| `16fdae252f8726ee9f73e1b8faee63079c3f9181` | `e24431c1ca63858cd26a3e063c570e2153efcd77` | Pre-register Gate 2 (Liu reproduction) protocol as amendment A19 before any reproduction score |
| `5012198cfe07e49eb07eec5b67db69bdf7f0ff27` | `4e8c4fecc112790255df66b5e78089f418bc2534` | Findings F016: Gate 1 set sizes and status as provenance-gated macros |
| `3c144f60751fd7a5a9f0e77e900fef32ce23198b` | `3ea67a4507aad3f1a6f501b56d7f71d8f6ad0a37` | STATUS.md: trim to 40 lines |
| `11ef71728868b725764d32a48cef279572a0f759` | `f9992b747d7e1089f7c7a2fe230c043c0d2f1f1b` | STATUS.md: trim to 40 lines |
| `84758bb665506597de731c224b77986ecd2694f7` | `1eb5ad47eae4d4a6846d5aca740a829a1730eb07` | Add docs/STATUS.md: transfer harness state, next steps and open decisions |
| `d479b2d937d2abd86b6cd170249f57fae508f944` | `d49b4933a440827fd665bc163637df45e99cb13d` | Gate 1 run (transfer sets match Liu's released score files) and prefetch access records |
| `6c94421ab4e9f39d0e7e2290b4a9e40766ca5aa3` | `a0b9703896e9225aaa65d45b050aaf969e382bcb` | Set build: tau-bench steps in the AgentDojo env (Liu's lockfile has no pydantic); log the env of each step |
| `962958b2bdb9e45281847b8ec1038751240a1a06` | `d4373c34fd7238233bdb2758bc981d21accb4c7d` | Set build: run build_corpus.py without interpreter finalization |
| `62d1c18c4e9584f3da69ffcb6225173c72d3afb5` | `7e73ca1da62ee9be78ba776e41c2cb52b1e374b0` | Build job: uv binary fetched on the login node (compute nodes cannot reach GitHub release assets); probe every download host |
| `2b3148a7a1d145e0ababf9be6ea78d6fe251e5f7` | `c80fe85aa542cc9bf3fec8d225a270a9ac0914be` | Build job: take uv from its checksum-verified release binary (compute-node python3 cannot pip-install it); probe a small PyPI page |
| `96f986579c6866440920ee22db51be8133715247` | `32328f4e133fe4b39438e8716898535e19ca4a99` | Build job: install the login node's uv version on scratch, since compute nodes have no uv |
| `0087b6b69a58a0b1566251cfc852379274fc2585` | `d4c00bda9095c9f5d2b3a65eee3c5a91160600fe` | Pre-register the intent-pair construction as amendment A18 before any generation |
| `78a972ffd0d42670325908e4e35314c5239e2967` | `4b24699de03858c503384618af3f08f8e8c890dc` | Registry: InjecGuard as an alias of PIGuard (Liu 2026 §4.1), not fetched separately |
| `7c75099ddd25970d895308ba7ed617560758828a` | `bb02d544e611f8b15530c2ea624e78bf85622834` | Intent-pair specification v1 (C3): crossed quads, decision rules, generator design, annotation tooling, 20 pilot pairs |
| `b35150f04d152f2f1a305800b38dbafea7427965` | `2c5ed83a9f6916324fe0334eac03e759c2580317` | Detector registry with pinned revisions; login-node prefetch reporting gated access |
| `8f09892e9f8561b65fb217b4eb11d7f4bfabd5a0` | `7edae339e92554d52ed56ee6bdd27081ab41d57a` | probe_report writes its figure only into its run directory; make_tables copies it to paper/figures |
| `2592b613a2815106eddfd7e8b483dea26fa1410a` | `7cd19a523fb293775951e259c8c8bad871e3c761` | Move F007 and F012 to the clean re-scored runs (both now full); add F015, the variant-rule judge |
| `881c90f2627c198b0b204a5a389512b9510d0316` | `635c2eeff24e3d920060f33f46303c8d31ee0afc` | Cluster storage layout: venvs and caches on scratch, set build as a resumable sbatch job |
| `71750a5d3acab4f072ba48aa885cba95b14473a2` | `886b3804cf558f21df30b8500fea009a262317c3` | Probe report from the clean re-scored runs (adds the variant-rule judge) |
| `86d9a087f1fbc7768a844aa056bf544aaa4cfb27` | `3e54318ace424f2b7ec15851a672b08847a58db0` | Read the variant-rule judge runs in the probe, truncation and TPR reports; allow glob exclusions in findings macros |
| `c879deeb323c6ee7f59791b2f05dd86205e955ab` | `86d8b909c489558b803aa1bd6410271e9a336e31` | Add the clean re-score of the V2 probe runs and the V2 val calibration scores (job 422174, commit 7f56fc9) |
| `0e015c331913c3b40da823d727c776c6212d5c26` | `305c7b72a7d85b97b5e69416a76dc44772e63f18` | GPU smoke test: time each stage and isolate the MPS VRAM-cap variable |
| `0f7b79e3e8430348e9d999c39a5b5ae215f036f6` | `d6799300e4e5c1231fe91133b290f319fe42f790` | Transfer harness step a: rebuild Liu 2026 sets from pinned sources; Gate 1 check; GPU smoke test |
| `228ca7cefe3c2f018e4b0dd2be6288696b4a230c` | `ea80c5c5c610291a9d028f364fbacbb759ee8cc3` | Pre-register the calibrated-protocol gate, AD negatives and the Liu 2026 reproduction protocol (A16, A17) |
| `154295048a50e6ce50785e0c91f4cf10fdbfd5ac` | `4cdae64482dd84b820514aad3f9c135b3a8ee4f6` | Task-stratified bootstrap and task-macro metric; split-independent long slice; standalone judge re-score job |
| `7f56fc97beb43a6dcd1eb2f5962ac3c05cc2dee4` | `c2834148fb8af653cd86450a326838761781c19f` | One shared Yes/No rule for every judge; task-macro V3 headline; frontier comparator deadline (A10-A12) |
| `1d897ead369c82828c30b61cf0f9cec45b6edf25` | `059d20c71b916cf4d8be7487a65416d0601fdfca` | Record V3 pre-freeze gate and integrity results (F013, F014) on the draft benign pool |
| `507e9ee5acdbdab7decf9737fd7291c834e32934` | `5d3a531abb0153283dfea8188f9607aa980e51c0` | Add an independent V3 integrity check and the hand-review sheet for the benign pool |
| `eec967482c2ce35f5f92fd17303fee69a624b647` | `1e707fba601f2080616199dd51eec6f830ab4891` | Freeze the frontier-LLM comparator config and prompt; record build order and annotation plan (A7-A9) |
| `45303c9ef8320610fe6c42f92b03033d3e9e0e4a` | `9e6e1344dd0829a4569fd6b402c41ad3073e51c8` | Add PromptArmor to related work; it is the basis of H3 comparator (c) |
| `7abd155776e5794e74a285db8b55e2f072c5c7e0` | `569aea68ebb0c747761a6e1ce46bfe8aac6e2dca` | Amend pre-registration before scoring: request-aware H3 comparators, non-inferiority H3b, calibration pool |
| `82e66b61315f0f6e9d2b1cdda4ec1700e8e05a4c` | `b0ae9850fcce1d12214eac75fcd8ad113c6639dd` | Pre-register claims, hypotheses H1-H4, decision table and eval-set roles; draft related work |
| `45771cd8bb15b2d07385cccd9dac15b09bd5e763` | `e633ab45561788813615c2a7ae63fe9ef300615e` | Generate the run registry from run directories; replace typed counts in the probe findings with macros |
| `a03181369f5d77097a2af420ab68fcd101ea543b` | `2133a7347409e8f4474ef8e0f0c5438eabe6995b` | Convert findings to provenance-tagged entries F001-F012; record clean TF-IDF probe re-score |
| `a59aae446032dd8cab107cfb79b86ea44ff441ed` | `f02feea02225dcbbe2a72da5c97abea07a5d5a34` | Gate paper macros on findings provenance; cluster InjecAgent by user case; merge V2 re-score job |
| `9b135c29f9623f96860fb776baffed5a576be8c1` | `2dfe392c2e1c103baf940ba0b46105eab25e29fa` | Record V2 val TF-IDF calibration scores and the TPR@FPR recompute of saved V2 scores |
| `c80a037ac43645090b9884f3928dda73b5478036` | `34d039d92faffde1cc507e56bb8d25346aea05bb` | Refuse runs on a dirty tree; add oracle/calibrated TPR@1%/5%FPR with clean-count gates |
| `deb09f8a74b6c7a06734c6565563f515daac3511` | `66e68c5dbe310318c35a1f836a27f0946c20dc2a` | Record LLM-assisted origins for V2 benign pool, table templates, neutral pool and judge prompt v1 |
| `8816bdd750c2b80ceb7d4515f42e4fe63130d6a0` | `3415c203ecd689dd2c101094029551cc5ee6bf6b` | Add data provenance; correct pair_specs_v0 to LLM-drafted; stop tracking local instructions file |
| `cf8f76db1a7bc2a4087905b19978f330532fc09a` | `85a6e2514cba29fe800972215cc0a37aec3c72cc` | Round-3 dry run: all V3 gates pass on split_seeds 0-4; review sheet regenerated |
| `4a2ad988e9c6ba927ed6a279dfd13aabc7e31545` | `478af4c67d4cdedc2bb0860451a0d7bc8c104ae4` | Rewrite code-register clean-A items: denser helpers matching train-attack chars-per-word and digit rate |
| `3ddcb3e347d6e484861932aca28f5379fcbbc4de` | `d50333bcee2abbe4e6c7de595b1c9ae69eba38d7` | Record R4; assert clean-B selection statistics come from train attacks only |
| `ae1a6e5805886343679deb6ee1104174cb0ce115` | `f67cb6c0c4d5903cb0ff6cbb661d934a258260d4` | Round-2 dry run iteration 3 results; review sheet regenerated from the revised draft pool |
| `16f2119fd1306d594cab5c19ec4ef5ae9af07ef5` | `8990022c08383d8da6c384f2447d6a1249226634` | Round-2 dry run iteration 2 results; iteration 3 fixes (more digit-free table cells, prose comment lines trimmed from code items) |
| `c134cfe757d2f45acbc2121738beef49fb6a55ea` | `6590aa6318f601b3e2013330656f1583e5e068a3` | Round-2 dry run iteration 1 results; iteration 2 fixes (chars-per-word bin, digit weight, fewer digits in code items) |
| `65bebfd9a0986dc063a5a4d381c290e019ca273c` | `ae5cd7e8f70d5abf0b8f7555e24505aa88b61cad` | Revised draft pool (URL lines, email second person, longer code, cross-pool near-templates); bin+digit clean-B order |
| `4f4dbec901754e89c06d94d93b3a0f64619ea92a` | `cd0bca07095c0d7a3876dd61e2884a3bbc6cc6af` | V2 artifact baselines re-check after R1-R3: metrics and predictions identical |
| `a537610a14ed11e290a901f140608338e90701d2` | `7beb8bd50e553b2792969580c8b965bfcc9dfa46` | Implement R1-R3: masked context gates, insertion-site gate, multi-seed pass rule, binned code clean B |
| `60374195387a2deee65e51985a13c1d866d53fd4` | `26db0d8288c2b0aa0fb2a47a981637a9a58d552b` | Record V3 design decisions R1-R3 (round 2) |
| `934ae28266e0ab08fb89bbf06d4fcb78326b2719` | `e02b1d28425c44134534664b884db7ae9774f861` | V3 dry-run gate results (3 iterations), draft pool check, code map |
| `fc9aa463dcc436133faf146bda11b010edf77594` | `27ce24988f04e78872631ed77152f4f55bf7d11e` | Benign review sheet, offline review page, apply_review |
| `02b7ebb5c15ccdb8051413cfc5500a95624ce76b` | `cdae0a3b5a3fc9290b567bb445d95b533339644b` | V3 dry run iteration 3 fixes: digit-first clean-B selection, 3 cells per table donor, span-masked window diagnostics |
| `1df796c26486f839bbc744dcdf18eb844facb56f` | `29fc79e7c52e8eb80bd3eee2001725b2b88f08b8` | V3 dry run iteration 2 fixes: span whitespace normalization, shape-matched clean B, longer code items |
| `24198dfad57ab5bf6a3f80627cf57554ad4a48be` | `85522f9657e9ca7fbdc5f3b6008347d1885679b0` | V2 artifact baselines rerun with the parameterized script |
| `3727d336c94372d3fdd0a5688db9d169da3b4039` | `566f119db473291dd31a8a32c32c3132e310bf4a` | artifact_baselines: log per-position F1 and span-feature means for v3 gate diagnosis |
| `b23108b6305cab7d2cb4ab0f875a8f68451c3dda` | `7752fd76c62c75e27ead0e1e0a1f80e00ed590cf` | V3 pipeline, dataset-parameterized artifact gates, draft benign pool |
| `4ea7407a33a467e5d56e9cf9efcf7e9c68e152f6` | `72311b1a778b5b55ac0744652ee67a18027b5c70` | Record approved V3 design decisions D1-D7 |
| `b22839502c698274e34365c8e15995c76d0d8fd8` | `a67d2f8e5d49ab67e002d4b7b1f853def1971f12` | V3 design (Phase A) for approval, with its raw-data fact run |
| `2b1eff09b6076adb2c8e5432ea48cd3d671b8167` | `9091d1fc211b53ef8d803610f13fae1e7b493b44` | Add read-only V3 design fact check over raw BIPIA |
| `d7e8a88d759147c9217f42594fbb5b8e732cac57` | `326478538bab56ce4604e4c9447dc334bae75928` | CLAUDE.md: add make_findings.py and docs/findings.md to the code map |
| `56c7b2f0e6353b335639ab0663a0ebd692a664e0` | `652c7e9f79e524b69810c8ccc46f673fa3583506` | Regenerate report with the judge V2-test run; add docs/findings.md |
| `2a370a8660e39c543151eaded76f54269eb450df` | `767c8d11fc86d5e259542af13e3c30ac4a52ef99` | runlog: keep the leading status column when parsing git status |
| `783f3f3005c9b0a9bdb068df09a545f4dce6c7d1` | `6b75e065f9942c8c488bdee52003010ba9c03d70` | V2-test group cells, summary.json bug table, findings generator |
| `d10916a87dc85aeeb36bb443114b901454c56577` | `bb700918e34804245e94e5bf455cf7144f66c646` | Add judge V2-test scores (Slurm job 407923) |
| `5a2a782dbebdb6b6b3a96a63b69f442be65b85c6` | `faf09fd75879fd38b537029bac570a9886619c04` | Seed-variance and truncation tables; regenerate report outputs |
| `1e1e820ff7c2b438dd1c1229a5cedd95c1402370` | `fc50ea15ecb39f96c91272c4b5461a2586c50619` | Judge V2-test job; raw-clean flag rate and P4 - raw contrast |
| `a7cbcd8980e78420d53bfe357cb484e4c6117964` | `4fa6462dd17f9a20883c8809ffd7212d9cbd9d94` | Truncation control for the position probes |
| `48aba4c6487635089a3cc4cfbb29924288c90687` | `76312c68e789c45715d717b3503daadf9958835a` | Probe report for all models, with a DeBERTa seed-mean row |
| `f29060c1ecb8aed6dc573248be87cc5df174ca56` | `772acb916f4863b0e0ba2e613d581eaa0bef07a5` | Add HPC probe scores (Slurm job 404404) |
| `6f81bc260bb8830d3ebfeee7f62e27b6b4d1e506` | `139c912dc156c95b3fbf485956cd1acdad52d414` | runlog: git_dirty ignores run outputs under results/ |
| `518b572d69634dc96ef74ca48f074bc32a4bf8ad` | `cd4645646135e71e2f604726b8fcc14944f46c77` | CLAUDE.md: record whitespace visibility and the summary.json Recall@FPR bug |
| `a02a1417a16a486a0e640b015f475dfa2a5e96d8` | `caaad7d1082eddb7d9b3857227a0f8b0617ddc5d` | Add probe report, figure and generated probe tables/macros |
| `1c29f2137664a3b350c198c7e9764d461074da4c` | `3dab84dba02cfd98f6f6416933ae98e1e29f384d` | Score probes: shared scorer script, local TF-IDF run, Slurm job for GPU models |
| `8861d87a6cadde0414f30721aca814e2706944f1` | `78d5c186c915c363313f79ed569b5cdb262fcf3e` | Build position-matched probe sets P1-P4 from V2 test docs |
| `1b03f071523d82fc6f2ca2a4f7726cbfa696d679` | `b02e626b4cb18a42d4470161bbfd9f2bb5393087` | Add artifact-only baselines on V2 test |
| `c61e42382fbbe3b3f08050681838a7477aa8d8f0` | `2516068359f70bf372f5b0676ef23112cb8cf4d9` | Document the new infrastructure in CLAUDE.md; commit the audit |
| `c04ca0931d8406075201fafa3f5e6114d9478f49` | `fe6b5c52113ef48510a54a5354348bfd068f1a4d` | Pin environment and parameterize Slurm site settings |
| `d6a5a2d29fa33a195b39962472e6951fd9d3be08` | `f2628239268e125b06e74a657458d2d51819d73a` | Diagnose the DeBERTa tokenizer regex warning |
| `ede4bab44f66d3e762a0cbfbb1217a68d0a1e698` | `5bb1127c47532359ed0c372459a45f2d87d099c2` | Standardize V2 predictions and script every reported V2 number |
| `f01a23c4a15e8d83c2a9fbcedba1fb0c044d2e70` | `233bfd228a9c944790769db98494213e163811c6` | Add run bookkeeping and frozen-path guard |
| `1b02945032b7fff9d57a40a9f8be51f428171ca8` | `66e816915afb7d5ffc019b47991e5529c8a60b97` | Add src/metrics.py as the single metrics implementation |
| `f3346c6fe1f73c3b151cb3c82afcacd9e9669042` | `cf1b9252fb83334f4b2a7708a3e8e4873d7ea6a4` | Rewrite CLAUDE.md to match the audited repo at v2-frozen |
| `a4a74a184b6d24882749545a032b78860adffa35` | `5b3adcb04a34f2e3cc95a1966768f08f8914140a` | Ignore local CLAUDE.md |
| `bab750e812c64d11fdfe68c4f4a1cc0dccd6c0e0` | `065a4eab88c86be56444f23ab94cc36feca77edd` | Score detectors zero-shot on InjecAgent, deepset, and pairs; add size-matched LOTO control |
| `f141238f270382f540381be1732e6d61c7ebf672` | `d6b41156410c5afc4ac3c9293b186f0260d58550` | README: flag training-size confound in TableQA LOTO fold |
| `e18a34fa8a59d9e1cc388e0bb47e518970596eb3` | `74c9612f63296b46135e463a97613a7315604fb4` | Add query-counterfactual pair pilot and multi-seed DeBERTa LOTO |
| `dc1eda6efeda52e1d2aed6544b2165f217a79249` | `e1979efcc140ba0d33489f07fd452eab7f598731` | Add DeBERTa-v3 results and BITS HPC Slurm config |
| `01d1c788b637ab4a30c68308c5f452202c2e1d22` | `7eb7721f34b740e01590659144ce31069f03b10b` | Remove report and course docs; rewrite README |
| `5dae025ac5f7368efc50d1881e43fa61b7ec10c8` | `54f3cbe561024cde0afbc4bdd90f3c17095c1738` | Initial commit: IPI detection on BIPIA (Phase 2) |
