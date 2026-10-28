# Related work (draft)

Draft prose for the paper's related-work section, plus BibTeX for every cited paper.

**Verification (2026-10-28):**
- Each arXiv entry's title, authors, first-version year and primary category were checked against the arXiv API
  (`export.arxiv.org/api/query`).
- ACL Anthology entries are the Anthology's own `.bib` files, unchanged.
- Peer-reviewed venues that could not be verified on arXiv or the ACL Anthology are **not** put in BibTeX. Those
  papers are cited as arXiv preprints. Before camera-ready, replace them with the venue version after checking it.
- No entry was typed from memory.

**Public detectors and models without a paper** (ProtectAI v2, Prompt Guard 2, Horizon-Labs, Wolf Defender, Sheltron,
Qualifire Sentinel, …) are cited by Hugging Face model id and revision in the paper's experimental setup, not in the
bibliography. Their ids and revisions are taken from Liu (2026)'s released scripts when we pin them. They are not listed here yet.

## Benchmarks for indirect prompt injection

**Origins and benchmarks.**
- Indirect prompt injection (IPI) was introduced by Greshake et al. (2023).
- Liu et al. (2023) formalize prompt-injection attacks and defenses.
- BIPIA (Yi et al., 2023) inserts attack strings into email, table, code, web and summarization contexts, at the
  start, middle or end of the document.
- InjecAgent (Zhan et al., 2024) places attacker instructions in tool responses of tool-using agents.
- AgentDojo (Debenedetti et al., 2024) and τ-bench (Yao et al., 2024) provide agent environments whose tool outputs
  can carry injections.

**How we differ.** Our C1 audits how BIPIA-style insertion data is *constructed*: label-dependent position, whitespace
rewriting, span surface form and row-level leakage. We show that content-free features predict the label.

## Detectors and their evaluation

**Content-only classifiers.**
- PIGuard/InjecGuard (Li et al., 2025; arXiv v1: Li and Liu, 2024) targets over-defense on trigger words.
- Sentinel (Ivry and Nahum, 2025) and Prompt Guard 2 in LlamaFirewall (Chennabasappa et al., 2025) are fine-tuned classifiers.

**LLM- and internals-based detectors.**
- DataSentinel (Liu et al., 2025) detects injections with a fine-tuned LLM via a known-answer test.
- Attention Tracker (Hung et al., 2024) scores attention distraction.
- MELON (Zhu et al., 2025) compares agent actions with and without the user task. It is a defense on agent
  trajectories, not a text classifier.

**Evaluation critiques.**
- Liu (2026) re-evaluates 15 detectors and two task-aware LLM judges on AgentDojo and τ-bench tool outputs, replayed
  without an LLM:
  - BIPIA rankings do not transfer.
  - The best BIPIA detector was trained on BIPIA attacks.
  - Input form, not attack-string overlap, determines where a detector works.
- Fomin (2026) shows that in-distribution splits overstate the generalization of malicious-prompt classifiers, and
  proposes leave-one-dataset-out evaluation.
- Akinrele and Gowda (2026) find that detector rankings depend on regime and threshold.

**Shortcuts.**
- Li et al. (2026) show that defenses learn surface heuristics (position, trigger tokens, topic).
- Kale and Vishwanath (2026) show that, on AgentDojo, conversation position and length alone nearly solve the
  standard probe evaluation. Training on matched negatives (the same conversation without the injection, and harmless
  look-alikes) reduces false alarms.

**How we differ.** Our C1 is confirmatory for "IPI evaluations contain content-free shortcuts". It is new for the
measured artifact inventory of BIPIA-style insertion data and for the artifact-controlled V3 rebuild. C2/H1 asks a
question none of these studies answer: does removing construction artifacts from training data improve transfer?

## Task-, intent- and alignment-aware detection

**Prior work** asks whether an instruction in external data deviates from the user's task.
- TaskTracker (Abdelnabi et al., 2024) detects task drift from activation deltas.
- PromptSleuth (Wang et al., 2025) reasons over task-level intent.
- IntentGuard (Kang et al., 2025) analyzes a reasoning model's instruction-following intent.
- AlignSentinel (Jia et al., 2026) is the closest.
  - It is a three-class classifier (misaligned instruction / aligned instruction / no instruction) on attention
    features from a backend LLM, with a GPT-4o-generated benchmark.
  - In its benchmark, aligned and misaligned inputs are *different texts* under the same user prompt and tool.
  - We found no code, data or model release.
- Liu (2026) evaluates two task-aware LLM judges and a request-conditioned classifier variant (Sheltron with a goal
  segment). Giving the request to the small classifier did not help.

**How we differ.**
- Our pairs (C3) hold the document and the inserted instruction byte-identical and vary only the user request. Any
  request-blind detector is therefore at chance by construction.
- This applies contrast sets (Gardner et al., 2020) and counterfactually augmented data (Kaushik et al., 2020) to IPI.
- ICT (C4) uses such pairs as a training signal for a black-box text cross-encoder. Request-conditioning itself is not new.

## Shortcut learning and dataset artifacts (general)

- Annotation artifacts let hypothesis-only models solve NLI (Gururangan et al., 2018).
- Models adopt syntactic heuristics (McCoy et al., 2019).
- Shortcut learning is a general failure mode (Geirhos et al., 2020).

C1 is an instance of this line of work for IPI detection data.

## Models used

DeBERTa-v3 (He et al., 2021) is the backbone of our trained detectors.

## BibTeX

```bibtex
% ---------- ACL Anthology (official .bib, unchanged) ----------
@inproceedings{li-etal-2025-piguard,
    title = "{PIG}uard: Prompt Injection Guardrail via Mitigating Overdefense for Free",
    author = "Li, Hao  and
      Liu, Xiaogeng  and
      Zhang, Ning  and
      Xiao, Chaowei",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.1468/",
    doi = "10.18653/v1/2025.acl-long.1468",
    pages = "30420--30437",
    ISBN = "979-8-89176-251-0"
}

@inproceedings{zhan-etal-2024-injecagent,
    title = "{I}njec{A}gent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents",
    author = "Zhan, Qiusi  and
      Liang, Zhixiang  and
      Ying, Zifan  and
      Kang, Daniel",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {ACL} 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.624/",
    doi = "10.18653/v1/2024.findings-acl.624",
    pages = "10471--10506"
}

@inproceedings{gururangan-etal-2018-annotation,
    title = "Annotation Artifacts in Natural Language Inference Data",
    author = "Gururangan, Suchin  and
      Swayamdipta, Swabha  and
      Levy, Omer  and
      Schwartz, Roy  and
      Bowman, Samuel R.  and
      Smith, Noah A.",
    editor = "Walker, Marilyn  and
      Ji, Heng  and
      Stent, Amanda",
    booktitle = "Proceedings of the 2018 Conference of the {N}orth {A}merican Chapter of the {A}ssociation for {C}omputational {L}inguistics: Human Language Technologies, Volume 2 (Short Papers)",
    month = jun,
    year = "2018",
    address = "New Orleans, Louisiana",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N18-2017/",
    doi = "10.18653/v1/N18-2017",
    pages = "107--112"
}

@inproceedings{mccoy-etal-2019-right,
    title = "Right for the Wrong Reasons: Diagnosing Syntactic Heuristics in Natural Language Inference",
    author = "McCoy, R. Thomas  and
      Pavlick, Ellie  and
      Linzen, Tal",
    editor = "Korhonen, Anna  and
      Traum, David  and
      M{\`a}rquez, Llu{\'i}s",
    booktitle = "Proceedings of the 57th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P19-1334/",
    doi = "10.18653/v1/P19-1334",
    pages = "3428--3448"
}

@inproceedings{gardner-etal-2020-evaluating,
    title = "Evaluating Models' Local Decision Boundaries via Contrast Sets",
    author = "Gardner, Matt  and
      Artzi, Yoav  and
      Basmov, Victoria  and
      Berant, Jonathan  and
      Bogin, Ben  and
      Chen, Sihao  and
      Dasigi, Pradeep  and
      Dua, Dheeru  and
      Elazar, Yanai  and
      Gottumukkala, Ananth  and
      Gupta, Nitish  and
      Hajishirzi, Hannaneh  and
      Ilharco, Gabriel  and
      Khashabi, Daniel  and
      Lin, Kevin  and
      Liu, Jiangming  and
      Liu, Nelson F.  and
      Mulcaire, Phoebe  and
      Ning, Qiang  and
      Singh, Sameer  and
      Smith, Noah A.  and
      Subramanian, Sanjay  and
      Tsarfaty, Reut  and
      Wallace, Eric  and
      Zhang, Ally  and
      Zhou, Ben",
    editor = "Cohn, Trevor  and
      He, Yulan  and
      Liu, Yang",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EMNLP} 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.117/",
    doi = "10.18653/v1/2020.findings-emnlp.117",
    pages = "1307--1323"
}

% ---------- arXiv (metadata from the arXiv API; venue versions not verified here) ----------
@misc{liu2026passing,
  title         = {Passing the Test You Trained On: Re-evaluating Prompt-Injection Detectors for {LLM} Agents},
  author        = {Liu, Zhuowen},
  year          = {2026},
  eprint        = {2610.03448},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{jia2026alignsentinel,
  title         = {{AlignSentinel}: Alignment-Aware Detection of Prompt Injection Attacks},
  author        = {Jia, Yuqi and Wang, Ruiqi and Wang, Xilong and Xiang, Chong and Gong, Neil},
  year          = {2026},
  eprint        = {2602.13597},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{yi2023bipia,
  title         = {Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models},
  author        = {Yi, Jingwei and Xie, Yueqi and Zhu, Bin and Kiciman, Emre and Sun, Guangzhong and Xie, Xing and Wu, Fangzhao},
  year          = {2023},
  eprint        = {2312.14197},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  doi           = {10.1145/3690624.3709179}
}

@misc{greshake2023signed,
  title         = {Not what you've signed up for: Compromising Real-World {LLM}-Integrated Applications with Indirect Prompt Injection},
  author        = {Greshake, Kai and Abdelnabi, Sahar and Mishra, Shailesh and Endres, Christoph and Holz, Thorsten and Fritz, Mario},
  year          = {2023},
  eprint        = {2302.12173},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{liu2023formalizing,
  title         = {Formalizing and Benchmarking Prompt Injection Attacks and Defenses},
  author        = {Liu, Yupei and Jia, Yuqi and Geng, Runpeng and Jia, Jinyuan and Gong, Neil Zhenqiang},
  year          = {2023},
  eprint        = {2310.12815},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{debenedetti2024agentdojo,
  title         = {{AgentDojo}: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for {LLM} Agents},
  author        = {Debenedetti, Edoardo and Zhang, Jie and Balunovi{\'c}, Mislav and Beurer-Kellner, Luca and Fischer, Marc and Tram{\`e}r, Florian},
  year          = {2024},
  eprint        = {2406.13352},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{yao2024taubench,
  title         = {{$\tau$}-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains},
  author        = {Yao, Shunyu and Shinn, Noah and Razavi, Pedram and Narasimhan, Karthik},
  year          = {2024},
  eprint        = {2406.12045},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}

@misc{li2024injecguard,
  title         = {{InjecGuard}: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models},
  author        = {Li, Hao and Liu, Xiaogeng},
  year          = {2024},
  eprint        = {2410.22770},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  note          = {arXiv version of li-etal-2025-piguard; cite the ACL version}
}

@misc{abdelnabi2024drift,
  title         = {Get my drift? Catching {LLM} Task Drift with Activation Deltas},
  author        = {Abdelnabi, Sahar and Fay, Aideen and Cherubin, Giovanni and Salem, Ahmed and Fritz, Mario and Paverd, Andrew},
  year          = {2024},
  eprint        = {2406.00799},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{liu2025datasentinel,
  title         = {{DataSentinel}: A Game-Theoretic Detection of Prompt Injection Attacks},
  author        = {Liu, Yupei and Jia, Yuqi and Jia, Jinyuan and Song, Dawn and Gong, Neil Zhenqiang},
  year          = {2025},
  eprint        = {2504.11358},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{hung2024attentiontracker,
  title         = {Attention Tracker: Detecting Prompt Injection Attacks in {LLMs}},
  author        = {Hung, Kuo-Han and Ko, Ching-Yun and Rawat, Ambrish and Chung, I-Hsin and Hsu, Winston H. and Chen, Pin-Yu},
  year          = {2024},
  eprint        = {2411.00348},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{wang2025promptsleuth,
  title         = {{PromptSleuth}: Detecting Prompt Injection via Semantic Intent Invariance},
  author        = {Wang, Mengxiao and Zhang, Yuxuan and Gu, Guofei},
  year          = {2025},
  eprint        = {2508.20890},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{kang2025intentguard,
  title         = {Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis},
  author        = {Kang, Mintong and Xiang, Chong and Kariyappa, Sanjay and Xiao, Chaowei and Li, Bo and Suh, Edward},
  year          = {2025},
  eprint        = {2512.00966},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{zhu2025melon,
  title         = {{MELON}: Provable Defense Against Indirect Prompt Injection Attacks in {AI} Agents},
  author        = {Zhu, Kaijie and Yang, Xianjun and Wang, Jindong and Guo, Wenbo and Wang, William Yang},
  year          = {2025},
  eprint        = {2502.05174},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{chennabasappa2025llamafirewall,
  title         = {{LlamaFirewall}: An open source guardrail system for building secure {AI} agents},
  author        = {Chennabasappa, Sahana and Nikolaidis, Cyrus and Song, Daniel and Molnar, David and Ding, Stephanie and Wan, Shengye and Whitman, Spencer and Deason, Lauren and Doucette, Nicholas and Montilla, Abraham and Gampa, Alekhya and de Paola, Beto and Gabi, Dominik and Crnkovich, James and Testud, Jean-Christophe and He, Kat and Chaturvedi, Rashnil and Zhou, Wu and Saxe, Joshua},
  year          = {2025},
  eprint        = {2505.03574},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{ivry2025sentinel,
  title         = {Sentinel: {SOTA} model to protect against prompt injections},
  author        = {Ivry, Dror and Nahum, Oran},
  year          = {2025},
  eprint        = {2506.05446},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{fomin2026benchmarks,
  title         = {When Benchmarks Lie: Evaluating Malicious Prompt Classifiers Under True Distribution Shift},
  author        = {Fomin, Max},
  year          = {2026},
  eprint        = {2602.14161},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{li2026surface,
  title         = {Defenses Against Prompt Attacks Learn Surface Heuristics},
  author        = {Li, Shawn and Yu, Chenxiao and Ni, Zhiyu and Li, Hao and Peris, Charith and Xiao, Chaowei and Zhao, Yue},
  year          = {2026},
  eprint        = {2601.07185},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR}
}

@misc{kale2026probes,
  title         = {Prompt-Injection Probes Learn the Conversation, Not Just the Injection},
  author        = {Kale, Kaan and Vishwanath, Sriram},
  year          = {2026},
  eprint        = {2610.18408},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{akinrele2026regime,
  title         = {Prompt Injection Detection is Regime-Dependent: A Deployment-Aware Evaluation with Interpretable Structural Signals},
  author        = {Akinrele, Akindoyin and Gowda, Shreyank N},
  year          = {2026},
  eprint        = {2605.26999},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{kaushik2019counterfactual,
  title         = {Learning the Difference that Makes a Difference with Counterfactually-Augmented Data},
  author        = {Kaushik, Divyansh and Hovy, Eduard and Lipton, Zachary C.},
  year          = {2019},
  eprint        = {1909.12434},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@misc{geirhos2020shortcut,
  title         = {Shortcut Learning in Deep Neural Networks},
  author        = {Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
  year          = {2020},
  eprint        = {2004.07780},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.1038/s42256-020-00257-z}
}

@misc{he2021debertav3,
  title         = {{DeBERTaV3}: Improving {DeBERTa} using {ELECTRA}-Style Pre-Training with Gradient-Disentangled Embedding Sharing},
  author        = {He, Pengcheng and Gao, Jianfeng and Chen, Weizhu},
  year          = {2021},
  eprint        = {2111.09543},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
