CS F429 Natural Language Processing - Part 1 Proposal
Generalization and Robustness of NLP-Based Indirect Prompt-Injection Detection

1. Problem Statement

Large Language Model (LLM) applications increasingly consume untrusted external content - retrieved
web pages, emails, tables, code, and other documents - as part of retrieval-augmented generation (RAG) and
agentic pipelines. This integration introduces indirect prompt injection (IPI): an attacker embeds instructions
inside the external content itself, rather than in the user’s own prompt, in an attempt to hijack the model’s
behavior away from the user’s original intent [1].

We study the following research question: can NLP-based text classifiers detect indirect prompt injec-
tion reliably when the attack wording, task context, or document type differs from what the detector
was trained on, and can such detectors distinguish maliciously injected instructions from superficially
similar, benign imperative language that occurs naturally within documents (e.g., “ignore the intro-
duction and focus on the results section”)? This is fundamentally a generalization and discrimination
problem rather than a straightforward binary classification task, since the same surface-level linguistic pat-
tern (an embedded instruction) can be either benign or malicious depending on context. This motivates a
systematic evaluation of how lightweight text classifiers generalize across task contexts and attack formu-
lations, together with an analysis of the linguistic conditions under which they produce false positives and
false negatives.

2. Motivation

Indirect prompt injection is an active, unresolved vulnerability class in deployed LLM systems. Yi et al. [1]
show that a wide range of production and open-source LLMs are universally vulnerable to such attacks, and
identify two root causes: models’ inability to distinguish informational context from actionable instructions,
and their lack of awareness that instructions embedded in external content should not automatically be exe-
cuted. As LLM-integrated applications (document assistants, coding copilots, autonomous agents) become
more widely deployed, a detector that can reliably flag injected instructions - and that generalizes beyond
the specific attacks it was trained on - has direct real-world security value. Prior detection work has largely
evaluated in-distribution performance or relied on white-box access to model internals (e.g., hidden-state
or gradient-based signals during inference); the extent to which lightweight, model-agnostic text classifiers
hold up specifically under cross-task and cross-attack distribution shift, and how they behave on benign
instruction-like language, is the gap this project targets.

3. Dataset

We will use BIPIA (Benchmark for Indirect Prompt Injection Attacks) [1], the benchmark introduced by
Yi et al. at KDD 2025. BIPIA is constructed along three axes: application task, attack type, and the spe-
cific external content into which an attack is embedded, spanning five task categories - EmailQA, TableQA,
CodeQA, WebQA, and Summarization. Note that BIPIA was originally designed to measure LLM vulner-
ability (attack success rate) rather than as a ready-made binary classification dataset; we will formulate a
binary detection task from BIPIA’s clean and attack-injected contextual examples, while retaining the task
and attack metadata for controlled generalization experiments. This is our own project-level formulation of

1

the benchmark, not its original design intent, and we will need to construct or source a benign-class sample
sufficient for supervised training (see limitations below).

Limitations to note upfront: (i) the public BIPIA release requires cloning the authors’ code repository and
running their data-preparation scripts; for EmailQA, TableQA, and CodeQA the context files are directly
available, but for WebQA and Summarization the context data must be generated via separate third-party
pipelines (NewsQA and XSum respectively) due to licensing restrictions on redistribution. We will budget
the first two weeks of the project for this setup, and may prioritize the three directly-available task types if the
WebQA/Summarization pipelines prove too time-consuming. (ii) Constructing a balanced binary detection
dataset requires a benign class, which BIPIA does not directly provide at scale; a recent independent study
[2] addressed this by generating benign user-intent/content pairs with an LLM. We will adopt a similar
strategy at a scope appropriate for a course project, and will verify class balance and quality manually
before training.

4. Plan

1. Weeks 1–2: Set up the BIPIA repository and data pipeline; construct the binary injected/clean/benign

dataset; verify data quality with manually inspected samples.

2. Weeks 3–5: Implement and train the classical baseline (TF-IDF features with Logistic Regression) and

the modern baseline (a fine-tuned DeBERTa classifier).

3. Weeks 6–7 (Part 2 target, Oct 15): Run in-distribution evaluation and cross-task generalization exper-
iments (train on a subset of task types, test on held-out task types); conduct preliminary quantitative and
qualitative error analysis to identify the dominant failure mode(s) - e.g. cross-task generalization gaps,
sensitivity to attack wording, or false positives on benign instruction-like content.

4. Weeks 8–11: Based on the failure modes identified in the baseline experiments and a targeted review
of recent literature, design and implement a robustness-oriented modeling improvement targeting the
dominant source of error observed in our own experiments.

5. Weeks 12–13 (Part 3 target, Nov 21): Final evaluation across all experimental conditions (in-distribution,
cross-task, benign-vs-malicious discrimination), full error analysis, and a CLI/GUI demonstration inter-
face for the viva.

References

[1] Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. Bench-
marking and Defending Against Indirect Prompt Injection Attacks on Large Language Models. In Pro-
ceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Volume
1 (KDD ’25), pages 1809–1820, Toronto, Canada, August 2025. ACM. https://doi.org/10.
1145/3690624.3709179

[2] Mohammed Alamsabi, Michael Tchuindjang, and Sarfraz Brohi. Embedding-Based Detection of Indi-
rect Prompt Injection Attacks in Large Language Models Using Semantic Context Analysis. Algorithms,
19(1):92, January 2026. https://doi.org/10.3390/a19010092

2

