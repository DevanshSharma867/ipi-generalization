# CS F429 NLP — Indirect Prompt Injection Detection

**Phase 2 Submission | October 12, 2026**

> **Research Question:** Can lightweight, model-agnostic text classifiers detect indirect prompt injection reliably under cross-task distribution shift, and can they distinguish malicious injections from benign imperative language?

---

## Quick Start (For Demo)

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Run full demo (generates data + trains TF-IDF + loads DeBERTa)
python3 demo.py --mode full --quick

# 3. Or run step by step:
python3 src/data_pipeline.py                  # Build leakage-free dataset from BIPIA
python3 run_experiments.py                    # TF-IDF experiments, LOTO, ablations -> results/
python3 demo.py --mode stats                  # Show dataset stats
python3 demo.py --mode train --model tfidf_lr # Train classical baseline
python3 demo.py --mode compare                # Show results
python3 demo.py --mode predict                # Live prediction
```

## Using Real BIPIA Data

```bash
# Clone BIPIA
git clone https://github.com/microsoft/BIPIA data/raw/BIPIA

# Run pipeline with real data
python3 src/data_pipeline.py --bipia_dir data/raw/BIPIA --output_dir data/processed
```

---

## Project Structure

```
nlp/
├── demo.py                    # Main demo entry point
├── run_experiments.py         # TF-IDF + LR benchmarks, leave-one-task-out, ablations
├── submit_train.sh            # Slurm script for multi-seed / LOTO DeBERTa runs
├── report_phase2.tex / .pdf   # Phase 2 report
├── requirements.txt
├── docs/
│   ├── proposal.md            # Phase 1 proposal
│   ├── project_rubric.md      # Course project description and rubric
│   ├── architecture_spec.md   # Project spec and technical architecture
│   └── architecture_diagram.svg
├── src/
│   ├── data_pipeline.py       # BIPIA loader, doc-level splits, held-out attacks
│   ├── preprocess.py          # Text cleaning + tokenization
│   ├── evaluate.py            # Metrics + error analysis + plots
│   └── models/
│       ├── tfidf_lr.py        # Classical baseline
│       └── deberta.py         # Modern baseline (fine-tuned DeBERTa)
├── data/
│   ├── benign/                # Benign instruction templates
│   ├── processed/             # train.csv, val.csv, test.csv, dataset.csv
│   └── raw/                   # BIPIA repo (clone here, not tracked)
├── demo_examples/             # Sample inputs for live prediction
├── checkpoints/               # Saved TF-IDF model (DeBERTa checkpoints not tracked)
└── results/                   # Metrics JSON, predictions, ROC + confusion matrices
```

---

## Models

| Model | Type | F1 Macro | ROC-AUC |
|-------|------|----------|---------|
| TF-IDF + LR | Classical | 0.771 | 0.869 |
| DeBERTa-v3-base | Modern (Transformer) | pending | pending |

---

## Phase 2 Rubric Compliance

| Requirement | Where |
|-------------|-------|
| Literature survey | `report_phase2.tex` Section 2 |
| Dataset & preprocessing | `report_phase2.tex` Section 3, `src/data_pipeline.py` |
| Metrics identification | `report_phase2.tex` Section 4.3 |
| Model details + hyperparams | `report_phase2.tex` Section 4.1-4.2 |
| Preliminary results | `results/` + `demo.py --mode compare` |
| Error analysis | `report_phase2.tex` Section 5.3 + `demo.py --mode errors` |
| **Demonstration (CLI)** | **`demo.py` — training shown live** |

---

## References

1. Yi et al., KDD 2025 — BIPIA benchmark
2. Alamsabi et al., Algorithms 2026 — Embedding-based detection
3. Greshake et al., AISec 2023 — Real-world IPI attacks
4. Perez & Ribeiro, arXiv 2022 — Prompt injection taxonomy
5. He et al., ICLR 2021 — DeBERTa architecture
