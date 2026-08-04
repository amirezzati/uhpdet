# UHP Detection

**LVLMs have their hallucination pattern in the consistency space**

[PAPER_LINK] · [AUTHORS]

This repository contains the implementation for UHP (Unified Hallucination Pattern)
Detection: a framework that detects hallucinations in Large Vision-Language Models
(LVLMs) by measuring how *consistent* a model's answers stay under image and text
perturbations, rather than by inspecting a single response in isolation. For a given
image/claim pair, the model is queried under a set of image transformations (noise,
blur, style augmentation, ...) and text paraphrases (lexical substitution, syntactic
restructuring, ...). Agreement/disagreement patterns across these perturbed queries are
turned into consistency features, which a downstream classifier uses to predict whether
the original answer was hallucinated.

## Repository structure

```
uhpdet/
├── Benchmarks/                  # Benchmark datasets used for evaluation
│   ├── AMBER_no_questions/      # AMBER benchmark, discriminative "no" questions
│   ├── AMBER_yes_questions/     # AMBER benchmark, discriminative "yes" questions
│   ├── PHD_no_questions/        # PHD benchmark, "no" questions
│   ├── PHD_yes_questions/       # PHD benchmark, "yes" questions
│   └── unified.py               # Normalizes raw benchmark JSON into a unified schema
│
├── Framework/                   # Core pipeline: inference, features, classification
│   ├── inference.py      # Step 1-4: factual-statement generation + perturbed inference
│   ├── image_transformation_func.py   # Image perturbations (noise, blur, style, ...)
│   ├── compute_features.py      # Step 5: turns raw inference results into consistency features
│   ├── classifier.py            # Step 6: trains/evaluates the main hallucination classifier
│   ├── utils/                   # Supporting helper modules/scripts
│   │   ├── internVL.py          # InternVL image preprocessing helper (used by inference.py)
│   │   └── visualize_features.py  # Optional: histogram / joint-PDF plots of features
│   ├── ablation/                # Classifier variants for ablation/generalizability experiments
│   │   ├── Classifier_ablation_sampling.py    # Capped training-set size (sample-size ablation)
│   │   ├── Classifier_feature_ablation.py     # Leave-one-feature-out ablation study
│   │   ├── Classifier_Generalizability.py     # Cross-benchmark generalizability (train on one benchmark, test on another)
│   │   └── Classifier_category.py             # Per-hallucination-category train/test split experiment
│   ├── halclassifier.yml        # Conda env for feature computation / classification / plotting
│   └── results/                 # Output of inference.py (results_full_<benchmark>_<model>.json)
│
└── Results/                      # Generated outputs (features, plots, trained classifier results)
    ├── Features/<model>/<benchmark>/data_features.csv       # Output of compute_features.py
    ├── Features/<model>/<benchmark>/plots/                  # Output of visualize_features.py
    └── Classifier/<model>_<benchmark>/                      # Output of classifier.py (per-model metrics, learning curves, splits)
```

`<model>` is one of `blip`, `llava`, `qwen`, `internvl` (`Results/Features/` currently has
precomputed features for `blip`, `qwen`, and `internvl` across all four benchmark splits).

## Setup

Two conda environments are provided:

- **`uhpdet.yml`** (repo root) — heavier environment with PyTorch/CUDA/`transformers`,
  used to run `inference.py` (LVLM + Phi-2 inference).
- **`Framework/halclassifier.yml`** — lightweight environment (pandas, scikit-learn,
  matplotlib/seaborn, xgboost, lightgbm) for `compute_features.py`, `visualize_features.py`,
  and the `Classifier*.py` scripts.

```bash
conda env create -f uhpdet.yml
conda env create -f Framework/halclassifier.yml
```

`inference.py` loads VLM checkpoints (LLaVA, InstructBLIP, Qwen2.5-VL, InternVL)
from local paths hardcoded near the top of `if __name__ == "__main__"` — point
`local_model_path` at your own Hugging Face cache, or swap back to the commented-out
`model_path` (Hugging Face hub ID) to download on first run.

## Pipeline / How to run

All commands below assume you `cd Framework` first (paths are relative to that directory).

### 1. Generate factual statements + run perturbed inference

```bash
# Step 1: Phi-2 factual-statement generation only (fast, CPU/GPU)
python inference.py --step 1 --input-file ../Benchmarks/AMBER_yes_questions/data_with_outputs_unified.json

# Steps 2-4: baseline answer, image-perturbation, and text-perturbation inference for one VLM
python inference.py --input-file results/results_full_AMBER_yes_questions.json \
    --model-name blip --step 4
```
`--model-name` accepts `llava`, `blip`, `qwen`, or `internvl`. Repeat for each of the
four benchmark splits (`AMBER_no_questions`, `AMBER_yes_questions`, `PHD_no_questions`,
`PHD_yes_questions`).

### 2. Compute consistency features

```bash
python compute_features.py --input-file results/results_full_AMBER_yes_questions_blip.json
# -> ../Results/Features/blip/AMBER_yes_questions/data_features.csv
```

### 3. Train / evaluate the hallucination classifier

```bash
# Automatic path derivation (reads both yes/no splits of a benchmark)
python classifier.py --model blip --benchmark AMBER --type both
# -> ../Results/Classifier/blip_amber_full/
```

The `ablation/` scripts cover related experiments, run as `python ablation/<script>.py ...`
from `Framework/` (see the usage examples in each file's header/footer comments):
- `ablation/Classifier_ablation_sampling.py` / `ablation/Classifier_feature_ablation.py` — same `--file1/--file2` CLI as `classifier.py`.
- `ablation/Classifier_Generalizability.py --train-csv ... --test-csv ... --output-dir ...`
- `ablation/Classifier_category.py --csv1 ... --csv2 ... --test-type ... --output-dir ...`

### 4. (Optional) Visualize features

```bash
python utils/visualize_features.py --model-name blip --dataset AMBER_yes_questions
# -> ../Results/Features/blip/AMBER_yes_questions/plots/{hist,joint_pdf}/
```

## Benchmarks

- **AMBER** — discriminative yes/no questions about object/attribute/state hallucinations.
- **PHD** — discriminative yes/no questions from the PhD hallucination benchmark.

Each is split into `*_yes_questions` (ground-truth answer "yes") and `*_no_questions`
("no") folders; `unified.py` normalizes the raw benchmark JSON (image paths, dropped
fields) into the schema consumed by `inference.py`.

## Citation

```
[CITATION]
```
