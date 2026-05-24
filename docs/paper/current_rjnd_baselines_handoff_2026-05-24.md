# Current RJND Baselines Handoff 2026-05-24

This handoff is for the paper-side `master` worker. Select files from
`origin/experiment/current-rjnd-baselines`; do not merge the whole experiment branch.

## Completed Current-Path Evidence

| Model | Seed | Dataset | P/% | R/% | F1/% | Result path | Status |
|---|---:|---|---:|---:|---:|---|---|
| EDBP min_freq=1 | 42 | `datasets/raw/RedJujube` | 86.94 | 81.92 | 84.36 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/results.json` | Lexicon-threshold comparison |
| EDBP min_freq=3 | 42 | `datasets/raw/RedJujube` | 88.56 | 86.49 | 87.51 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/results.json` | Lexicon-threshold comparison |
| EDBP min_freq=4 | 42 | `datasets/raw/RedJujube` | 87.79 | 86.12 | 86.95 | `experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/results.json` | Lexicon-threshold comparison |
| Boundary Smoothing | 42 | `datasets/raw/RedJujube` | 87.36 | 85.61 | 86.48 | `experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/results.json` | Table 3 candidate |

The current registered EDBP main result remains the existing `min_freq=2` result
with F1 88.16 unless the paper side decides to recompute the full table.
If `Boundary Smoothing` is added to Table 3, the EDBP improvement over it is
88.16 - 86.48 = +1.68 percentage points.

## Excluded Or Limited Results

- `SoftLexicon` seed 42 exists at
  `experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline/seed_42/softlexicon_20260421-212809/results.json`,
  but it uses `_2DATA/RedJujube` and `data/HZ/softlexicon_train.txt`; keep the
  lexicon-source limitation in any table note, or rerun with a RJND training lexicon.
- `BERT-MRC` / `BERT-MRC+DSC` has no usable result. `_9LOGS/dice_loss_redjujube_train.log`
  shows training stopped with `ValueError` in `_5TRAIN/tasks/mrc_ner/train.py`.
- `RA_NER / AdaSeq BERT-CRF` has no usable result. `_9LOGS/adaseq_redjujube_train.log`
  shows dataset generation failed on an invalid BMES label sequence.
- `LatticeLSTM` remains blocked by Python 2/PyTorch 0.3 style code and fixed
  `seed_num=100`.
- `NFLAT` remains blocked by missing RedJujube adapter, fixed seed 2022, old
  Python 3.7 dependency assumptions, and GPU/env readiness checks.

## Minimal Checkout List

```bash
git checkout origin/experiment/current-rjnd-baselines -- \
  docs/paper/current_rjnd_experiment_requirements.md \
  docs/paper/needed_experiment_results.md \
  docs/paper/paper_result_registry.md \
  docs/paper/plans/2026-05-24-needed-experiments-execution.md \
  docs/paper/current_rjnd_baselines_handoff_2026-05-24.md \
  experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/auto_lexicon.txt \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/auto_lexicon.txt \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/auto_lexicon.txt
```

Do not checkout checkpoints, `*.pth`, tensorboard files, caches, or full training logs.
