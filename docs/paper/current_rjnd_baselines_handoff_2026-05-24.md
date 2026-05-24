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
| SoftLexicon-TrainLex | 42 | `datasets/raw/RedJujube` | 84.42 | 86.71 | 85.55 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_trainlex_seed42_current/softlexicon_trainlex_20260524-171342/results.json` | Current-path SoftLexicon candidate; token matching from `datasets/raw/RedJujube/softlexicon_train.txt`, embeddings from Chinese 50d vectors |
| SoftLexicon-External | 42 | `datasets/raw/RedJujube` | 84.32 | 85.65 | 84.98 | `experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_external_seed42_current/softlexicon_20260524-173106/results.json` | Current-path SoftLexicon candidate; token matching from `assets/vectors/ctb.50d.vec` |
| AdaSeq BERT-CRF | 42 | `datasets/raw/RedJujube` via BIO conversion | 84.42 | 85.90 | 85.16 | `experiments/EXP-010-optimization/results_needed_20260524/AdaSeq_bert_crf_seed42_current/metrics_summary.json` | Completed after BMES-to-BIO conversion |
| BERT-MRC+DSC | 42 | current RedJujube MRC data | 83.06 | 77.77 | 80.33 | `experiments/EXP-010-optimization/results_needed_20260524/MRC_DSC_current_redjujube_status_20260524.json` | Completed in external `dice_loss_for_NLP`; Dice loss, OHEM disabled |

The current registered EDBP main result remains the existing `min_freq=2` result
with F1 88.16 unless the paper side decides to recompute the full table.
If `Boundary Smoothing` is added to Table 3, the EDBP improvement over it is
88.16 - 86.48 = +1.68 percentage points.

## Excluded Or Limited Results

- The old `SoftLexicon` seed 42 result at
  `experiments/EXP-010-optimization/results_newdata/SoftLexicon_baseline/seed_42/softlexicon_20260421-212809/results.json`
  is superseded for current-path evidence by the two 2026-05-24 runs above.
- `BERT-MRC+DSC` is no longer blocked by the old `span_loss_candidates`
  `ValueError`: `/home/shiwenlong/NERlabs/dice_loss_for_NLP/_5TRAIN/tasks/mrc_ner/train.py`
  was patched to use `gold_pred` by default and device-aware random matrices.
  Full current RedJujube Dice/DSC training completed in tmux session
  `rjnd-mrc-dsc-20260524` with `dice_ohem=0`, `train_batch_size=4`, and output
  directory `/home/shiwenlong/NERlabs/dice_loss_for_NLP/_9LOG/mrc_ner/redjujube_current_dice_noohem_bs4_seed42_20260524`.
  `eval_result_log.txt` reports test P/R/F1=83.06/77.77/80.33 and best dev
  F1=80.19 at epoch 8. The external checkpoint is not part of this checkout list.
- `BERT-MRC+DSC` with `dice_ohem=0.3` still fails on the current server:
  first with PyTorch `nonzero` INT_MAX in Dice OHEM, then with CUDA OOM at
  `train_batch_size=10`. The completed configuration disables OHEM and reduces
  batch size while preserving Dice/DSC.
- `LatticeLSTM` source is available under `references/external_projects/LatticeLSTM-master`,
  but it remains Python 2/PyTorch 0.3-style code; `/usr/bin/python2` has no
  `torch`. It needs a Python 3/PyTorch migration or a restored legacy env before
  a fair current-path run.
- `NFLAT` source is available under `references/external_projects/NFLAT4CNER-main`.
  It has been minimally adapted for `redjujube`, `seed`, `n_epochs`,
  `refresh_data`, `smoke_samples`, CPU device selection, and current vector paths;
  `flat37` now imports FastNLP after installing `prettytable`. A 16-sample CPU
  smoke run completed with exit 0, covering data loading, lexicon equipment,
  model construction, training, and dev/test callbacks. This smoke result is not
  a paper metric. A full run is still pending because the GPU remains busy with
  another resident Python process after MRC/DSC completed.

## Minimal Checkout List

```bash
git checkout origin/experiment/current-rjnd-baselines -- \
  docs/paper/current_rjnd_experiment_requirements.md \
  docs/paper/needed_experiment_results.md \
  docs/paper/paper_result_registry.md \
  docs/paper/plans/2026-05-24-needed-experiments-execution.md \
  docs/paper/current_rjnd_baselines_handoff_2026-05-24.md \
  references/external_projects/NFLAT4CNER-main/main.py \
  references/external_projects/NFLAT4CNER-main/utils/load_data.py \
  references/external_projects/NFLAT4CNER-main/utils/paths.py \
  experiments/EXP-010-optimization/results_needed_20260524/BS_nodict_seed42_current/bert_bs_pure_20260524-164044/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq1_seed42_current/expert_boundary_20260524-124317/auto_lexicon.txt \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq3_seed42_current/expert_boundary_20260524-125722/auto_lexicon.txt \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/Q_bs_focal_minfreq4_seed42_current/expert_boundary_20260524-132104/auto_lexicon.txt \
  experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_trainlex_seed42_current/softlexicon_trainlex_20260524-171342/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/SoftLexicon_external_seed42_current/softlexicon_20260524-173106/results.json \
  experiments/EXP-010-optimization/results_needed_20260524/AdaSeq_bert_crf_seed42_current/metrics_summary.json \
  experiments/EXP-010-optimization/results_needed_20260524/MRC_DSC_current_redjujube_status_20260524.json \
  experiments/EXP-010-optimization/results_needed_20260524/NFLAT_current_redjujube_status_20260524.json
```

Do not checkout checkpoints, `*.pth`, tensorboard files, caches, or full training logs.
