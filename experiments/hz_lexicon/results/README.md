# eznlp project index

This repository is currently used mainly for paper writing and source lookup.
The core library remains in `eznlp/`; paper drafts, submission files, figures,
and validation notes live under `docs/paper/`.

## Common entry points

- `docs/paper/`: paper drafts, submission package, figures, validation reports.
- `docs/PROJECT_STRUCTURE.md`: directory map for finding files quickly.
- `eznlp/`: reusable NLP package code.
- `research/`: legacy experiment code, configs, training scripts, evaluation
  scripts, and tooling kept for evidence lookup.
- `datasets/`: small tracked demo datasets and project-specific data snippets.
- `experiments/`: historical experiment records and result summaries.
- `references/external_projects/`: copied external baseline projects and
  reference implementations.
- `tests/`: package tests.

## Paper workflow

For writing, start in `docs/paper/`:

- Main manuscript and submission files are in `docs/paper/submission_package/`.
- Figures are in `docs/paper/figures/` and `docs/paper/figures_png/`.
- Official journal templates and checklists are in `docs/paper/official_docs/`.
- Reproducibility and consistency checks are in the dated validation files.

Experiment code is preserved under `research/` for lookup, but the desktop
workflow should not depend on running long training jobs.
