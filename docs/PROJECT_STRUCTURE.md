# Project structure

This layout prioritizes quick file search for paper writing while preserving
legacy experiment assets.

## Top-level map

- `README.md`: short entry point for the repository.
- `pyproject.toml`: Python package metadata for `eznlp`.
- `eznlp/`: core package code.
- `tests/`: package tests.
- `docs/`: documentation and paper assets.
- `docs/paper/`: active paper workspace.
- `research/`: legacy experiment code and scripts.
- `datasets/`: tracked small datasets and demo data.
- `experiments/`: historical experiment result folders.
- `logs/`: retained lightweight log and process records.
- `references/external_projects/`: copied third-party projects used as
  baselines or implementation references.
- `third_party/`: vendored dependency snippets used by package code.

## Research code map

- `research/configs/`: old run scripts and experiment configs.
- `research/data_processing/`: dataset conversion, lexicon extraction, and
  augmentation scripts.
- `research/model_variants/`: custom model blocks and experimental model
  builders.
- `research/training/`: training entry points and data loaders.
- `research/evaluation/`: result collection, analysis, debugging, and ensemble
  scripts.
- `research/pipelines/`: pipeline runners and config templates.
- `research/tools/`: monitoring, visualization, vector, and helper utilities.

## Paper workspace map

- `docs/paper/submission_package/`: files intended for submission handoff.
- `docs/paper/figures/`: editable vector figures.
- `docs/paper/figures_png/`: exported figure images.
- `docs/paper/official_docs/`: journal template and official checklists.
- `docs/paper/tools/`: paper validation and packaging scripts.
- `docs/paper/assets/`: miscellaneous design diagrams and source reference
  files moved out of the repository root.

## Search tips

- Search paper text and validation evidence in `docs/paper/`.
- Search training or evaluation code in `research/`.
- Search data examples in `datasets/`.
- Search copied baseline projects in `references/external_projects/`.
