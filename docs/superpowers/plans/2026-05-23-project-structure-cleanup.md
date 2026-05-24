# Project Structure Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the repository for paper-writing file search while preserving legacy experiment assets.

**Architecture:** Keep `eznlp/` and `tests/` stable. Move numbered legacy experiment directories into descriptive `research/` folders, move data snippets to `datasets/`, move root paper assets to `docs/paper/assets/`, and update path references mechanically.

**Tech Stack:** Python package repository, shell scripts, Taskfile YAML, Markdown documentation.

---

### Task 1: Move Directories to Searchable Names

**Files:**
- Move: `_1CONFIG/` to `research/configs/`
- Move: `_2DATA/` to `datasets/`
- Move: `bmes/` to `datasets/bmes/`
- Move: `_3DATA_PROCESS/` to `research/data_processing/`
- Move: `_4MODELS/` to `research/model_variants/`
- Move: `_5TRAIN/` to `research/training/`
- Move: `_6EVALUATE/` to `research/evaluation/`
- Move: `_7PIPELINE/` to `research/pipelines/`
- Move: `_8TOOL/` to `research/tools/`
- Move: `_9LOGS/logs/` to `logs/`
- Move: `projects/` to `references/external_projects/`
- Move: root diagrams and PDF to `docs/paper/assets/`

- [ ] Create target directories with `mkdir -p`.
- [ ] Move tracked files with `git mv` where possible.
- [ ] Move any remaining ignored local files with `mv`.
- [ ] Run `git status --short` and verify no unexpected deletions.

### Task 2: Update Path References

**Files:**
- Modify: `Taskfile.yml`
- Modify: `scripts/Taskfile.yml`
- Modify: moved files under `research/`
- Modify: tracked docs and config files containing old paths

- [ ] Replace `_1CONFIG` with `research/configs`.
- [ ] Replace `_2DATA` with `datasets`.
- [ ] Replace `_3DATA_PROCESS` with `research/data_processing`.
- [ ] Replace `_4MODELS` with `research/model_variants`.
- [ ] Replace `_5TRAIN` with `research/training`.
- [ ] Replace `_6EVALUATE` with `research/evaluation`.
- [ ] Replace `_7PIPELINE` with `research/pipelines`.
- [ ] Replace `_8TOOL` with `research/tools`.
- [ ] Replace `_9LOGS/logs` and `_9LOGS` with `logs`.
- [ ] Replace root `bmes/` paths with `datasets/bmes/`.
- [ ] Replace `projects/` references with `references/external_projects/`.

### Task 3: Add Paper-Search Entry Documentation

**Files:**
- Create: `README.md`
- Create: `docs/PROJECT_STRUCTURE.md`

- [ ] Add a concise root index focused on paper writing.
- [ ] Add a directory map that names where paper files, data, experiments, and external projects live.
- [ ] Ensure `pyproject.toml` can read `README.md`.

### Task 4: Verify Structure

**Files:**
- Read-only verification across repository.

- [ ] Search for old directory names and classify any remaining matches as historical text or stale paths.
- [ ] Run `python -m compileall eznlp research tests`.
- [ ] Run a lightweight pytest subset if dependencies are available.
- [ ] Report final structure and any verification limitations.
