# Current RJND Baselines Experiment Branch Goal Plan

> **For Goal Mode workers:** Execute this plan task-by-task. Update checkboxes as work completes, keep evidence paths beside every result, and stop before adding checkpoints, caches, or full training logs to git.

**Goal:** 在 `experiment/current-rjnd-baselines` 分支上补齐当前 RJND/RedJujube 投稿口径的可追溯基线结果，并产出可由论文端 `master` 按文件挑选的最小证据集。

**Current Pull Snapshot:** 本计划基于 2026-05-24 拉取后的实验分支状态编写。`experiment/current-rjnd-baselines` 已跟踪 `origin/experiment/current-rjnd-baselines`，`git pull --ff-only origin experiment/current-rjnd-baselines` 返回已经是最新的，当前提交为 `ecbb9f8 docs: add paper experiment branch workflow`。`origin/master` 另有更新文档提交，但不属于本实验分支；除非用户明确要求，不要把 `master` 合入实验分支。

**Architecture:** 实验端只负责生成、复评和登记结果。论文端后续从实验分支挑选 `docs/paper/*` 和小型结果文件进入 `master`，不整分支合并。所有进入正文主对比表的结果必须使用同一 RJND/RedJujube 当前数据划分、`seed=42`、test split 实体级严格 P/R/F1。

**Tech Stack:** Git experiment branch workflow, Markdown evidence registry, Bash/tmux, Conda `eznlp11`, PyTorch training/evaluation scripts, `Taskfile.yml` where available.

**Primary Source Docs:**
- `docs/paper/branch_workflow_for_paper_and_experiments.md`
- `docs/paper/current_rjnd_experiment_requirements.md`
- `docs/paper/needed_experiment_results.md`
- `docs/paper/paper_result_registry.md`
- `docs/paper/plans/2026-05-24-needed-experiments-execution.md`

---

### Task 1: Confirm Branch Hygiene

**Files:**
- Read-only: git state
- Do not modify: unrelated untracked local data/tool directories

- [x] Run `git status --short --branch` and confirm current branch is `experiment/current-rjnd-baselines` (`experiment/current-rjnd-baselines...origin/experiment/current-rjnd-baselines`).
- [x] Run `git pull --ff-only origin experiment/current-rjnd-baselines` before starting new work (`已经是最新的`).
- [x] Confirm no `master` merge is pending and no unrelated tracked files are modified.
- [x] Treat local untracked directories such as `_2DATA/`, `_8TOOL/`, `projects/`, and untracked docs as local workspace assets unless the user explicitly asks to add them.

### Task 2: Lock The RJND Result口径

**Files:**
- Read: `docs/paper/current_rjnd_experiment_requirements.md`
- Read: `docs/paper/needed_experiment_results.md`
- Read: `docs/paper/paper_result_registry.md`

- [x] Confirm every candidate result uses current RJND/RedJujube data, not old RedJujube, HZ, MSRA, Boson, CLUENER, WeiboNER, or ResumeNER.
- [x] Confirm `seed=42` for any result intended for the current正文主表.
- [x] Confirm evaluation is test split entity-level strict P/R/F1 in percent.
- [x] Record `model_name`, `seed`, `dataset_split`, `precision`, `recall`, `f1`, `result_path`, `config_path`, `eval_script`, `bert_backbone`, `lexicon`, `data_process`, and `note` in `docs/paper/needed_experiment_results.md` and `docs/paper/paper_result_registry.md`.
- [x] Reject results that only report dev F1, use different labels, use different evaluation scripts, or mix old dataset paths without an explicit compatibility note.

### Task 3: Verify Existing Completed Evidence

**Files:**
- Verify: `experiments/EXP-010-optimization/results_needed_20260524/`
- Verify: `docs/paper/plans/2026-05-24-needed-experiments-execution.md`
- Modify if stale: `docs/paper/needed_experiment_results.md`
- Modify if stale: `docs/paper/paper_result_registry.md`

- [x] Check that `EDBP min_freq=1`, `min_freq=3`, and `min_freq=4` results exist under `results_needed_20260524`.
- [x] Verify each result directory has `results.json`, `training.log`, and `auto_lexicon.txt` where applicable.
- [x] Confirm lexicon sizes match the registry: `5317`, `1087`, and `786`.
- [x] Re-run P/R/F1复评 only if registry values cannot be traced to a saved model and `auto_lexicon.txt` (复评 confirmed Micro P/R/F1 for min_freq 1/3/4).
- [x] Keep `min_freq=2` as the registered主模型 result unless the user asks for full table recomputation.

### Task 4: Search Before Rerunning Missing Priority Baselines

**Files:**
- Read/search: `experiments/`
- Read/search: `research/`
- Read/search: `_1CONFIG/`, `_5TRAIN/`, `_6EVALUATE/` if present locally
- Modify: `docs/paper/needed_experiment_results.md`

- [x] Search existing outputs for `Boundary Smoothing` on current RJND/RedJujube `seed=42`.
- [x] Search existing outputs for `BERT+SoftLexicon` or `SoftLexicon+BERT` on current RJND/RedJujube `seed=42`.
- [x] Search existing outputs for `BERT-MRC` on current RJND/RedJujube `seed=42`.
- [x] Search existing outputs for `BERT-MRC+DSC` on current RJND/RedJujube `seed=42`.
- [x] Search existing outputs for `RA_NER` or `AdaSeq BERT-CRF` on current RJND/RedJujube `seed=42`.
- [x] For every located result, inspect config/log paths before registration; do not infer口径 from directory names alone.

### Task 5: Run Only The Highest Value Missing Experiments

**Files:**
- Execute: relevant training scripts under `research/training/` or legacy training paths
- Output: `experiments/EXP-010-optimization/results_needed_YYYYMMDD/<model_slug>/`
- Log: `_9LOGS/logs/` or local result `training.log`, but commit only compact summaries

- [x] If `Boundary Smoothing` is absent under current口径, adapt the closest existing script/config and run it first (`BS_nodict_seed42_current/bert_bs_pure_20260524-164044`).
- [x] If `BERT+SoftLexicon` is absent or uses a non-RJND lexicon, run or clearly mark the lexicon-source limitation.
- [ ] If MRC baselines are required, generate MRC-format data from current RJND before training and commit the conversion script/config, not generated bulky caches.
- [x] Use tmux for long-running jobs and name sessions with model, seed, and date (`rjnd-bs-nodict-20260524`).
- [x] Save each run under a unique dated output directory to avoid overwriting older evidence.
- [x] After each run, immediately复评 or parse test P/R/F1 and update the evidence table before starting lower-priority jobs.

### Task 6: Preserve Known Blockers For Lattice Baselines

**Files:**
- Read: `projects/LatticeLSTM-master/` if present
- Read: `projects/NFLAT4CNER-main/` if present
- Modify: `docs/paper/needed_experiment_results.md`

- [x] Do not start `LatticeLSTM` until Python 2/PyTorch 0.3 compatibility or a Python 3 migration path is confirmed.
- [x] Do not start `NFLAT` until RedJujube data adapters, seed control, Python 3.7 dependencies, and GPU availability are confirmed.
- [x] Record blocker details as evidence so the论文端 can justify excluding these models from the current正文主表.

### Task 7: Register Results In Paper Evidence Docs

**Files:**
- Modify: `docs/paper/needed_experiment_results.md`
- Modify: `docs/paper/paper_result_registry.md`
- Modify if task priorities changed: `docs/paper/current_rjnd_experiment_requirements.md`

- [x] Add every completed result to `needed_experiment_results.md` with full trace fields.
- [x] Add only adopted or decision-relevant results to `paper_result_registry.md`.
- [x] For each table row, include dataset path/split, seed, backbone, lexicon source and size, result path, and config path.
- [x] Explicitly mark results that are useful for screening but not eligible for the current正文主表.
- [x] Recompute improvement deltas if any adopted F1 value changes (Boundary Smoothing candidate delta recorded as +1.68; no existing adopted F1 changed).

### Task 8: Verify Before Commit

**Files:**
- Verify: changed docs and small result artifacts

- [x] Run `git diff --check`.
- [x] Run `git status --short` and inspect every staged path.
- [x] Check staged files for large artifacts before commit with `git diff --cached --stat`.
- [x] Do not stage checkpoints, `*.pt`, `*.pth`, `*.ckpt`, large caches, full debug logs, or generated bulky datasets.
- [x] If manuscript files changed, run the relevant numeric consistency validator before reporting completion (not needed; manuscript files were not edited).

### Task 9: Commit And Push Experiment Branch

**Files:**
- Commit: docs and small evidence files only

- [ ] Stage only result registries, compact configs, compact result JSON/CSV, and short summaries.
- [ ] Commit with a message like `experiments: register current RJND baseline results`.
- [ ] Push with `git push origin experiment/current-rjnd-baselines`.
- [ ] If push is rejected, run `git pull --rebase origin experiment/current-rjnd-baselines`, resolve only relevant conflicts, then push again.

### Task 10: Prepare Paper-Side Handoff

**Files:**
- Read: `docs/paper/branch_workflow_for_paper_and_experiments.md`

- [x] Summarize completed models, F1 values, and result paths in `docs/paper/current_rjnd_baselines_handoff_2026-05-24.md`.
- [x] List excluded models and exact blocker reasons in `docs/paper/current_rjnd_baselines_handoff_2026-05-24.md`.
- [x] Provide the minimal file list for paper-side checkout from `origin/experiment/current-rjnd-baselines`.
- [x] Remind the paper-side worker to select files instead of running `git merge experiment/current-rjnd-baselines`.

## Definition Of Done

- [ ] Experiment branch is up to date and pushed.
- [ ] Every newly adopted result has reproducible path, config, evaluation script, and口径 note.
- [ ] No bulky training artifacts are staged or committed.
- [ ] `docs/paper/needed_experiment_results.md` and `docs/paper/paper_result_registry.md` agree on adopted values.
- [ ] The handoff summary is sufficient for `master` to pick files without merging the whole experiment branch.
