#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

LOG_FILE="_9LOGS/logs/per_type_adaptive_all.log"
mkdir -p "_9LOGS/logs"
touch "$LOG_FILE"

echo "============================================================" | tee -a "$LOG_FILE"
echo "[START] $(date '+%F %T') run_per_type_adaptive_all.sh" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

BERT_ARCH="hfl/chinese-macbert-base"
SEEDS_PUBLIC=(43 44)
SEEDS_REDJUJUBE=(42 43 44)

COMMON_ARGS=(--bert_arch "$BERT_ARCH" --num_epochs 30 --batch_size 16 --sb_epsilon 0.1 --sb_size 2 --no_fgm --no_ema --fl_gamma 2.0)

LEX_BOSON="experiments/EXP-011-lexicon_strategy/analysis/per_type_lexicon_boson.txt"
LEX_CLUE="experiments/EXP-011-lexicon_strategy/analysis/per_type_lexicon_clue.txt"
LEX_REDJUJUBE="experiments/EXP-011-lexicon_strategy/analysis/per_type_lexicon_redjujube.txt"

has_finished() {
  local seed_dir="$1"
  if [ ! -d "$seed_dir" ]; then
    return 1
  fi
  if rg -n "最终测试 F1" "$seed_dir"/expert_boundary_*/training.log >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

run_general() {
  local dataset="$1"
  local train_file="$2"
  local dev_file="$3"
  local test_file="$4"
  local lexicon="$5"
  local seed="$6"

  local save_dir="experiments/EXP-010-optimization/results_public/${dataset}_bs_dict_focal_per_type_adaptive/seed_${seed}"
  if has_finished "$save_dir"; then
    echo "[SKIP] ${dataset} seed=${seed} already finished" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "[RUN ] ${dataset} seed=${seed}" | tee -a "$LOG_FILE"
  conda run -n eznlp11 python "_5TRAIN/train_general_expert_boundary.py" \
    --train_file "$train_file" \
    --dev_file "$dev_file" \
    --test_file "$test_file" \
    --save_dir "$save_dir" \
    --expert_dict_path "$lexicon" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  echo "[DONE] ${dataset} seed=${seed}" | tee -a "$LOG_FILE"
}

run_redjujube() {
  local seed="$1"
  local save_dir="experiments/EXP-010-optimization/results_public/redjujube_bs_dict_focal_per_type_adaptive/seed_${seed}"
  if has_finished "$save_dir"; then
    echo "[SKIP] redjujube seed=${seed} already finished" | tee -a "$LOG_FILE"
    return 0
  fi

  echo "[RUN ] redjujube seed=${seed}" | tee -a "$LOG_FILE"
  conda run -n eznlp11 python "_5TRAIN/train_redjujube_expert_boundary.py" \
    --data_dir "_2DATA/RedJujube" \
    --save_dir "$save_dir" \
    --expert_dict_path "$LEX_REDJUJUBE" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  echo "[DONE] redjujube seed=${seed}" | tee -a "$LOG_FILE"
}

# ---------- boson / clue: seeds 43,44 ----------
for seed in "${SEEDS_PUBLIC[@]}"; do
  run_general "boson" "_2DATA/boson/boson.train.bmes" "_2DATA/boson/boson.dev.bmes" "_2DATA/boson/boson.test.bmes" "$LEX_BOSON" "$seed"
done

for seed in "${SEEDS_PUBLIC[@]}"; do
  run_general "clue" "_2DATA/clue/train.char.bmes" "_2DATA/clue/dev.char.bmes" "_2DATA/clue/test.char.bmes" "$LEX_CLUE" "$seed"
done

# ---------- redjujube: seeds 42,43,44 ----------
for seed in "${SEEDS_REDJUJUBE[@]}"; do
  run_redjujube "$seed"
done

echo "============================================================" | tee -a "$LOG_FILE"
echo "[DONE ] $(date '+%F %T') all per-type adaptive runs finished" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

