#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/home/shiwenlong/NERlabs/eznlp"
cd "$PROJECT_ROOT"

LOG_FILE="_9LOGS/logs/per_type_adaptive_resume.log"
mkdir -p "_9LOGS/logs"
rm -f "$LOG_FILE"
touch "$LOG_FILE"

# Capture all stdout/stderr into the progress log.
exec >>"$LOG_FILE" 2>&1

echo "============================================================"
echo "[START] $(date '+%F %T') run_per_type_adaptive_resume.sh"
echo "============================================================"

BERT_ARCH="hfl/chinese-macbert-base"
SEEDS_BOSON=(44)
SEEDS_CLUE=(43 44)
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
  # training.log is nested under expert_boundary_*/training.log
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
    echo "[SKIP] ${dataset} seed=${seed} already finished: $save_dir"
    return 0
  fi

  echo "[RUN ] ${dataset} seed=${seed} => $save_dir"
  conda run -n eznlp11 python "_5TRAIN/train_general_expert_boundary.py" \
    --train_file "$train_file" \
    --dev_file "$dev_file" \
    --test_file "$test_file" \
    --bert_arch "$BERT_ARCH" \
    --save_dir "$save_dir" \
    --expert_dict_path "$lexicon" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}"

  echo "[DONE] ${dataset} seed=${seed}"
}

run_redjujube() {
  local seed="$1"
  local save_dir="experiments/EXP-010-optimization/results_public/redjujube_bs_dict_focal_per_type_adaptive/seed_${seed}_full30"
  if has_finished "$save_dir"; then
    echo "[SKIP] redjujube seed=${seed} full30 already finished: $save_dir"
    return 0
  fi

  echo "[RUN ] redjujube seed=${seed} full30 => $save_dir"
  conda run -n eznlp11 python "_5TRAIN/train_redjujube_expert_boundary.py" \
    --data_dir "_2DATA/RedJujube" \
    --bert_arch "$BERT_ARCH" \
    --save_dir "$save_dir" \
    --expert_dict_path "$LEX_REDJUJUBE" \
    --seed "$seed" \
    "${COMMON_ARGS[@]}"

  echo "[DONE] redjujube seed=${seed} full30"
}

# Data paths
BOSON_TRAIN="_2DATA/boson/boson.train.bmes"
BOSON_DEV="_2DATA/boson/boson.dev.bmes"
BOSON_TEST="_2DATA/boson/boson.test.bmes"

CLUE_TRAIN="_2DATA/clue/train.char.bmes"
CLUE_DEV="_2DATA/clue/dev.char.bmes"
CLUE_TEST="_2DATA/clue/test.char.bmes"

for seed in "${SEEDS_BOSON[@]}"; do
  run_general "boson" "$BOSON_TRAIN" "$BOSON_DEV" "$BOSON_TEST" "$LEX_BOSON" "$seed"
done

for seed in "${SEEDS_CLUE[@]}"; do
  run_general "clue" "$CLUE_TRAIN" "$CLUE_DEV" "$CLUE_TEST" "$LEX_CLUE" "$seed"
done

for seed in "${SEEDS_REDJUJUBE[@]}"; do
  run_redjujube "$seed"
done

echo "============================================================"
echo "[DONE ] $(date '+%F %T') per-type adaptive resume finished"
echo "Log: $LOG_FILE"
echo "============================================================"

