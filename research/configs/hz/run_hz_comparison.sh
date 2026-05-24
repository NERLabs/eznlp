#!/bin/bash
# HZ 数据集 NER 对比实验启动脚本

echo "========================================="
echo "HZ NER 对比实验: Baseline vs +ExpertDict"
echo "========================================="
echo ""

# 设置 Python 路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 确保使用当前激活的 Python 环境
PYTHON_CMD=${PYTHON_CMD:-python}

# 运行对比实验
$PYTHON_CMD scripts/train_hz_ner_baseline_vs_expert_dict.py \
    --run_both \
    --data_dir data/HZ \
    --expert_dict_path data/HZ/expert_lexicon.txt \
    --save_dir cache/hz_ner_comparison \
    --bert_arch hfl/chinese-macbert-base \
    --bert_drop_rate 0.2 \
    --hid_dim 256 \
    --num_layers 1 \
    --dropout 0.5 \
    --expert_dict_dim 50 \
    --num_epochs 30 \
    --batch_size 16 \
    --lr 2e-3 \
    --finetune_lr 2e-5 \
    --weight_decay 1e-4 \
    --grad_clip 5.0 \
    --disp_every_steps 50 \
    --eval_every_steps 200 \
    --seed 42 \
    "$@"

echo ""
echo "========================================="
echo "实验完成！"
echo "========================================="
