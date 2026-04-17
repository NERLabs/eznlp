#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-011: 模型真实预测的类别/长度贡献对比（redjujube）

对指定的两个模型目录（通常是 min_freq 不同的 BS+Dict+Focal 模型），
在 redjujube test split 上计算：
  1) 各实体类别 P/R/F1（基于 precision_recall_f1_report）
  2) 各长度分桶 P/R/F1（按 token span 长度分桶，bucket: 1字/2字/3字/4-5字/6字+）

目的：
  用模型层面的错误分解解释“min_freq 为什么会涨、长实体是否是主要得失点”。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure repo root on sys.path when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TRAIN_ROOT = _REPO_ROOT / "_5TRAIN"
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

import torch

from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.token import LexiconTokenizer
from eznlp.training import Trainer

# Reuse training utilities for consistent data processing
from train_redjujube_expert_boundary import load_redjujube_data, truncate_long_sequences


def get_entity_length(span: Tuple) -> int:
    """span: (type,start,end) or (something,type,start,end)"""
    if len(span) == 3:
        _typ, start, end = span
    elif len(span) == 4:
        _a, _typ, start, end = span
    else:
        return 0
    return end - start


def get_length_bucket(length: int) -> str:
    if length == 1:
        return "1字"
    if length == 2:
        return "2字"
    if length == 3:
        return "3字"
    if 4 <= length <= 5:
        return "4-5字"
    return "6字+"


def load_config_and_model(save_dir: str, device) -> Tuple[object, torch.nn.Module]:
    files = os.listdir(save_dir)
    config_files = [f for f in files if f.endswith("-config.pth")]
    if not config_files:
        raise FileNotFoundError(f"未找到配置文件 (*-config.pth): {save_dir}")
    config_file = sorted(config_files)[0]
    model_file = config_file.replace("-config.pth", ".pth")
    if model_file not in files:
        # Sometimes model file naming differs; try find any .pth besides config
        pth_files = [f for f in files if f.endswith(".pth") and not f.endswith("-config.pth")]
        if not pth_files:
            raise FileNotFoundError(f"未找到模型文件: {save_dir}")
        model_file = pth_files[0]

    config_path = os.path.join(save_dir, config_file)
    model_path = os.path.join(save_dir, model_file)

    config = torch.load(config_path, map_location=device, weights_only=False)
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    return config, model


def find_auto_lexicon(save_dir: str) -> Optional[str]:
    p = os.path.join(save_dir, "auto_lexicon.txt")
    return p if os.path.exists(p) else None


def evaluate_single(save_dir: str, data_dir: str, device: torch.device, max_batch_size: int = 16) -> Dict[str, object]:
    results = {}
    # Load train results for reporting (optional)
    results_path = os.path.join(save_dir, "results.json")
    if os.path.exists(results_path):
        train_results = json.loads(Path(results_path).read_text(encoding="utf-8", errors="ignore"))
        results["train_test_f1"] = train_results.get("test_f1", None)
    else:
        results["train_test_f1"] = None

    config, model = load_config_and_model(save_dir, device)

    # Compatibility patch:
    # Some older saved configs may miss `sb_size_map` in BoundarySelectionDecoderConfig.
    # Boundaries() expects it to exist, so we add it as None when absent.
    try:
        if hasattr(config, "decoder") and config.decoder is not None:
            if not hasattr(config.decoder, "sb_size_map"):
                config.decoder.sb_size_map = None
    except Exception:
        # Don't crash evaluation if patching fails; let downstream raise if truly incompatible.
        pass

    train_data, dev_data, test_data = load_redjujube_data(data_dir)
    num_trunc = truncate_long_sequences([train_data, dev_data, test_data], max_char_len=510)
    if num_trunc > 0:
        # keep silent in most cases
        pass

    # Add expert dict features if auto lexicon exists
    auto_lexicon_path = find_auto_lexicon(save_dir)
    if auto_lexicon_path:
        lexicon = []
        with open(auto_lexicon_path, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if w:
                    lexicon.append(w)
        tokenizer = LexiconTokenizer(lexicon, max_len=10)
        for data in (train_data, dev_data, test_data):
            for entry in data:
                entry["tokens"].build_expert_dict_tags(tokenizer.tokenize)

    # Dataset & prediction
    test_set = Dataset(test_data, config, training=False)
    trainer = Trainer(model, device=device)
    pred_chunks = trainer.predict(test_set, batch_size=max_batch_size)
    gold_chunks = [ex["chunks"] for ex in test_set.data]

    scores, ave_scores = precision_recall_f1_report(gold_chunks, pred_chunks, macro_over="types")
    results["scores_by_type"] = scores
    results["micro_macro"] = {"micro": ave_scores["micro"], "macro": ave_scores["macro"]}

    # Length bucket stats
    length_stats: Dict[str, Dict[str, float]] = {b: {"gold": 0, "pred": 0, "tp": 0} for b in ["1字", "2字", "3字", "4-5字", "6字+"]}
    for golds, preds in zip(gold_chunks, pred_chunks):
        gold_set = set(golds)
        pred_set = set(preds)
        tp_set = gold_set & pred_set

        for span in golds:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]["gold"] += 1
        for span in preds:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]["pred"] += 1
        for span in tp_set:
            length = get_entity_length(span)
            bucket = get_length_bucket(length)
            length_stats[bucket]["tp"] += 1

    for bucket, st in length_stats.items():
        p = st["tp"] / st["pred"] if st["pred"] > 0 else 0.0
        r = st["tp"] / st["gold"] if st["gold"] > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        st["precision"] = p
        st["recall"] = r
        st["f1"] = f1

    results["length_stats"] = length_stats
    results["save_dir"] = save_dir
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, required=True, help="mf1 model save_dir")
    parser.add_argument("--model_b", type=str, required=True, help="mf2 model save_dir")
    parser.add_argument("--name_a", type=str, default="mf1")
    parser.add_argument("--name_b", type=str, default="mf2")
    parser.add_argument("--data_dir", type=str, default="_2DATA/RedJujube")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device(args.device)

    print("=" * 80)
    print(f"[EXP-011] Compare type & length contribution")
    print(f"device={device}")
    print(f"{args.name_a}: {args.model_a}")
    print(f"{args.name_b}: {args.model_b}")
    print("=" * 80)

    res_a = evaluate_single(args.model_a, args.data_dir, device, max_batch_size=args.batch_size)
    res_b = evaluate_single(args.model_b, args.data_dir, device, max_batch_size=args.batch_size)

    # Type comparison: print top delta in F1
    type_names = sorted(set(res_a["scores_by_type"].keys()) | set(res_b["scores_by_type"].keys()))
    deltas = []
    for t in type_names:
        fa = res_a["scores_by_type"].get(t, {}).get("f1", 0.0)
        fb = res_b["scores_by_type"].get(t, {}).get("f1", 0.0)
        deltas.append((t, fb - fa, fa, fb))
    deltas.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n" + "-" * 80)
    print("Top type F1 deltas (abs)")
    print("-" * 80)
    for t, d, fa, fb in deltas[:15]:
        print(f"  {t:<10} delta(F1)={d:+.4f}  {args.name_a}={fa:.4f}  {args.name_b}={fb:.4f}")

    # Length comparison
    print("\n" + "-" * 80)
    print("Length bucket F1 comparison")
    print("-" * 80)
    bucket_order = ["1字", "2字", "3字", "4-5字", "6字+"]
    for b in bucket_order:
        fa = res_a["length_stats"][b]["f1"]
        fb = res_b["length_stats"][b]["f1"]
        pa = res_a["length_stats"][b]["precision"]
        pb = res_b["length_stats"][b]["precision"]
        ra = res_a["length_stats"][b]["recall"]
        rb = res_b["length_stats"][b]["recall"]
        print(f"  {b:<6} F1 {args.name_a}={fa:.4f} -> {args.name_b}={fb:.4f} (P {pa:.3f}->{pb:.3f}, R {ra:.3f}->{rb:.3f})")

    # Write CSV
    out_dir = Path("experiments/EXP-011-lexicon_strategy/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "redjujube_mf1_vs_mf2_type_length.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["type", f"f1_{args.name_a}", f"f1_{args.name_b}", "delta"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in type_names:
            fa = res_a["scores_by_type"].get(t, {}).get("f1", 0.0)
            fb = res_b["scores_by_type"].get(t, {}).get("f1", 0.0)
            writer.writerow({"type": t, f"f1_{args.name_a}": fa, f"f1_{args.name_b}": fb, "delta": fb - fa})

    print(f"\n[EXP-011] wrote: {csv_path}")


if __name__ == "__main__":
    main()

