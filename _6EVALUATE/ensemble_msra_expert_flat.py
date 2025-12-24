#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRA-ER ExpertDict vs FLAT-MSRA 结果级 ensemble 脚本

功能：
- 加载 ExpertDict 模型的预测结果（test_msra_ner.py 导出）
- 加载 FLAT-MSRA 模型的预测结果（test_flat_msra_expert.py 导出）
- 按 gold 实体集合对齐样本（处理样本数不一致的情况）
- 计算：
  - ExpertDict 单模型 PRF1
  - FLAT-MSRA 单模型 PRF1
  - Union 融合（集合并集） PRF1
  - Intersection 融合（集合交集） PRF1
"""

import argparse
import torch
from eznlp.metrics import precision_recall_f1_report


def load_pred_file(path):
    """
    读取形如：
    [
      {"chunks": [... gold 实体 ...],
       "chunks_pred": [... pred 实体 ...]},
      ...
    ]
    的 .pt 文件。
    """
    data = torch.load(path, map_location="cpu")
    return data


def eval_and_print(gold, pred, name: str):
    scores, ave_scores = precision_recall_f1_report(
        gold, pred, macro_over="types"
    )
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]

    print(f"\n===== {name} =====")
    print("[Macro] "
          f"P={macro['precision']:.4%}  "
          f"R={macro['recall']:.4%}  "
          f"F1={macro['f1']:.4%}")
    print("[Micro] "
          f"P={micro['precision']:.4%}  "
          f"R={micro['recall']:.4%}  "
          f"F1={micro['f1']:.4%}")

    return scores, ave_scores


def main():
    parser = argparse.ArgumentParser(
        description="MSRA ExpertDict + FLAT-MSRA 结果级 ensemble"
    )
    parser.add_argument(
        "--expert_pred",
        type=str,
        default="cache/MSRA-ER/20251208-011438-972934/predictions_test_msra.pt",
        help="test_msra_ner.py 导出的预测文件路径",
    )
    parser.add_argument(
        "--flat_pred",
        type=str,
        default="cache/flat_inter_msra_ft3/flat_20251224-001755/predictions_test_flat_msra.pt",
        help="test_flat_msra_expert.py 导出的预测文件路径",
    )
    args = parser.parse_args()

    # 1. 加载两套预测（列表，每个元素是 dict）
    data_exp = load_pred_file(args.expert_pred)
    data_flat = load_pred_file(args.flat_pred)

    # 2. 按 gold 实体集合构建索引（key = sorted(chunks)）
    def build_map(data):
        m = {}
        for d in data:
            gold = d["chunks"]
            # gold 是 [(type, s, e), ...]，转成排序后的 tuple 作为 key
            key = tuple(sorted(gold))
            # 如遇重复 key，保留第一个即可
            if key not in m:
                m[key] = (gold, d["chunks_pred"])
        return m

    map_exp = build_map(data_exp)
    map_flat = build_map(data_flat)

    keys_common = sorted(set(map_exp.keys()) & set(map_flat.keys()))
    
    print(f"ExpertDict 样本数: {len(map_exp)}")
    print(f"FLAT-MSRA 样本数: {len(map_flat)}")
    print(f"共同对齐样本数: {len(keys_common)}")
    
    if not keys_common:
        print("\n⚠️ 两套预测几乎没有共同的 gold 句子，无法做 ensemble。")
        print("可能原因：两边测试集预处理方式不一致（如 BERT 切分 vs 不切分）。")
        print("建议：直接对比各自的单模型 F1 即可。")
        return

    gold = []
    pred_exp = []
    pred_flat = []
    for k in keys_common:
        g1, pe = map_exp[k]
        g2, pf = map_flat[k]
        gold.append(g1)
        pred_exp.append(pe)
        pred_flat.append(pf)

    # 3. 单模型结果（用于 sanity check）
    eval_and_print(gold, pred_exp, "ExpertDict 单模型（交集样本）")
    eval_and_print(gold, pred_flat, "FLAT-MSRA 单模型（交集样本）")

    # 4. Union / Intersection 融合
    pred_union = []
    pred_inter = []

    for g, pe, pf in zip(gold, pred_exp, pred_flat):
        se = set(pe)
        sf = set(pf)

        # 并集：两个模型的所有实体都保留
        pred_union.append(list(se | sf))

        # 交集：只有两个模型都认同的实体
        pred_inter.append(list(se & sf))

    eval_and_print(gold, pred_union, "Union 融合（并集）")
    eval_and_print(gold, pred_inter, "Intersection 融合（交集）")


if __name__ == "__main__":
    main()