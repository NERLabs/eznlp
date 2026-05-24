#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RedJujube NER 多模型 Ensemble 脚本

支持融合多个模型的预测结果：
- Baseline
- ExpertDict (自动)
- ExpertDict (带类型)
- FLAT-CTB
- FLAT-专家词典
"""

import argparse
import torch
from eznlp.metrics import precision_recall_f1_report


def load_pred_file(path):
    """加载预测文件"""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data


def build_map(data):
    """按 gold 实体集合构建索引"""
    m = {}
    for d in data:
        gold = d["chunks"]
        key = tuple(sorted(gold))
        if key not in m:
            m[key] = (gold, d["chunks_pred"])
    return m


def eval_and_print(gold, pred, name: str):
    """评估并打印结果"""
    scores, ave_scores = precision_recall_f1_report(
        gold, pred, macro_over="types"
    )
    macro = ave_scores["macro"]
    micro = ave_scores["micro"]

    print(f"\n===== {name} =====")
    print(f"[Macro] P={macro['precision']:.2%}  R={macro['recall']:.2%}  F1={macro['f1']:.2%}")
    print(f"[Micro] P={micro['precision']:.2%}  R={micro['recall']:.2%}  F1={micro['f1']:.2%}")

    return micro['f1']


def main():
    parser = argparse.ArgumentParser(description="RedJujube 多模型 Ensemble")
    parser.add_argument("--pred1", type=str, required=True, help="第一个模型的预测文件")
    parser.add_argument("--pred2", type=str, required=True, help="第二个模型的预测文件")
    parser.add_argument("--name1", type=str, default="Model1", help="第一个模型名称")
    parser.add_argument("--name2", type=str, default="Model2", help="第二个模型名称")
    args = parser.parse_args()

    data1 = load_pred_file(args.pred1)
    data2 = load_pred_file(args.pred2)

    map1 = build_map(data1)
    map2 = build_map(data2)

    keys_common = sorted(set(map1.keys()) & set(map2.keys()))

    print(f"\n{args.name1} 样本数: {len(map1)}")
    print(f"{args.name2} 样本数: {len(map2)}")
    print(f"共同对齐样本数: {len(keys_common)}")

    if not keys_common:
        print("\n⚠️ 两套预测没有共同样本，无法 ensemble")
        return

    gold, pred1, pred2 = [], [], []
    for k in keys_common:
        g1, p1 = map1[k]
        g2, p2 = map2[k]
        gold.append(g1)
        pred1.append(p1)
        pred2.append(p2)

    f1_1 = eval_and_print(gold, pred1, f"{args.name1} 单模型")
    f1_2 = eval_and_print(gold, pred2, f"{args.name2} 单模型")

    pred_union = [list(set(p1) | set(p2)) for p1, p2 in zip(pred1, pred2)]
    pred_inter = [list(set(p1) & set(p2)) for p1, p2 in zip(pred1, pred2)]

    f1_union = eval_and_print(gold, pred_union, "Union 融合（并集，召回优先）")
    f1_inter = eval_and_print(gold, pred_inter, "Intersection 融合（交集，精确优先）")

    print("\n" + "="*60)
    print("📊 Ensemble 效果汇总")
    print("="*60)
    print(f"{args.name1}: {f1_1:.2%}")
    print(f"{args.name2}: {f1_2:.2%}")
    print(f"Union 融合: {f1_union:.2%} ({'+' if f1_union > max(f1_1, f1_2) else ''}{(f1_union - max(f1_1, f1_2))*100:.2f}%)")
    print(f"Intersection: {f1_inter:.2%}")


if __name__ == "__main__":
    main()