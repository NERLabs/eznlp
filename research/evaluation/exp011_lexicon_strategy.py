#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-011: 通用词典抽取策略（自适应 min_freq，兼顾长短实体）

策略定义：
1) 候选 min_freq k ∈ {1,2,3}
2) 在数据集 D 的 train 上，从 k 抽取词典 Lexicon(D,k)
3) 匹配 proxy：
   - 覆盖/召回（总体）match_recall
   - 覆盖/召回（短实体）recall_short：长度 < long_min_len 的 gold 实体，被词典匹配到的比例
   - 覆盖/召回（长实体）recall_long：长度 >= long_min_len 的 gold 实体，被词典匹配到的比例
   - 噪声 proxy match_precision：交集 span 占匹配 span 的比例
   - 类别分布平衡召回 balanced_recall：
       对每个实体类型 t 计算 recall_short(t) 与 recall_long(t)，再用该类型在短/长上的 gold 分布做幂次融合，
       最后按该类型的 gold 支持量加权求和。该项实现了“根据不同类别数据分布使用不同策略”。
4) 双向相对约束过滤（兼顾长短）：
   keep k 使得 recall_short(k) >= beta_short * max_recall_short 且 recall_long(k) >= beta_long * max_recall_long
5) 最终选择：在 keep 中最大化 score(k)=match_precision(k) * balanced_recall(k)
6) 用已有 public 训练结果（boson/clue）评估 strategy_k* 的 test F1，并与 fixed mf2/mf3 与 candidates best-of 做对比

输出写入：
  experiments/EXP-011-lexicon_strategy/analysis/
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sys

# Ensure repo root is on PYTHONPATH when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eznlp.token import LexiconTokenizer


EntitySpan = Tuple[int, int]  # (start, end) in character indices (token == char in BMES files)


def load_bmes_file(path: str) -> List[Dict[str, List[str]]]:
    """
    读取 BMES 数据文件（token tag），空行分句。
    返回每句：{"tokens": [...], "labels": [...]}
    """
    sentences: List[Dict[str, List[str]]] = []
    tokens: List[str] = []
    labels: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "labels": labels})
                    tokens = []
                    labels = []
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Unexpected BMES line format in {path}: {line!r}")
            tok, lab = parts
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        sentences.append({"tokens": tokens, "labels": labels})
    return sentences


def extract_gold_entity_spans(tokens: List[str], labels: List[str]) -> List[EntitySpan]:
    """
    从 BMES 标签抽取实体 span（start,end），label scheme:
    - O
    - B-XXX, M-XXX, E-XXX, S-XXX
    返回 span 列表（同一 span 不会重复）。
    """
    spans: List[EntitySpan] = []
    current_start: Optional[int] = None
    current_type: Optional[str] = None

    for i, (tok, lab) in enumerate(zip(tokens, labels)):
        if lab == "O":
            if current_start is not None:
                # 异常数据：B/M 后突然 O
                current_start = None
                current_type = None
            continue

        if lab.startswith("B-"):
            current_start = i
            current_type = lab[2:]
        elif lab.startswith("M-"):
            # 只延续，不检查
            if current_start is None:
                # 异常数据
                current_start = i
                current_type = lab[2:]
        elif lab.startswith("E-"):
            if current_start is None:
                current_start = i
            # end=i+1 (end exclusive)
            spans.append((current_start, i + 1))
            current_start = None
            current_type = None
        elif lab.startswith("S-"):
            spans.append((i, i + 1))
            current_start = None
            current_type = None
        else:
            raise ValueError(f"Unknown label in BMES file: {lab!r}")

    return spans


def extract_lexicon_from_train(
    train_sentences: List[Dict[str, List[str]]], min_freq: int
) -> Set[str]:
    """
    与 train_general_expert_boundary.py 保持一致：
    lexicon = [entity_text for entity_text,count in Counter(...) if count>=min_freq]
    entity_text 来自 gold entities 的 span 拼接：''.join(tokens[start:end])
    """
    from collections import Counter

    counter: Counter[str] = Counter()
    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        spans = extract_gold_entity_spans(tokens, labels)
        for s, e in spans:
            entity_text = "".join(tokens[s:e])
            if entity_text:
                counter[entity_text] += 1

    lex = {w for w, c in counter.items() if c >= min_freq}
    return lex


def compute_match_proxy(
    sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
    max_len: int = 10,
) -> Dict[str, float]:
    """
    返回：
    - match_precision: correct_spans / matched_spans
    - match_recall: correct_spans / gold_spans
    其中 correct_spans = matched_spans ∩ gold_spans
    """
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)

    correct_total = 0
    matched_total = 0
    gold_total = 0

    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        text = "".join(tokens)

        gold_spans = set(extract_gold_entity_spans(tokens, labels))
        gold_total += len(gold_spans)

        matched_spans: Set[EntitySpan] = set()
        for word_text, s, e in tokenizer.tokenize(text):
            # token==char, use span indices directly
            matched_spans.add((s, e))

        matched_total += len(matched_spans)
        correct_total += len(gold_spans.intersection(matched_spans))

    match_precision = (correct_total / matched_total) if matched_total > 0 else 0.0
    match_recall = (correct_total / gold_total) if gold_total > 0 else 0.0
    return {
        "match_precision": float(match_precision),
        "match_recall": float(match_recall),
    }


def parse_final_test_f1(training_log_path: str) -> Optional[float]:
    """
    从 training.log 中解析最终测试 F1。
    """
    txt = Path(training_log_path).read_text(encoding="utf-8", errors="ignore")
    # prefer the canonical format
    m = re.findall(r"最终测试 F1:\s*([0-9.]+)", txt)
    if m:
        return float(m[-1])
    # fallback
    m2 = re.findall(r"F1:\s*([0-9.]+)", txt)
    return float(m2[-1]) if m2 else None


def find_latest_training_log(glob_pat: str) -> Optional[str]:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        return None
    return paths[-1]


@dataclass
class CandidateResult:
    k: int
    lexicon_size: int
    covered_rate_short: float  # recall_short
    covered_rate_long: float  # recall_long
    balanced_recall: float  # per-type balanced recall
    match_precision: float
    match_precision_short: float
    match_precision_long: float
    f1_short: float
    f1_long: float
    balanced_f1: float


def select_k_star_balanced(
    candidates: List[CandidateResult],
    beta_short: float,
    beta_long: float,
    min_keep_candidates: int = 2,
    objective: str = "balanced_f1",
) -> Tuple[int, List[int]]:
    """
    recall_short >= beta_short * max_short AND recall_long >= beta_long * max_long
    then argmax score = match_precision * balanced_recall
    """
    max_short = max(x.covered_rate_short for x in candidates)
    max_long = max(x.covered_rate_long for x in candidates)

    keep = [
        x.k
        for x in candidates
        if x.covered_rate_short >= beta_short * max_short and x.covered_rate_long >= beta_long * max_long
    ]

    if len(keep) < min_keep_candidates:
        # 如果硬约束过严导致退化，只用目标函数做选择。
        keep = [x.k for x in candidates]

    kept = [x for x in candidates if x.k in keep]
    if not kept:
        kept = candidates

    if objective == "balanced_f1":
        chosen = max(kept, key=lambda z: z.balanced_f1).k
    elif objective == "match_precision":
        chosen = max(kept, key=lambda z: z.match_precision).k
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    return chosen, keep


def extract_gold_typed_entity_spans(tokens: List[str], labels: List[str]) -> List[Tuple[int, int, str]]:
    """
    从 BMES 标签抽取实体 span（start,end,type）
    """
    spans: List[Tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_type: Optional[str] = None

    for i, (_tok, lab) in enumerate(zip(tokens, labels)):
        if lab == "O":
            current_start = None
            current_type = None
            continue

        if lab.startswith("B-"):
            current_start = i
            current_type = lab[2:]
        elif lab.startswith("M-"):
            if current_start is None:
                current_start = i
                current_type = lab[2:]
        elif lab.startswith("E-"):
            if current_start is None:
                current_start = i
            spans.append((current_start, i + 1, current_type if current_type is not None else "None"))
            current_start = None
            current_type = None
        elif lab.startswith("S-"):
            spans.append((i, i + 1, lab[2:]))
            current_start = None
            current_type = None
        else:
            raise ValueError(f"Unknown label in BMES file: {lab!r}")

    return spans


def compute_balanced_match_proxy(
    sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
    *,
    max_len: int = 10,
    long_min_len: int = 6,
) -> Dict[str, float]:
    """
    计算兼顾长短实体的匹配 proxy：
      - match_precision（总体）
      - recall_short（短实体召回）
      - recall_long（长实体召回）
      - balanced_recall（按类型分布做短/长融合）
    """
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)

    # overall stats
    correct_total = 0
    matched_total = 0
    gold_total = 0

    # short/long buckets for precision/recall
    matched_short_total = 0
    matched_long_total = 0

    gold_short_total = 0
    gold_long_total = 0
    tp_short_total = 0
    tp_long_total = 0

    # per-type counts (for balanced_recall only)
    gold_short_by_type: Dict[str, int] = {}
    gold_long_by_type: Dict[str, int] = {}
    tp_short_by_type: Dict[str, int] = {}
    tp_long_by_type: Dict[str, int] = {}

    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        text = "".join(tokens)

        typed_spans = extract_gold_typed_entity_spans(tokens, labels)
        gold_map: Dict[EntitySpan, str] = {(s, e): t for s, e, t in typed_spans}
        gold_spans = set(gold_map.keys())

        # gold counts (short/long)
        for s, e, t in typed_spans:
            span_len = e - s
            if span_len >= long_min_len:
                gold_long_total += 1
                gold_long_by_type[t] = gold_long_by_type.get(t, 0) + 1
            else:
                gold_short_total += 1
                gold_short_by_type[t] = gold_short_by_type.get(t, 0) + 1

        gold_total += len(gold_spans)

        matched_spans: Set[EntitySpan] = set()
        matched_short_spans = set()
        matched_long_spans = set()
        for _word_text, s, e in tokenizer.tokenize(text):
            matched_spans.add((s, e))
            if (e - s) >= long_min_len:
                matched_long_spans.add((s, e))
            else:
                matched_short_spans.add((s, e))

        matched_total += len(matched_spans)
        correct = gold_spans.intersection(matched_spans)
        correct_total += len(correct)

        matched_short_total += len(matched_short_spans)
        matched_long_total += len(matched_long_spans)

        # tp counts (short/long) and type-balanced recall inputs
        for s, e in correct:
            t = gold_map[(s, e)]
            span_len = e - s
            if span_len >= long_min_len:
                tp_long_total += 1
                tp_long_by_type[t] = tp_long_by_type.get(t, 0) + 1
            else:
                tp_short_total += 1
                tp_short_by_type[t] = tp_short_by_type.get(t, 0) + 1

    match_precision = correct_total / matched_total if matched_total > 0 else 0.0
    recall_short = tp_short_total / gold_short_total if gold_short_total > 0 else 0.0
    recall_long = tp_long_total / gold_long_total if gold_long_total > 0 else 0.0
    match_precision_short = tp_short_total / matched_short_total if matched_short_total > 0 else 0.0
    match_precision_long = tp_long_total / matched_long_total if matched_long_total > 0 else 0.0

    f1_short = (
        2 * match_precision_short * recall_short / (match_precision_short + recall_short)
        if (match_precision_short + recall_short) > 0
        else 0.0
    )
    f1_long = (
        2 * match_precision_long * recall_long / (match_precision_long + recall_long)
        if (match_precision_long + recall_long) > 0
        else 0.0
    )

    # dataset-dependent fusion: weight long/short by their gold support
    gold_total_sb = gold_short_total + gold_long_total
    w_long = gold_long_total / gold_total_sb if gold_total_sb > 0 else 0.5
    w_short = 1.0 - w_long
    balanced_f1 = w_short * f1_short + w_long * f1_long

    # balanced recall:
    # for each type t: fuse short/long recalls with exponent alpha_t derived from that type's gold split
    total_gold_by_type = set(gold_short_by_type.keys()) | set(gold_long_by_type.keys()) | set(tp_short_by_type.keys()) | set(tp_long_by_type.keys())
    type_weights_sum = 0
    balanced_sum = 0.0

    for t in total_gold_by_type:
        g_short = gold_short_by_type.get(t, 0)
        g_long = gold_long_by_type.get(t, 0)
        g_total = g_short + g_long
        if g_total == 0:
            continue

        r_short = tp_short_by_type.get(t, 0) / g_short if g_short > 0 else 0.0
        r_long = tp_long_by_type.get(t, 0) / g_long if g_long > 0 else 0.0

        alpha_t = g_short / g_total  # short proportion for this type

        if alpha_t >= 1.0:
            balanced_t = r_short
        elif alpha_t <= 0.0:
            balanced_t = r_long
        else:
            # geometric-style fusion to penalize imbalance
            balanced_t = (r_short**alpha_t) * (r_long ** (1.0 - alpha_t))

        w_t = g_total
        type_weights_sum += w_t
        balanced_sum += w_t * balanced_t

    balanced_recall = balanced_sum / type_weights_sum if type_weights_sum > 0 else 0.0

    return {
        "match_precision": float(match_precision),
        "recall_short": float(recall_short),
        "recall_long": float(recall_long),
        "balanced_recall": float(balanced_recall),
        "match_precision_short": float(match_precision_short),
        "match_precision_long": float(match_precision_long),
        "f1_short": float(f1_short),
        "f1_long": float(f1_long),
        "balanced_f1": float(balanced_f1),
    }


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.pstdev(vals)


def public_model_dir_name(dataset: str, k: int) -> str:
    """
    对应你目前 public 目录的命名：
    - min_freq=2: <dataset>_bs_dict_focal
    - min_freq=1: <dataset>_bs_dict_focal_mf1
    - min_freq=3: <dataset>_bs_dict_focal_mf3
    """
    if k == 2:
        return f"{dataset}_bs_dict_focal"
    if k == 1:
        return f"{dataset}_bs_dict_focal_mf1"
    if k == 3:
        return f"{dataset}_bs_dict_focal_mf3"
    raise ValueError(f"Unsupported k for public mapping: {k}")


def eval_public_test_f1(dataset: str, k: int, seeds: Sequence[int]) -> Tuple[List[float], float, float]:
    base_dir = Path("experiments/EXP-010-optimization/results_public")
    model_group = public_model_dir_name(dataset, k)

    f1s: List[float] = []
    for seed in seeds:
        log_glob = str(base_dir / model_group / f"seed_{seed}" / "expert_boundary_*" / "training.log")
        log_path = find_latest_training_log(log_glob)
        if not log_path:
            continue
        f1 = parse_final_test_f1(log_path)
        if f1 is None:
            continue
        f1s.append(f1)

    mean, std = mean_std(f1s)
    return f1s, mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.5, help="兼容长短实体时的默认相对约束系数（可由 beta_short/beta_long 覆盖）")
    parser.add_argument("--beta_short", type=float, default=None)
    parser.add_argument("--beta_long", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--candidates", type=str, default="1,2,3")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--long_min_len", type=int, default=6, help="定义“长实体”的最小长度")
    parser.add_argument(
        "--objective",
        type=str,
        default="balanced_f1",
        choices=["balanced_f1", "match_precision"],
        help="A: balanced_f1（兼顾长短目标）；B: match_precision（约束确保长短达标后只选精度更高）",
    )
    parser.add_argument("--out_suffix", type=str, default="", help="输出文件名后缀，避免覆盖")
    args = parser.parse_args()

    beta = args.beta
    beta_short = args.beta_short if args.beta_short is not None else beta
    beta_long = args.beta_long if args.beta_long is not None else beta
    max_len = args.max_len
    candidates = [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    long_min_len = args.long_min_len
    objective = args.objective
    out_suffix = args.out_suffix

    out_dir = Path("experiments/EXP-011-lexicon_strategy/analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dataset train BMES paths
    data_root = Path(".datasets/raw")  # not used; keep readability
    data_root = Path("datasets/raw")

    dataset_paths = {
        # train splits
        "redjujube": str(data_root / "RedJujube" / "redjujube_train.bmes"),
        "boson": str(data_root / "boson" / "boson.train.bmes"),
        "clue": str(data_root / "clue" / "train.char.bmes"),
    }

    # For public evaluation we only report boson/clue (strategy uses redjujube train as explanation),
    # but selection is computed per dataset using its own train statistics as described.
    public_datasets = ["boson", "clue"]

    # 1) compute candidate table (train proxy) for each dataset
    candidate_rows: List[Dict[str, object]] = []
    strategy_rows: List[Dict[str, object]] = []

    for dataset, train_path in dataset_paths.items():
        train_sents = load_bmes_file(train_path)

        candidates_res: List[CandidateResult] = []
        for k in candidates:
            lexicon = extract_lexicon_from_train(train_sents, min_freq=k)
            proxy = compute_balanced_match_proxy(
                train_sents,
                lexicon,
                max_len=max_len,
                long_min_len=long_min_len,
            )
            candidates_res.append(
                CandidateResult(
                    k=k,
                    lexicon_size=len(lexicon),
                    covered_rate_short=proxy["recall_short"],
                    covered_rate_long=proxy["recall_long"],
                    balanced_recall=proxy["balanced_recall"],
                    match_precision=proxy["match_precision"],
                    match_precision_short=proxy["match_precision_short"],
                    match_precision_long=proxy["match_precision_long"],
                    f1_short=proxy["f1_short"],
                    f1_long=proxy["f1_long"],
                    balanced_f1=proxy["balanced_f1"],
                )
            )

            candidate_rows.append(
                {
                    "dataset": dataset,
                    "min_freq": k,
                    "|lexicon|": len(lexicon),
                    "covered_rate_short_train": proxy["recall_short"],
                    "covered_rate_long_train": proxy["recall_long"],
                    "balanced_recall_train": proxy["balanced_recall"],
                    "match_precision_train": proxy["match_precision"],
                    "match_precision_short_train": proxy["match_precision_short"],
                    "match_precision_long_train": proxy["match_precision_long"],
                    "f1_short_train": proxy["f1_short"],
                    "f1_long_train": proxy["f1_long"],
                    "balanced_f1_train": proxy["balanced_f1"],
                }
            )

        k_star, keep = select_k_star_balanced(
            candidates_res,
            beta_short=beta_short,
            beta_long=beta_long,
            objective=objective,
        )

        # strategy on train proxy
        strat = {
            "dataset": dataset,
            "beta_short": beta_short,
            "beta_long": beta_long,
            "k_star": k_star,
            "keep_ks": ",".join(map(str, keep)),
        }
        strategy_rows.append(strat)

    # write candidate/proxy tables
    cand_csv = out_dir / f"candidate_proxy_table{out_suffix}.csv"
    with open(cand_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "min_freq",
                "|lexicon|",
                "covered_rate_short_train",
                "covered_rate_long_train",
                "balanced_recall_train",
                "match_precision_train",
                "match_precision_short_train",
                "match_precision_long_train",
                "f1_short_train",
                "f1_long_train",
                "balanced_f1_train",
            ],
        )
        writer.writeheader()
        writer.writerows(candidate_rows)

    strat_csv = out_dir / f"strategy_selection_table{out_suffix}.csv"
    with open(strat_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "beta", "beta_short", "beta_long", "k_star", "keep_ks"],
        )
        writer.writeheader()
        # Backward compatible columns: keep a legacy "beta" field for older readers.
        for row in strategy_rows:
            row.setdefault("beta", beta)
            writer.writerow(row)

    # 2) evaluate public datasets on test F1
    summary_rows: List[Dict[str, object]] = []
    for dataset in public_datasets:
        # select k* according to its own train proxy
        train_path = dataset_paths[dataset]
        train_sents = load_bmes_file(train_path)

        candidates_res: List[CandidateResult] = []
        for k in candidates:
            lexicon = extract_lexicon_from_train(train_sents, min_freq=k)
            proxy = compute_balanced_match_proxy(
                train_sents,
                lexicon,
                max_len=max_len,
                long_min_len=long_min_len,
            )
            candidates_res.append(
                CandidateResult(
                    k=k,
                    lexicon_size=len(lexicon),
                    covered_rate_short=proxy["recall_short"],
                    covered_rate_long=proxy["recall_long"],
                    balanced_recall=proxy["balanced_recall"],
                    match_precision=proxy["match_precision"],
                    match_precision_short=proxy["match_precision_short"],
                    match_precision_long=proxy["match_precision_long"],
                    f1_short=proxy["f1_short"],
                    f1_long=proxy["f1_long"],
                    balanced_f1=proxy["balanced_f1"],
                )
            )
        k_star, keep = select_k_star_balanced(
            candidates_res,
            beta_short=beta_short,
            beta_long=beta_long,
            objective=objective,
        )

        # strategy evaluation
        f1s_strategy, mean_strategy, std_strategy = eval_public_test_f1(dataset, k_star, seeds)

        # fixed baselines: mf2 and mf3
        _, mean_mf2, std_mf2 = eval_public_test_f1(dataset, 2, seeds)
        _, mean_mf3, std_mf3 = eval_public_test_f1(dataset, 3, seeds)

        # candidates best
        cand_means = []
        for k in candidates:
            _, m, _ = eval_public_test_f1(dataset, k, seeds)
            cand_means.append((k, m))
        best_k, best_mean = max(cand_means, key=lambda x: x[1])
        regret = best_mean - mean_strategy

        summary_rows.append(
            {
                "dataset": dataset,
                "beta": beta,
                "strategy_k": k_star,
                "strategy_test_f1_mean": mean_strategy,
                "strategy_test_f1_std": std_strategy,
                "fixed_mf2_test_f1_mean": mean_mf2,
                "fixed_mf2_test_f1_std": std_mf2,
                "fixed_mf3_test_f1_mean": mean_mf3,
                "fixed_mf3_test_f1_std": std_mf3,
                "candidates_best_k": best_k,
                "candidates_best_test_f1_mean": best_mean,
                "regret(best-strategy)": regret,
                "seeds_used_strategy": ",".join(map(str, seeds)),
                "keep_ks": ",".join(map(str, keep)),
            }
        )

    summary_csv = out_dir / f"strategy_vs_fixed_f1_summary{out_suffix}.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(summary_rows[0].keys()) if summary_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # 3) print small summary
    print(f"[EXP-011] beta={beta}, max_len={max_len}, candidates={candidates}, seeds={seeds}")
    print(f"[EXP-011] long_min_len={long_min_len}, beta_short={beta_short}, beta_long={beta_long}")
    print(f"[EXP-011] candidate proxy table: {cand_csv}")
    print(f"[EXP-011] strategy selection table: {strat_csv}")
    print(f"[EXP-011] f1 summary: {summary_csv}")
    for row in summary_rows:
        print(
            f"- {row['dataset']}: strategy(k={row['strategy_k']}) "
            f"F1={row['strategy_test_f1_mean']:.4f}±{row['strategy_test_f1_std']:.4f}, "
            f"best_k={row['candidates_best_k']} bestF1={row['candidates_best_test_f1_mean']:.4f}, "
            f"regret={row['regret(best-strategy)']:.4f}"
        )


if __name__ == "__main__":
    main()

