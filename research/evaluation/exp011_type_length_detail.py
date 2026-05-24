#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-011: 更深层次对比 - 词典覆盖按类别 & 按长度分桶

计算对象：在某个数据集的 train split 上
对每个 min_freq k ∈ candidates：
  1) 构建 lexicon = {entity_text | entity_text 频次 >= k}
  2) 统计 gold 实体：
     - per type：covered_instances / total_instances, covered_unique / total_unique
     - per length bucket：covered_instances / total_instances
  3) 进一步计算匹配代理（与训练一致的最大匹配长度 max_len）：
     - match_precision / match_recall（全局）
     - 每个长度分桶上的 match_precision / match_recall（用于解释“长实体贡献”）

输出：
  - 直接在控制台打印 top 类型覆盖提升、长度分桶覆盖率变化
  - 写入 CSV：per_type_coverage.csv, per_length_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sys

import sys as _sys

# Ensure repo root on sys.path when running this file directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from eznlp.token import LexiconTokenizer


EntitySpan = Tuple[int, int]  # (start, end) token index, end exclusive
TypedSpan = Tuple[int, int, str]  # (start, end, type)


def load_bmes_file(path: str) -> List[Dict[str, List[str]]]:
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
                raise ValueError(f"Unexpected BMES line format: {line!r}")
            tok, lab = parts
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        sentences.append({"tokens": tokens, "labels": labels})
    return sentences


def extract_gold_typed_spans(tokens: List[str], labels: List[str]) -> List[TypedSpan]:
    spans: List[TypedSpan] = []
    current_start: Optional[int] = None
    current_type: Optional[str] = None

    for i, (tok, lab) in enumerate(zip(tokens, labels)):
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
            raise ValueError(f"Unknown BMES label: {lab!r}")

    return spans


def length_bucket(length: int) -> str:
    if length == 1:
        return "1字"
    if length == 2:
        return "2字"
    if length == 3:
        return "3字"
    if 4 <= length <= 5:
        return "4-5字"
    return "6字+"


def build_lexicon_from_train(train_sentences: List[Dict[str, List[str]]], min_freq: int) -> Set[str]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_spans(tokens, labels)
        for s, e, _t in typed_spans:
            entity_text = "".join(tokens[s:e])
            if entity_text:
                counter[entity_text] += 1
    return {w for w, c in counter.items() if c >= min_freq}


def compute_coverage_by_type_and_length(
    train_sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """
    返回：
      per_type: type -> {total_instances, covered_instances, total_unique, covered_unique}
      per_len: bucket -> {total_instances, covered_instances}
    """
    per_type = {}
    per_len = {}

    # For unique computation we need entity_text sets
    # type_unique_texts: type -> set(entity_text)
    type_unique_texts: Dict[str, Set[str]] = {}
    type_unique_covered_texts: Dict[str, Set[str]] = {}

    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_spans(tokens, labels)

        for s, e, t in typed_spans:
            entity_text = "".join(tokens[s:e])
            if not entity_text:
                continue

            b = length_bucket(e - s)
            per_len.setdefault(b, {"total_instances": 0, "covered_instances": 0})
            per_len[b]["total_instances"] += 1
            if entity_text in lexicon:
                per_len[b]["covered_instances"] += 1

            per_type.setdefault(t, {"total_instances": 0, "covered_instances": 0})
            per_type[t]["total_instances"] += 1
            if entity_text in lexicon:
                per_type[t]["covered_instances"] += 1

            type_unique_texts.setdefault(t, set()).add(entity_text)
            if entity_text in lexicon:
                type_unique_covered_texts.setdefault(t, set()).add(entity_text)

    for t in per_type.keys():
        per_type[t]["total_unique"] = len(type_unique_texts.get(t, set()))
        per_type[t]["covered_unique"] = len(type_unique_covered_texts.get(t, set()))

    return per_type, per_len


def compute_match_proxy(
    train_sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
    max_len: int = 10,
) -> Dict[str, float]:
    """
    用 LexiconTokenizer 做最大正向匹配，统计 matched spans 与 gold spans 的交集。
    这里 match_* 忽略类型，只评价边界覆盖与误匹配。
    """
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)
    correct_total = 0
    matched_total = 0
    gold_total = 0

    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_spans(tokens, labels)
        gold_spans: Set[EntitySpan] = {(s, e) for s, e, _t in typed_spans}
        gold_total += len(gold_spans)

        text = "".join(tokens)
        matched_spans: Set[EntitySpan] = set()
        for _word_text, s, e in tokenizer.tokenize(text):
            matched_spans.add((s, e))
        matched_total += len(matched_spans)

        correct_total += len(gold_spans.intersection(matched_spans))

    match_precision = correct_total / matched_total if matched_total > 0 else 0.0
    match_recall = correct_total / gold_total if gold_total > 0 else 0.0
    return {"match_precision": float(match_precision), "match_recall": float(match_recall)}


def compute_match_proxy_by_length(
    train_sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
    max_len: int = 10,
) -> Dict[str, dict]:
    """
    按长度分桶输出：
      bucket -> {gold, matched, tp, precision, recall}
    """
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)

    buckets = ["1字", "2字", "3字", "4-5字", "6字+"]
    stats = {b: {"gold": 0, "matched": 0, "tp": 0} for b in buckets}

    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_spans(tokens, labels)
        gold_spans_by_bucket: Dict[str, Set[EntitySpan]] = {b: set() for b in buckets}
        for s, e, _t in typed_spans:
            b = length_bucket(e - s)
            gold_spans_by_bucket[b].add((s, e))

        text = "".join(tokens)
        matched_spans_by_bucket: Dict[str, Set[EntitySpan]] = {b: set() for b in buckets}
        for _word_text, s, e in tokenizer.tokenize(text):
            b = length_bucket(e - s)
            matched_spans_by_bucket[b].add((s, e))

        for b in buckets:
            gold_set = gold_spans_by_bucket[b]
            matched_set = matched_spans_by_bucket[b]
            tp = len(gold_set.intersection(matched_set))
            stats[b]["gold"] += len(gold_set)
            stats[b]["matched"] += len(matched_set)
            stats[b]["tp"] += tp

    for b in buckets:
        matched = stats[b]["matched"]
        gold = stats[b]["gold"]
        tp = stats[b]["tp"]
        stats[b]["precision"] = tp / matched if matched > 0 else 0.0
        stats[b]["recall"] = tp / gold if gold > 0 else 0.0
        stats[b]["f1"] = (
            2 * stats[b]["precision"] * stats[b]["recall"] / (stats[b]["precision"] + stats[b]["recall"])
            if (stats[b]["precision"] + stats[b]["recall"]) > 0
            else 0.0
        )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="redjujube", choices=["redjujube", "boson", "clue"])
    parser.add_argument("--candidates", type=str, default="1,2,3")
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="experiments/EXP-011-lexicon_strategy/analysis")
    args = parser.parse_args()

    dataset = args.dataset
    candidates = [int(x.strip()) for x in args.candidates.split(",") if x.strip()]
    max_len = args.max_len
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # dataset train paths
    if dataset == "redjujube":
        train_path = "datasets/raw/RedJujube/redjujube_train.bmes"
    elif dataset == "boson":
        train_path = "datasets/raw/boson/boson.train.bmes"
    else:
        train_path = "datasets/raw/clue/train.char.bmes"

    train_path = Path(train_path)
    train_sents = load_bmes_file(str(train_path))
    print(f"[EXP-011] dataset={dataset} train_sentences={len(train_sents)}")

    per_k_type: Dict[int, Dict[str, dict]] = {}
    per_k_len: Dict[int, Dict[str, dict]] = {}
    per_k_match_len: Dict[int, Dict[str, dict]] = {}

    for k in candidates:
        lexicon = build_lexicon_from_train(train_sents, min_freq=k)
        per_type, per_len = compute_coverage_by_type_and_length(train_sents, lexicon)
        per_k_type[k] = per_type
        per_k_len[k] = per_len
        per_k_match_len[k] = compute_match_proxy_by_length(train_sents, lexicon, max_len=max_len)
        global_proxy = compute_match_proxy(train_sents, lexicon, max_len=max_len)
        print(
            f"  k={k}: |lexicon|={len(lexicon)} "
            f"match_precision={global_proxy['match_precision']:.4f} match_recall={global_proxy['match_recall']:.4f}"
        )

    # Write CSVs (per type)
    type_csv = out_dir / f"{dataset}_per_type_coverage.csv"
    with open(type_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "min_freq", "type", "total_instances", "covered_instances", "covered_rate", "total_unique", "covered_unique"],
        )
        writer.writeheader()
        for k in candidates:
            for t, st in per_k_type[k].items():
                total_instances = st["total_instances"]
                covered_instances = st["covered_instances"]
                covered_rate = covered_instances / total_instances if total_instances > 0 else 0.0
                writer.writerow(
                    {
                        "dataset": dataset,
                        "min_freq": k,
                        "type": t,
                        "total_instances": total_instances,
                        "covered_instances": covered_instances,
                        "covered_rate": covered_rate,
                        "total_unique": st.get("total_unique", 0),
                        "covered_unique": st.get("covered_unique", 0),
                    }
                )

    # Write CSVs (per length bucket)
    len_csv = out_dir / f"{dataset}_per_length_coverage.csv"
    with open(len_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "min_freq", "bucket", "total_instances", "covered_instances", "covered_rate", "match_precision", "match_recall", "match_f1"],
        )
        writer.writeheader()
        for k in candidates:
            # coverage by type-entity_text membership
            per_len = per_k_len[k]
            match_len = per_k_match_len[k]
            for bucket in ["1字", "2字", "3字", "4-5字", "6字+"]:
                st = per_len.get(bucket, {"total_instances": 0, "covered_instances": 0})
                total_instances = st["total_instances"]
                covered_instances = st["covered_instances"]
                covered_rate = covered_instances / total_instances if total_instances > 0 else 0.0

                m = match_len[bucket]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "min_freq": k,
                        "bucket": bucket,
                        "total_instances": total_instances,
                        "covered_instances": covered_instances,
                        "covered_rate": covered_rate,
                        "match_precision": m["precision"],
                        "match_recall": m["recall"],
                        "match_f1": m["f1"],
                    }
                )

    # Console summary: top type improvements between min_freq=1 and 2 (if exists)
    if 1 in candidates and 2 in candidates:
        improv = []
        for t in per_k_type[1].keys():
            c1 = per_k_type[1][t]["covered_instances"] / per_k_type[1][t]["total_instances"]
            c2 = per_k_type[2].get(t, {"covered_instances": 0, "total_instances": 1})
            c2 = c2["covered_instances"] / c2["total_instances"] if c2["total_instances"] > 0 else 0.0
            improv.append((t, c2 - c1, c1, c2))
        improv.sort(key=lambda x: x[1], reverse=True)
        print("\n[Top type coverage gain: k=2 - k=1]")
        for t, delta, c1, c2 in improv[:10]:
            print(f"  {t:<10} delta={delta:+.4f} (k1={c1:.4f} -> k2={c2:.4f})")

    if 2 in candidates and 3 in candidates:
        improv = []
        for t in per_k_type[2].keys():
            c2 = per_k_type[2][t]["covered_instances"] / per_k_type[2][t]["total_instances"]
            c3 = per_k_type[3].get(t, {"covered_instances": 0, "total_instances": 1})
            c3 = c3["covered_instances"] / c3["total_instances"] if c3["total_instances"] > 0 else 0.0
            improv.append((t, c3 - c2, c2, c3))
        improv.sort(key=lambda x: x[1], reverse=True)
        print("\n[Top type coverage gain: k=3 - k=2]")
        for t, delta, c2, c3 in improv[:10]:
            print(f"  {t:<10} delta={delta:+.4f} (k2={c2:.4f} -> k3={c3:.4f})")

    # Console summary: length contribution between k=1 and 2
    if 1 in candidates and 2 in candidates:
        print("\n[Length bucket coverage: k=2 - k=1]")
        for bucket in ["1字", "2字", "3字", "4-5字", "6字+"]:
            c1 = per_k_len[1].get(bucket, {"total_instances": 0, "covered_instances": 0})
            c2 = per_k_len[2].get(bucket, {"total_instances": 0, "covered_instances": 0})
            r1 = c1["covered_instances"] / c1["total_instances"] if c1["total_instances"] > 0 else 0.0
            r2 = c2["covered_instances"] / c2["total_instances"] if c2["total_instances"] > 0 else 0.0
            print(f"  {bucket:<6} covered_rate k1={r1:.4f} k2={r2:.4f} delta={r2-r1:+.4f}")

    print(f"\n[EXP-011] wrote:\n  {type_csv}\n  {len_csv}")


if __name__ == "__main__":
    # Ensure repo root on sys.path when executed directly
    # (Some environments need it for eznlp imports.)
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    main()

