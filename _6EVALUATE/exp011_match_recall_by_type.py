#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-011: 词典匹配代理（match_recall）按类别/按长实体分桶

在数据集 train 上：
  对每个 min_freq k：
    1) 构建 lexicon（来自 gold entity_text 频次 >= k）
    2) 用 LexiconTokenizer(lexicon, max_len=10) 对句子 raw_text 做最大正向匹配得到 matched spans
    3) 对每个 matched span，如果边界与某个 gold span 完全一致，则计为 tp，并归入该 gold 的实体类型

输出：
  - 每个 type 的 match_recall_proxy（tp_by_type / gold_by_type）
  - 以及仅在 6字+ 长度分桶上的 match_recall_proxy（tp_long_by_type / gold_long_by_type）
  - 对 k=1->2, 2->3 的 recall 变化 top 列表
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Ensure repo root on sys.path when running directly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eznlp.token import LexiconTokenizer


EntitySpan = Tuple[int, int]  # (start, end)
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
            raise ValueError(f"Unknown BMES label: {lab!r}")
    return spans


def build_lexicon_from_train(train_sentences: List[Dict[str, List[str]]], min_freq: int) -> Set[str]:
    counter: Counter[str] = Counter()
    for sent in train_sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        spans = extract_gold_typed_spans(tokens, labels)
        for s, e, _t in spans:
            w = "".join(tokens[s:e])
            if w:
                counter[w] += 1
    return {w for w, c in counter.items() if c >= min_freq}


def bucket_6plus(length: int) -> bool:
    return length >= 6


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

    if dataset == "redjujube":
        train_path = "_2DATA/RedJujube/redjujube_train.bmes"
    elif dataset == "boson":
        train_path = "_2DATA/boson/boson.train.bmes"
    else:
        train_path = "_2DATA/clue/train.char.bmes"

    train_sents = load_bmes_file(train_path)
    print(f"[EXP-011] dataset={dataset}, train_sentences={len(train_sents)}")

    per_k_gold_by_type: Dict[int, Dict[str, int]] = {}
    per_k_tp_by_type: Dict[int, Dict[str, int]] = {}
    per_k_gold_by_type_long: Dict[int, Dict[str, int]] = {}
    per_k_tp_by_type_long: Dict[int, Dict[str, int]] = {}

    # overall matched counts
    per_k_total_tp: Dict[int, int] = {}
    per_k_total_matched: Dict[int, int] = {}

    for k in candidates:
        lexicon = build_lexicon_from_train(train_sents, min_freq=k)
        tokenizer = LexiconTokenizer(lexicon, max_len=max_len)

        gold_by_type = defaultdict(int)
        tp_by_type = defaultdict(int)
        gold_by_type_long = defaultdict(int)
        tp_by_type_long = defaultdict(int)

        total_tp = 0
        total_matched = 0

        for sent in train_sents:
            tokens = sent["tokens"]
            labels = sent["labels"]
            typed_spans = extract_gold_typed_spans(tokens, labels)

            gold_map: Dict[EntitySpan, str] = {(s, e): t for s, e, t in typed_spans}
            for s, e, t in typed_spans:
                gold_by_type[t] += 1
                if bucket_6plus(e - s):
                    gold_by_type_long[t] += 1

            text = "".join(tokens)
            matched_spans: List[EntitySpan] = [(s, e) for _w, s, e in tokenizer.tokenize(text)]
            total_matched += len(matched_spans)

            for s, e in matched_spans:
                span = (s, e)
                if span in gold_map:
                    t = gold_map[span]
                    tp_by_type[t] += 1
                    total_tp += 1
                    if bucket_6plus(e - s):
                        tp_by_type_long[t] += 1

        per_k_gold_by_type[k] = dict(gold_by_type)
        per_k_tp_by_type[k] = dict(tp_by_type)
        per_k_gold_by_type_long[k] = dict(gold_by_type_long)
        per_k_tp_by_type_long[k] = dict(tp_by_type_long)
        per_k_total_tp[k] = total_tp
        per_k_total_matched[k] = total_matched

        match_precision = total_tp / total_matched if total_matched > 0 else 0.0
        match_recall = total_tp / sum(gold_by_type.values()) if sum(gold_by_type.values()) > 0 else 0.0
        print(f"  k={k}: |lexicon|={len(lexicon)} match_precision={match_precision:.4f} match_recall={match_recall:.4f}")

    # compute recall tables
    for mode, gold_dict, tp_dict in [
        ("all", per_k_gold_by_type, per_k_tp_by_type),
        ("6plus", per_k_gold_by_type_long, per_k_tp_by_type_long),
    ]:
        csv_path = out_dir / f"{dataset}_match_recall_by_type_{mode}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["dataset", "min_freq", "type", "gold", "tp", "match_recall"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for k in candidates:
                golds = gold_dict[k]
                tps = tp_dict[k]
                for t in sorted(golds.keys()):
                    gold = golds.get(t, 0)
                    tp = tps.get(t, 0)
                    r = tp / gold if gold > 0 else 0.0
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "min_freq": k,
                            "type": t,
                            "gold": gold,
                            "tp": tp,
                            "match_recall": r,
                        }
                    )

    # Console summary for k=2 - k=1
    if 1 in candidates and 2 in candidates:
        # use all-mode recall
        gold1 = per_k_gold_by_type[1]
        gold2 = per_k_gold_by_type[2]
        types = sorted(set(gold1.keys()) | set(gold2.keys()))
        diffs = []
        for t in types:
            r1 = per_k_tp_by_type[1].get(t, 0) / gold1.get(t, 1) if gold1.get(t, 0) > 0 else 0.0
            r2 = per_k_tp_by_type[2].get(t, 0) / gold2.get(t, 1) if gold2.get(t, 0) > 0 else 0.0
            diffs.append((t, r2 - r1, r1, r2))
        diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        print("\n[Type match_recall delta: k=2 - k=1] top10 by abs")
        for t, d, r1, r2 in diffs[:10]:
            print(f"  {t:<10} delta={d:+.4f} (k1={r1:.4f} -> k2={r2:.4f})")

    print(f"\n[EXP-011] wrote CSVs under: {out_dir}")


if __name__ == "__main__":
    main()

