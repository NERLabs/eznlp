#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP-011: 分类别自适应词典（不再输出单一 min_freq）

目标：
  - 对每个数据集（默认 boson/clue），在 train 上统计 (entity_text, type) 的频次
  - 为每个类别 type 选择 k_t ∈ {1,2,3}，使得最终联合词典尽量满足：
      * recall_short >= beta_short * max_recall_short_global
      * recall_long  >= beta_long  * max_recall_long_global
    同时优先最大化 match_precision（并在相同 precision 下偏向更小词典，实现“更高效”）
  - 输出：
      * 每个 type 对应的 k_t（这是“分类别策略”，而非单一阈值）
      * 最终联合词典 auto_lexicon 替代文件（用于训练：--expert_dict_path）

说明：
  - 训练脚本/词典特征构建只接收“词表集合”，不需要词表带 type。
  - 因此我们选择 k_t 后得到的最终词典是：
      Lexicon = ⋃_t { entity_text | count(entity_text, t) >= k_t }
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eznlp.token import LexiconTokenizer


TypedSpan = Tuple[int, int, str]  # (start,end,type) in BMES char indices


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
                raise ValueError(f"Unexpected BMES line format in {path}: {line!r}")
            tok, lab = parts
            tokens.append(tok)
            labels.append(lab)

    if tokens:
        sentences.append({"tokens": tokens, "labels": labels})
    return sentences


def extract_gold_typed_entity_spans(tokens: List[str], labels: List[str]) -> List[TypedSpan]:
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
            raise ValueError(f"Unknown label in BMES file: {lab!r}")

    return spans


def bucket_short_long(span_len: int, long_min_len: int) -> str:
    return "long" if span_len >= long_min_len else "short"


@dataclass
class ProxyStats:
    match_precision: float
    recall_short: float
    recall_long: float
    lexicon_size: int


def proxy_objective(proxy: ProxyStats, *, w_short: float, w_long: float, size_penalty: float) -> float:
    # Dev-guided proxy objective:
    # precision * (weighted short/long recall) - lambda * normalized_size
    weighted_recall = w_short * proxy.recall_short + w_long * proxy.recall_long
    return proxy.match_precision * weighted_recall - size_penalty * (proxy.lexicon_size / 10000.0)


def compute_proxy_stats(
    sentences: List[Dict[str, List[str]]],
    lexicon: Set[str],
    *,
    max_len: int,
    long_min_len: int,
) -> ProxyStats:
    tokenizer = LexiconTokenizer(lexicon, max_len=max_len)

    correct_total = 0
    matched_total = 0

    gold_short_total = 0
    gold_long_total = 0
    tp_short_total = 0
    tp_long_total = 0

    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        text = "".join(tokens)

        typed_spans = extract_gold_typed_entity_spans(tokens, labels)
        gold_map = {(s, e): t for s, e, t in typed_spans}
        gold_spans = set(gold_map.keys())

        for s, e, _t in typed_spans:
            span_len = e - s
            if span_len >= long_min_len:
                gold_long_total += 1
            else:
                gold_short_total += 1

        matched_spans: Set[Tuple[int, int]] = set()
        for _word_text, s, e in tokenizer.tokenize(text):
            matched_spans.add((s, e))

        matched_total += len(matched_spans)
        correct = gold_spans.intersection(matched_spans)
        correct_total += len(correct)

        for s, e in correct:
            span_len = e - s
            if span_len >= long_min_len:
                tp_long_total += 1
            else:
                tp_short_total += 1

    match_precision = correct_total / matched_total if matched_total > 0 else 0.0
    recall_short = tp_short_total / gold_short_total if gold_short_total > 0 else 0.0
    recall_long = tp_long_total / gold_long_total if gold_long_total > 0 else 0.0

    return ProxyStats(
        match_precision=float(match_precision),
        recall_short=float(recall_short),
        recall_long=float(recall_long),
        lexicon_size=len(lexicon),
    )


def select_uniform_maxima(
    sentences: List[Dict[str, List[str]]],
    *,
    candidates: Sequence[int],
    max_len: int,
    long_min_len: int,
) -> Tuple[float, float]:
    """
    计算全局统一 min_freq 下的 max recall_short / max recall_long，
    用于作为约束基准。
    """
    # entity_text 总频次（不区分 type）
    total_counter: Counter[str] = Counter()
    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_entity_spans(tokens, labels)
        for s, e, _t in typed_spans:
            w = "".join(tokens[s:e])
            if w:
                total_counter[w] += 1

    max_short = 0.0
    max_long = 0.0
    for k in candidates:
        lexicon = {w for w, c in total_counter.items() if c >= k}
        proxy = compute_proxy_stats(sentences, lexicon, max_len=max_len, long_min_len=long_min_len)
        max_short = max(max_short, proxy.recall_short)
        max_long = max(max_long, proxy.recall_long)
    return max_short, max_long


def build_per_type_word_sets(
    sentences: List[Dict[str, List[str]]],
    *,
    candidates: Sequence[int],
) -> Tuple[Set[str], Dict[str, Dict[int, Set[str]]], Dict[str, int]]:
    """
    为每个 type 预计算：
      words_by_type_k[type][k] = {word | count(word,type)>=k}
    同时返回：
      all_types
      long_gold_count_by_type[type]（用于决定要不要调参的类型）
    """
    # (word, type) 计数
    ct: Counter[Tuple[str, str]] = Counter()
    long_gold_count_by_type: Dict[str, int] = defaultdict(int)

    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_entity_spans(tokens, labels)
        for s, e, t in typed_spans:
            w = "".join(tokens[s:e])
            if not w:
                continue
            ct[(w, t)] += 1

    all_types: Set[str] = {t for (_w, t) in ct.keys()}

    # 统计 long_gold_count 用于剪枝（long_min_len 由外部传；这里先按字符长度>=6）
    # 注意：这里仅用于“调参类型排序”，不影响 proxy 计算。
    # 后续 proxy 会用 long_min_len 真实约束。
    long_min_len_for_sort = 6
    for sent in sentences:
        tokens = sent["tokens"]
        labels = sent["labels"]
        typed_spans = extract_gold_typed_entity_spans(tokens, labels)
        for s, e, t in typed_spans:
            if (e - s) >= long_min_len_for_sort:
                long_gold_count_by_type[t] += 1

    words_by_type_k: Dict[str, Dict[int, Set[str]]] = {t: {} for t in all_types}
    # prepare counts per type
    counts_by_type: Dict[str, Dict[str, int]] = {t: defaultdict(int) for t in all_types}
    for (w, t), c in ct.items():
        counts_by_type[t][w] = c

    for t in all_types:
        for k in candidates:
            words_by_type_k[t][k] = {w for w, c in counts_by_type[t].items() if c >= k}

    return all_types, words_by_type_k, long_gold_count_by_type


def build_lexicon_from_thresholds(
    types: Iterable[str],
    k_by_type: Dict[str, int],
    words_by_type_k: Dict[str, Dict[int, Set[str]]],
) -> Set[str]:
    lex: Set[str] = set()
    for t in types:
        k = k_by_type[t]
        lex.update(words_by_type_k[t].get(k, set()))
    return lex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="boson,clue", help="逗号分隔")
    parser.add_argument("--candidates", type=str, default="1,2,3")
    parser.add_argument("--beta_short", type=float, default=0.98, help="相对基线（base_k）短实体召回下限比例")
    parser.add_argument("--beta_long", type=float, default=0.98, help="相对基线（base_k）长实体召回下限比例")
    parser.add_argument("--long_min_len", type=int, default=6)
    parser.add_argument("--max_len", type=int, default=10)
    parser.add_argument("--max_iter", type=int, default=3)
    parser.add_argument("--max_tune_types", type=int, default=30, help="调参类别数上限")
    parser.add_argument("--base_k", type=int, default=2, help="阈值词典基线（保底词典）")
    parser.add_argument("--w_short", type=float, default=0.5, help="dev objective 中短实体权重")
    parser.add_argument("--w_long", type=float, default=0.5, help="dev objective 中长实体权重")
    parser.add_argument("--size_penalty", type=float, default=0.0, help="词典规模惩罚系数（越大越偏向小词典）")
    parser.add_argument("--require_beats_base", action="store_true", default=True, help="若未超过基线则回退到 base_k 词典")
    parser.add_argument("--allow_degrade_if_none", action="store_true", default=False, help="若无可行改进时允许保留次优方案")
    parser.add_argument("--out_dir", type=str, default="experiments/EXP-011-lexicon_strategy/analysis")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    candidates = [int(x.strip()) for x in args.candidates.split(",") if x.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path("_2DATA")
    dataset_paths = {
        "redjujube": str(data_root / "RedJujube" / "redjujube_train.bmes"),
        "boson": str(data_root / "boson" / "boson.train.bmes"),
        "clue": str(data_root / "clue" / "train.char.bmes"),
    }
    dataset_dev_paths = {
        "redjujube": str(data_root / "RedJujube" / "redjujube_dev.bmes"),
        "boson": str(data_root / "boson" / "boson.dev.bmes"),
        "clue": str(data_root / "clue" / "dev.char.bmes"),
    }

    for dataset in datasets:
        train_path = dataset_paths[dataset]
        sentences = load_bmes_file(train_path)
        dev_sents = load_bmes_file(dataset_dev_paths[dataset])
        print(f"\n[PerTypeLexicon] dataset={dataset} train_sentences={len(sentences)}")

        all_types, words_by_type_k, long_gold_count_by_type = build_per_type_word_sets(
            sentences,
            candidates=candidates,
        )

        if args.base_k not in candidates:
            raise ValueError(f"base_k={args.base_k} must be in candidates={candidates}")

        # Baseline: uniform base_k per type
        base_k_by_type = {t: args.base_k for t in all_types}
        base_lex = build_lexicon_from_thresholds(all_types, base_k_by_type, words_by_type_k)
        base_train = compute_proxy_stats(
            sentences,
            base_lex,
            max_len=args.max_len,
            long_min_len=args.long_min_len,
        )
        base_dev = compute_proxy_stats(
            dev_sents,
            base_lex,
            max_len=args.max_len,
            long_min_len=args.long_min_len,
        )
        base_score = proxy_objective(
            base_dev, w_short=args.w_short, w_long=args.w_long, size_penalty=args.size_penalty
        )

        short_floor = args.beta_short * base_train.recall_short
        long_floor = args.beta_long * base_train.recall_long
        print(
            f"  base(k={args.base_k}) train: P={base_train.match_precision:.4f} "
            f"Rshort={base_train.recall_short:.4f} Rlong={base_train.recall_long:.4f} |lex|={base_train.lexicon_size}"
        )
        print(
            f"  base(k={args.base_k}) dev:   P={base_dev.match_precision:.4f} "
            f"Rshort={base_dev.recall_short:.4f} Rlong={base_dev.recall_long:.4f} score={base_score:.6f}"
        )
        print(f"  floors(w.r.t base): short>= {short_floor:.4f}, long>= {long_floor:.4f}")

        # tuning set: prioritize long-tail-sensitive types
        tuned_types = sorted(
            list(all_types),
            key=lambda t: long_gold_count_by_type.get(t, 0),
            reverse=True,
        )[: args.max_tune_types]
        # start from baseline k_by_type
        k_by_type = dict(base_k_by_type)

        def eval_current(k_trial: Dict[str, int]) -> Tuple[bool, ProxyStats, ProxyStats, float]:
            lex = build_lexicon_from_thresholds(all_types, k_trial, words_by_type_k)
            proxy_train = compute_proxy_stats(
                sentences,
                lex,
                max_len=args.max_len,
                long_min_len=args.long_min_len,
            )
            proxy_dev = compute_proxy_stats(
                dev_sents,
                lex,
                max_len=args.max_len,
                long_min_len=args.long_min_len,
            )
            score = proxy_objective(
                proxy_dev, w_short=args.w_short, w_long=args.w_long, size_penalty=args.size_penalty
            )
            ok = proxy_train.recall_short >= short_floor and proxy_train.recall_long >= long_floor
            return ok, proxy_train, proxy_dev, score

        ok0, train0, dev0, score0 = eval_current(k_by_type)
        best_ok = ok0
        best_k_by_type = dict(k_by_type)
        best_train = train0
        best_dev = dev0
        best_score = score0
        print(
            f"  init(base) ok={ok0} trainP={train0.match_precision:.4f} "
            f"trainRshort={train0.recall_short:.4f} trainRlong={train0.recall_long:.4f} "
            f"devScore={score0:.6f} |lex|={train0.lexicon_size}"
        )

        # greedy coordinate descent (dev-guided, train-constrained)
        for it in range(args.max_iter):
            changed = False
            print(f"  iter {it+1}/{args.max_iter}")
            for t in tuned_types:
                current_k = k_by_type[t]
                local_choice = None  # (k, ok, train_proxy, dev_proxy, score)

                for k in sorted(candidates):
                    if k == current_k:
                        continue
                    k_tmp = dict(k_by_type)
                    k_tmp[t] = k
                    ok, p_tr, p_dev, s = eval_current(k_tmp)

                    cand = (k, ok, p_tr, p_dev, s)
                    if local_choice is None:
                        local_choice = cand
                    else:
                        _, ok0_l, p_tr0, _p_dev0, s0 = local_choice
                        # feasible first; then higher dev score; tie smaller lexicon
                        better = False
                        if ok and not ok0_l:
                            better = True
                        elif ok == ok0_l:
                            if s > s0:
                                better = True
                            elif s == s0 and p_tr.lexicon_size < p_tr0.lexicon_size:
                                better = True
                        if better:
                            local_choice = cand

                if local_choice is None:
                    continue
                best_local_k, _ok_l, _p_tr_l, _p_dev_l, _s_l = local_choice

                # apply if improves global objective
                if best_local_k != current_k:
                    k_by_type[t] = best_local_k
                    ok_new, p_tr_new, p_dev_new, s_new = eval_current(k_by_type)

                    improved = False
                    if ok_new and (not best_ok):
                        improved = True
                    elif ok_new == best_ok:
                        if s_new > best_score:
                            improved = True
                        elif s_new == best_score and p_tr_new.lexicon_size < best_train.lexicon_size:
                            improved = True
                    elif (not ok_new) and (not best_ok) and args.allow_degrade_if_none and s_new > best_score:
                        improved = True

                    if improved:
                        best_ok = ok_new
                        best_train = p_tr_new
                        best_dev = p_dev_new
                        best_score = s_new
                        best_k_by_type = dict(k_by_type)
                        changed = True
                    else:
                        k_by_type[t] = current_k

            if not changed:
                break

        final_lex = build_lexicon_from_thresholds(all_types, best_k_by_type, words_by_type_k)
        final_train = best_train
        final_dev = best_dev
        final_score = best_score

        # Hard safeguard: do not output a strategy worse than base objective.
        if args.require_beats_base and final_score < base_score:
            print("  fallback to base lexicon (final_score < base_score)")
            final_lex = set(base_lex)
            best_k_by_type = dict(base_k_by_type)
            final_train = base_train
            final_dev = base_dev
            final_score = base_score

        print(
            f"  final train: ok={best_ok} P={final_train.match_precision:.4f} "
            f"Rshort={final_train.recall_short:.4f} Rlong={final_train.recall_long:.4f} |lex|={final_train.lexicon_size}"
        )
        print(
            f"  final dev:   P={final_dev.match_precision:.4f} "
            f"Rshort={final_dev.recall_short:.4f} Rlong={final_dev.recall_long:.4f} score={final_score:.6f}"
        )

        # output files
        thresholds_csv = out_dir / f"per_type_thresholds_{dataset}.csv"
        with open(thresholds_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "type", "k_t", "long_gold_count_approx"])
            writer.writeheader()
            for t in sorted(all_types):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "type": t,
                        "k_t": best_k_by_type[t],
                        "long_gold_count_approx": long_gold_count_by_type.get(t, 0),
                    }
                )

        lex_path = out_dir / f"per_type_lexicon_{dataset}.txt"
        with open(lex_path, "w", encoding="utf-8") as f:
            for w in sorted(final_lex):
                f.write(w + "\n")

        # proxy log
        proxy_log = out_dir / f"per_type_lexicon_proxy_{dataset}.txt"
        proxy_log.write_text(
            "\n".join(
                [
                    f"dataset={dataset}",
                    f"beta_short={args.beta_short} beta_long={args.beta_long}",
                    f"base_k={args.base_k}",
                    f"w_short={args.w_short} w_long={args.w_long}",
                    f"size_penalty={args.size_penalty}",
                    f"long_min_len={args.long_min_len}",
                    f"max_len={args.max_len}",
                    f"ok={best_ok}",
                    f"base_dev_score={base_score}",
                    f"final_dev_score={final_score}",
                    f"base_train_P={base_train.match_precision}",
                    f"base_train_Rshort={base_train.recall_short}",
                    f"base_train_Rlong={base_train.recall_long}",
                    f"base_train_lex_size={base_train.lexicon_size}",
                    f"final_train_P={final_train.match_precision}",
                    f"final_train_Rshort={final_train.recall_short}",
                    f"final_train_Rlong={final_train.recall_long}",
                    f"final_train_lex_size={final_train.lexicon_size}",
                ]
            ),
            encoding="utf-8",
        )

        print(f"  wrote: {thresholds_csv}")
        print(f"  wrote: {lex_path}")
        print(f"  wrote: {proxy_log}")


if __name__ == "__main__":
    main()

