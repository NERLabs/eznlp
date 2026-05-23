#!/usr/bin/env python3
"""Validate reproducibility paths referenced by the paper handoff notes."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_PATHS = [
    "Taskfile.yml",
    "_5TRAIN/train_redjujube_ner.py",
    "_5TRAIN/train_redjujube_expert_boundary.py",
    "_5TRAIN/train_general_expert_boundary.py",
    "_1CONFIG/redjujube/run_public_all_sequential.sh",
    "_1CONFIG/redjujube/run_bs_optimization_experiments.sh",
    "_2DATA/RedJujube/redjujube_train.bmes",
    "_2DATA/RedJujube/redjujube_dev.bmes",
    "_2DATA/RedJujube/redjujube_test.bmes",
    "_2DATA/RedJujube/expert_lexicon_auto_min1.txt",
    "_2DATA/MSRA",
    "_2DATA/WeiboNER",
    "_2DATA/ResumeNER",
    "_2DATA/boson",
    "_2DATA/clue",
    "_3DATA_PROCESS/extract_lexicon_from_training.py",
    "eznlp/model/encoder.py",
    "eznlp/model/nested_embedder.py",
    "eznlp/model/decoder/boundary_selection.py",
    "docs/paper/基于词典和边界预测的红枣栽培命名实体识别.md",
    "docs/paper/paper_result_registry.md",
    "experiments/EXP-011-lexicon_strategy/analysis/candidate_proxy_table.csv",
    "experiments/EXP-010-optimization/QUICK_SUMMARY_ZH.txt",
]

REQUIRED_RESULT_DIRS = [
    "experiments/EXP-010-optimization/results/Q_bs_focal",
    "experiments/EXP-010-optimization/results_public/msra_bs_dict_focal",
    "experiments/EXP-010-optimization/results_public/weibo_bs_dict_focal",
    "experiments/EXP-010-optimization/results_public/resume_bs_dict_focal",
    "experiments/EXP-010-optimization/results_public/boson_bs_dict_focal",
    "experiments/EXP-010-optimization/results_public/clue_bs_dict_focal",
]

REQUIRED_TEXT = [
    "88.28%±0.22%",
    "89.51/87.58/88.54",
    "5 317",
    "results_newdata",
    "hfl/chinese-macbert-base",
]


def _ok(results: list[str], message: str) -> None:
    results.append(f"- PASS: {message}")


def _fail(results: list[str], message: str) -> None:
    results.append(f"- FAIL: {message}")


def validate(root: Path, statement: Path) -> tuple[bool, str]:
    results: list[str] = []
    success = True

    if statement.exists() and statement.stat().st_size > 0:
        _ok(results, f"reproducibility statement exists: {statement}")
        text = statement.read_text(encoding="utf-8")
    else:
        _fail(results, f"missing reproducibility statement: {statement}")
        return False, "# Reproducibility Path Validation\n\n" + "\n".join(results) + "\n"

    for rel in REQUIRED_PATHS:
        path = root / rel
        if path.exists():
            _ok(results, f"path exists: {rel}")
        else:
            _fail(results, f"missing path: {rel}")
            success = False

    for rel in REQUIRED_RESULT_DIRS:
        path = root / rel
        result_files = list(path.rglob("results.json")) if path.exists() else []
        if result_files:
            _ok(results, f"result directory has results.json: {rel}")
        else:
            _fail(results, f"result directory lacks results.json: {rel}")
            success = False

    for needle in REQUIRED_TEXT:
        if needle in text:
            _ok(results, f"statement contains: {needle}")
        else:
            _fail(results, f"statement missing: {needle}")
            success = False

    report = ["# Reproducibility Path Validation", ""]
    report.extend(results)
    return success, "\n".join(report) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--statement",
        type=Path,
        default=Path("docs/paper/reproducibility_statement.md"),
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    success, report = validate(args.root, args.statement)
    if args.report:
        args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
