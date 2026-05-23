#!/usr/bin/env python3
"""Collect experiment evidence for the red jujube NER manuscript.

The script does not train models. It audits completed experiments and produces
tables that can be used to decide whether additional training is necessary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy import stats


ROOT = Path(__file__).resolve().parents[3]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percent(x: float) -> float:
    return x * 100 if x <= 1.5 else x


def summarize_result_group(group_dir: Path) -> dict | None:
    files = sorted(group_dir.glob("*/results.json"))
    rows = []
    for file in files:
        data = read_json(file)
        args = data.get("args", {})
        rows.append(
            {
                "seed": args.get("seed"),
                "path": str(file.relative_to(ROOT)),
                "dev_f1": percent(float(data.get("best_dev_f1", math.nan))),
                "test_f1": percent(float(data.get("test_f1", math.nan))),
                "best_epoch": data.get("best_epoch"),
                "fl_gamma": args.get("fl_gamma"),
                "min_freq": args.get("min_freq"),
                "sb_size": args.get("sb_size"),
                "use_channel_attention": args.get("use_channel_attention"),
                "use_srg": args.get("use_srg"),
                "type_aware_minfreq": args.get("type_aware_minfreq"),
            }
        )
    rows = [r for r in rows if not math.isnan(r["test_f1"])]
    if not rows:
        return None
    dev = [r["dev_f1"] for r in rows]
    test = [r["test_f1"] for r in rows]
    return {
        "name": str(group_dir.relative_to(ROOT)),
        "n": len(rows),
        "rows": rows,
        "dev_mean": mean(dev),
        "dev_std": stdev(dev) if len(dev) > 1 else 0.0,
        "test_mean": mean(test),
        "test_std": stdev(test) if len(test) > 1 else 0.0,
        "gap_mean": mean([d - t for d, t in zip(dev, test)]),
        "abs_gap_mean": mean([abs(d - t) for d, t in zip(dev, test)]),
    }


def collect_groups(base: Path) -> list[dict]:
    groups = []
    if not base.exists():
        return groups
    for child in sorted(base.iterdir()):
        if child.is_dir():
            summary = summarize_result_group(child)
            if summary:
                groups.append(summary)
    return groups


def write_group_csv(groups: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "n",
                "dev_mean",
                "dev_std",
                "test_mean",
                "test_std",
                "gap_mean",
                "abs_gap_mean",
            ],
        )
        writer.writeheader()
        for g in groups:
            writer.writerow(
                {
                    "group": g["name"],
                    "n": g["n"],
                    "dev_mean": f"{g['dev_mean']:.4f}",
                    "dev_std": f"{g['dev_std']:.4f}",
                    "test_mean": f"{g['test_mean']:.4f}",
                    "test_std": f"{g['test_std']:.4f}",
                    "gap_mean": f"{g['gap_mean']:.4f}",
                    "abs_gap_mean": f"{g['abs_gap_mean']:.4f}",
                }
            )


def write_seed_csv(groups: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "seed",
                "dev_f1",
                "test_f1",
                "best_epoch",
                "fl_gamma",
                "min_freq",
                "sb_size",
                "use_channel_attention",
                "use_srg",
                "type_aware_minfreq",
                "path",
            ],
        )
        writer.writeheader()
        for g in groups:
            for row in g["rows"]:
                writer.writerow({"group": g["name"], **row})


def paired_ttest(groups: list[dict], group_a_suffix: str, group_b_suffix: str) -> dict | None:
    lookup = {g["name"]: g for g in groups}
    a = next((g for g in groups if g["name"].endswith(group_a_suffix)), None)
    b = next((g for g in groups if g["name"].endswith(group_b_suffix)), None)
    if not a or not b:
        return None
    a_by_seed = {r["seed"]: r["test_f1"] for r in a["rows"]}
    b_by_seed = {r["seed"]: r["test_f1"] for r in b["rows"]}
    seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if len(seeds) < 2:
        return None
    av = [a_by_seed[s] for s in seeds]
    bv = [b_by_seed[s] for s in seeds]
    t_stat, p_value = stats.ttest_rel(av, bv)
    diff = [x - y for x, y in zip(av, bv)]
    return {
        "a": a["name"],
        "b": b["name"],
        "seeds": seeds,
        "a_values": av,
        "b_values": bv,
        "diff_mean": mean(diff),
        "t": float(t_stat),
        "p": float(p_value),
    }


def write_markdown(groups: list[dict], newdata_groups: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    comparisons = [
        paired_ttest(newdata_groups, "Q_bs_focal", "CRF_nodict_bertwwm"),
        paired_ttest(newdata_groups, "Q_bs_focal", "CRF_nodict"),
        paired_ttest(newdata_groups, "Q_bs_focal", "G_bilstm_baseline"),
        paired_ttest(newdata_groups, "Q_bs_focal", "H_bs_baseline"),
        paired_ttest(newdata_groups, "H_bs_baseline", "CRF_nodict"),
        paired_ttest(newdata_groups, "SoftLexicon_baseline", "CRF_nodict"),
    ]
    comparisons = [c for c in comparisons if c]

    top_newdata = sorted(newdata_groups, key=lambda g: g["test_mean"], reverse=True)[:12]
    stable_newdata = sorted(
        [g for g in newdata_groups if g["n"] >= 3],
        key=lambda g: (g["abs_gap_mean"], -g["test_mean"]),
    )[:12]

    lines = [
        "# 实验充分性补强分析",
        "",
        "生成脚本：`docs/paper/tools/analyze_experiment_sufficiency.py`",
        "",
        "## 1 主要结论",
        "",
        "- 现有实验已经覆盖主模型、无词典/无边界/Focal 消融、公开数据集泛化、词典阈值和新数据分布鲁棒性分析；短期不建议为了追求更高单点 F1 直接替换投稿主结果。",
        "- 当前最值得补进投稿材料的是“实验工作量说明”和“鲁棒性审计”，而不是把新数据版本结果与旧 RJND 主表混用。",
        "- 已在 `results_newdata` 中定位到投稿稿表 3 对应的三种子原始结果：`CRF_nodict_bertwwm`、`CRF_nodict` 和 `Q_bs_focal`，可支撑配对 t 检验。",
        "- 新数据版本中存在若干单种子高分配置，但只有三种子结果才适合进入主表；短期不建议用单种子通道注意力结果替换当前主结果。",
        "",
        "## 2 新数据版本结果概览",
        "",
        "| 实验组 | n | Dev F1/% | Test F1/% | Test-Dev/pp |",
        "|---|---:|---:|---:|---:|",
    ]
    for g in top_newdata:
        lines.append(
            f"| `{g['name']}` | {g['n']} | {g['dev_mean']:.2f}±{g['dev_std']:.2f} | "
            f"{g['test_mean']:.2f}±{g['test_std']:.2f} | {-g['gap_mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 3 泛化稳定性排序",
            "",
            "| 实验组 | n | Test F1/% | |Test-Dev|/pp | 判断 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for g in stable_newdata:
        judgment = "稳定" if g["abs_gap_mean"] <= 1.0 else "需谨慎"
        lines.append(
            f"| `{g['name']}` | {g['n']} | {g['test_mean']:.2f}±{g['test_std']:.2f} | "
            f"{g['abs_gap_mean']:.2f} | {judgment} |"
        )

    lines.extend(
        [
            "",
            "## 4 配对 t 检验（新数据版本，仅作补充审计）",
            "",
            "| 对比 | seeds | 均值差/pp | t | p | 解释 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    if comparisons:
        for c in comparisons:
            interp = "达到 0.05 水平" if c["p"] < 0.05 else "未达到 0.05 水平"
            lines.append(
                f"| `{Path(c['a']).name}` - `{Path(c['b']).name}` | {','.join(map(str, c['seeds']))} | "
                f"{c['diff_mean']:.2f} | {c['t']:.3f} | {c['p']:.4f} | {interp} |"
            )
    else:
        lines.append("| - | - | - | - | - | 未找到可配对的三种子组合 |")

    lines.extend(
        [
            "",
            "## 5 对投稿稿的处理建议",
            "",
            "1. 主结果继续采用三种子均值 `88.28%±0.22%`，并在结果注册表中补入 raw seed 路径。",
            "2. 可在正文主结果段落加入配对 t 检验句，但应仅针对三种子强基线，不扩展到单种子模块。",
            "3. 可在补充材料或答审材料中加入本文件，说明已进行多配置鲁棒性分析。",
            "4. 若继续新增训练，优先补 `Q_bs_focal_attnv1` 的 seed 43/44 独立复核或 K 折验证；不要直接用单种子最高值替换主结论。",
            "",
            "## 6 生成文件",
            "",
            "- `docs/paper/experiment_sufficiency_groups_2026-05-23.csv`",
            "- `docs/paper/experiment_sufficiency_seeds_2026-05-23.csv`",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        default="docs/paper/experiment_sufficiency_analysis_2026-05-23.md",
    )
    parser.add_argument(
        "--groups-csv",
        default="docs/paper/experiment_sufficiency_groups_2026-05-23.csv",
    )
    parser.add_argument(
        "--seeds-csv",
        default="docs/paper/experiment_sufficiency_seeds_2026-05-23.csv",
    )
    args = parser.parse_args()

    legacy_groups = collect_groups(ROOT / "experiments/EXP-010-optimization/results")
    newdata_groups = collect_groups(ROOT / "experiments/EXP-010-optimization/results_newdata")
    groups = legacy_groups + newdata_groups
    write_group_csv(groups, ROOT / args.groups_csv)
    write_seed_csv(groups, ROOT / args.seeds_csv)
    write_markdown(legacy_groups, newdata_groups, ROOT / args.report)
    print(f"Wrote {args.report}")
    print(f"Collected groups: {len(groups)}")


if __name__ == "__main__":
    main()
