#!/usr/bin/env python3
import json, glob, statistics
from collections import defaultdict


def collect(pattern, suffix):
    data = defaultdict(list)
    for p in sorted(glob.glob(pattern)):
        parts = p.split("/")
        ds = parts[3].replace(suffix, "")
        try:
            d = json.load(open(p))
            f1 = d.get("test_f1", 0)
            if not f1 and isinstance(d.get("test_metrics"), dict):
                f1 = d["test_metrics"].get("f1", 0)
            if f1 > 0.3:
                data[ds].append(f1)
        except Exception:
            pass
    return data


crf = collect(
    "experiments/EXP-010-optimization/results_public/*_crf_baseline/seed_*/*/results.json",
    "_crf_baseline",
)
# MSRA 的 CRF baseline 存在 msra_pure_baseline
msra_crf = collect(
    "experiments/EXP-010-optimization/results_public/msra_pure_baseline/seed_*/*/results.json",
    "_pure_baseline",
)
if "msra" in msra_crf:
    crf["msra"] = msra_crf["msra"]
edbs = collect(
    "experiments/EXP-010-optimization/results_public/*_bs_dict_focal/seed_*/*/results.json",
    "_bs_dict_focal",
)

order = ["msra", "peopledaily", "resume", "boson", "weibo", "clue"]
NAME = {
    "msra": "MSRA",
    "peopledaily": "PeopleDaily",
    "resume": "ResumeNER",
    "boson": "Boson",
    "weibo": "WeiboNER",
    "clue": "CLUENER",
}

print("=" * 82)
print(f'{"数据集":<14}{"BERT-BiLSTM-CRF":<26}{"EDBS (我们)":<26}{"Δ (%)":<10}')
print("-" * 82)
for ds in order:
    cf = sorted(crf.get(ds, []))
    ef = sorted(edbs.get(ds, []))
    cm = statistics.mean(cf) * 100 if cf else 0
    em = statistics.mean(ef) * 100 if ef else 0
    cs = statistics.stdev(cf) * 100 if len(cf) >= 2 else 0
    es = statistics.stdev(ef) * 100 if len(ef) >= 2 else 0
    delta = em - cm if (cf and ef) else 0
    sign = "+" if delta >= 0 else ""
    name = NAME[ds]
    cstr = f"{cm:.2f}±{cs:.2f}(n={len(cf)})"
    estr = f"{em:.2f}±{es:.2f}(n={len(ef)})"
    print(f"{name:<14}{cstr:<26}{estr:<26}{sign}{delta:.2f}")
print("=" * 82)

print("\n详细 seed 结果:")
for ds in order:
    cf = sorted(crf.get(ds, []))
    ef = sorted(edbs.get(ds, []))
    print(
        f"  {NAME[ds]:<13}  CRF: [{', '.join(f'{x*100:.2f}' for x in cf)}]"
        f"   EDBS: [{', '.join(f'{x*100:.2f}' for x in ef)}]"
    )
