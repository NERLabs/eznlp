#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRA FLAT 模型测试脚本（用于测试 cache/flat_msra_expert_bert/...）

- 加载训练时保存的 config.pth（包含 args / char_vocab / label_vocab）
- 加载 best_model.pth（完整模型对象）
- 按 BMES MSRA 格式读取 test.char.bmes
- 重建 FLAT 输入特征，进行实体级评估并打印各类 P/R/F1
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# 添加项目根目录，导入内部模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "_8TOOL"))

from _4MODELS.models.flat_data_processor import FLATDataProcessor, load_word_list
from _4MODELS.models.flat_extractor import FLATModel  # 仅为类型提示，实际直接 load
from _5TRAIN.train_flat_complete import extract_entities  # 复用 BMES 实体抽取逻辑
from eznlp.metrics import precision_recall_f1_report


def load_bmes_data(file_path):
    """复制 train_flat_complete.py 里的 BMES 加载逻辑（只用 test 集）"""
    data = []
    chars = []
    labels = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if chars:
                    data.append({"chars": chars, "labels": labels})
                    chars = []
                    labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    chars.append(parts[0])
                    labels.append(parts[1])

    if chars:
        data.append({"chars": chars, "labels": labels})

    return data


def evaluate_with_report(
    model,
    dataloader,
    device,
    bert_model=None,
    bert_tokenizer=None,
    use_bert=False,
    idx2label=None,
):
    """
    基于 train_flat_complete.py 的 evaluate 实现，返回：
    - loss
    - 每个实体类型的 P/R/F1
    - macro / micro 的 P/R/F1（实体级）
    """
    model.eval()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            lattice = batch["lattice"].to(device)
            pos_s = batch["pos_s"].to(device)
            pos_e = batch["pos_e"].to(device)
            seq_len = batch["seq_len"].to(device)
            lex_num = batch["lex_num"].to(device)
            target = batch["target"].to(device)

            # 提取 BERT embeddings（与 train_flat_complete 保持一致）
            bert_embed = None
            if use_bert and bert_model is not None:
                chars_list = batch["chars"]
                batch_size = len(chars_list)
                max_char_len = max(len(chars) for chars in chars_list)

                bert_embeds = []
                for chars in chars_list:
                    text = "".join(chars)
                    inputs = bert_tokenizer(
                        text,
                        return_tensors="pt",
                        add_special_tokens=False,
                        truncation=True,
                        max_length=len(chars),
                    ).to(device)
                    outputs = bert_model(**inputs)
                    char_embeds = outputs.last_hidden_state[0]

                    if len(char_embeds) != len(chars):
                        aligned = []
                        token_per_char = len(char_embeds) / len(chars)
                        for i in range(len(chars)):
                            start_idx = int(i * token_per_char)
                            end_idx = int((i + 1) * token_per_char)
                            if end_idx > start_idx:
                                aligned.append(char_embeds[start_idx:end_idx].mean(0))
                            else:
                                aligned.append(char_embeds[start_idx])
                        char_embeds = torch.stack(aligned)

                    bert_embeds.append(char_embeds[: len(chars)])

                bert_embed = torch.zeros(batch_size, max_char_len, 768, device=device)
                for i, emb in enumerate(bert_embeds):
                    bert_embed[i, : len(emb)] = emb

            # 先算 loss
            model.train()
            output = model(lattice, seq_len, lex_num, pos_s, pos_e, target, bert_embed=bert_embed)
            total_loss += output["loss"].item()
            num_batches += 1

            # 再用 eval 模式算预测
            model.eval()
            output = model(lattice, seq_len, lex_num, pos_s, pos_e, bert_embed=bert_embed)
            preds = output["pred"]

            for i, (pred, length) in enumerate(zip(preds, seq_len)):
                all_preds.append(pred[: length.item()])
                all_targets.append(target[i, : length.item()].cpu().tolist())

    # 转成实体级 span，和 eznlp 的 ER 评估对齐
    set_y_pred = []
    set_y_gold = []
    for pred, gold in zip(all_preds, all_targets):
        pred_entities_raw = list(extract_entities(pred, idx2label))  # (start, end, type)
        gold_entities_raw = list(extract_entities(gold, idx2label))  # (start, end, type)

        # 转成 (type, start, end)，匹配 precision_recall_f1_report 的期望格式
        pred_entities = [(t, s, e) for (s, e, t) in pred_entities_raw]
        gold_entities = [(t, s, e) for (s, e, t) in gold_entities_raw]

        set_y_pred.append(pred_entities)
        set_y_gold.append(gold_entities)

    # 计算每类 + macro/micro（实体级）
    scores, ave_scores = precision_recall_f1_report(
        set_y_gold, set_y_pred, macro_over="types"
    )
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        "loss": avg_loss,
        "scores": scores,
        "macro": ave_scores["macro"],
        "micro": ave_scores["micro"],
    }


def build_processor_from_config(config_dict: dict, override_word_file: str = None):
    """
    用 config.pth 里的信息重建 FLATDataProcessor，并恢复 char/label 词表。
    """
    args_dict = config_dict.get("args", {})
    char_vocab = config_dict["char_vocab"]
    label_vocab = config_dict["label_vocab"]

    word_file = override_word_file or args_dict.get("word_file", "assets/vectors/ctb.50d.vec")
    max_seq_len = args_dict.get("max_seq_len", 256)

    # 加载词表并创建 processor
    word_list = load_word_list(word_file)
    processor = FLATDataProcessor(word_list, max_seq_len=max_seq_len)

    # 恢复字符词表
    processor.char_vocab = char_vocab
    processor.idx2char = {idx: ch for ch, idx in char_vocab.items()}

    # 恢复标签词表
    processor.label_vocab = label_vocab
    processor.idx2label = {idx: lab for lab, idx in label_vocab.items()}

    return processor, args_dict


def process_data_for_flat(processor: FLATDataProcessor, data_list):
    """
    按 train_flat_complete.py 的方式把 BMES 数据转成 FLAT 输入特征。
    """
    processed = []
    for data in data_list:
        result = processor.process_sentence(data["chars"], data["labels"])
        # 标签转索引（使用保存下来的 label_vocab）
        result["target"] = [
            processor.label_vocab.get(l, 0) for l in data["labels"][: result["seq_len"]]
        ]
        # 字符 / 词汇转索引（使用保存下来的 char_vocab）
        result["lattice_ids"] = []
        for item in result["lattice"]:
            if len(item) == 1:
                result["lattice_ids"].append(processor.char_vocab.get(item, 1))
            else:
                # 对于词汇，使用第一个字符的索引（与训练脚本保持一致）
                result["lattice_ids"].append(processor.char_vocab.get(item[0], 1))
        processed.append(result)
    return processed


def build_test_loader(processor: FLATDataProcessor, test_data, batch_size: int):
    """
    构建与训练脚本一致的 SimpleDataset + collate_fn + DataLoader（只需要 test_loader）。
    """
    test_processed = process_data_for_flat(processor, test_data)

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    def collate_fn(batch):
        batch_size = len(batch)
        max_seq_len = max(d["seq_len"] for d in batch)
        max_lattice_len = max(d["seq_len"] + d["lex_num"] for d in batch)

        lattice_ids = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_s = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_e = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        seq_len = torch.zeros(batch_size, dtype=torch.long)
        lex_num = torch.zeros(batch_size, dtype=torch.long)
        target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)

        for i, d in enumerate(batch):
            cur_len = d["seq_len"] + d["lex_num"]
            lattice_ids[i, :cur_len] = torch.tensor(d["lattice_ids"][:cur_len])
            pos_s[i, :cur_len] = torch.tensor(d["pos_s"][:cur_len])
            pos_e[i, :cur_len] = torch.tensor(d["pos_e"][:cur_len])
            seq_len[i] = d["seq_len"]
            lex_num[i] = d["lex_num"]
            target[i, : d["seq_len"]] = torch.tensor(d["target"])

        return {
            "lattice": lattice_ids,
            "pos_s": pos_s,
            "pos_e": pos_e,
            "seq_len": seq_len,
            "lex_num": lex_num,
            "target": target,
            "chars": [d["chars"] for d in batch],  # BERT 用
        }

    test_dataset = SimpleDataset(test_processed)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="测试 FLAT-MSRA 实验（best_model.pth + config.pth）")
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="训练输出目录（包含 best_model.pth / config.pth）",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/MSRA",
        help="MSRA 数据目录（包含 train.char.bmes / dev.char.bmes / test.char.bmes）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="测试时 batch size（可与训练不同）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备：cuda 或 cpu",
    )
    parser.add_argument(
        "--no_bert",
        action="store_true",
        help="忽略 config 里的 use_bert，强制不加载 BERT",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. 加载 config.pth（包含 args/词表）
    config_path = os.path.join(args.save_dir, "config.pth")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 config.pth: {config_path}")
    config_dict = torch.load(config_path, map_location=device, weights_only=False)

    processor, train_args = build_processor_from_config(config_dict)
    use_bert = bool(train_args.get("use_bert", False)) and not args.no_bert
    bert_model_name = train_args.get("bert_model", "bert-base-chinese")

    # 2. 加载 MSRA 测试集（BMES）
    test_file = os.path.join(args.data_dir, "test.char.bmes")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"未找到 MSRA 测试文件: {test_file}")
    print(f"加载测试集: {test_file}")
    test_data = load_bmes_data(test_file)
    print(f"  测试样本数: {len(test_data)}")

    # 3. 构建 test_loader
    test_loader = build_test_loader(processor, test_data, args.batch_size)

    # 4. 初始化 BERT（如果训练时用了 BERT 且未禁用）
    bert_model = None
    bert_tokenizer = None
    if use_bert:
        print("\n初始化 BERT 用于测试...")
        from transformers import BertModel, BertTokenizer

        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        bert_model = BertModel.from_pretrained(bert_model_name)
        bert_model.eval()
        bert_model.to(device)
        print(f"  BERT 模型: {bert_model_name}")

    # 5. 加载 best_model.pth（完整模型对象）
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"未找到 best_model.pth: {best_model_path}")
    print(f"\n加载最佳模型: {best_model_path}")
    model: FLATModel = torch.load(best_model_path, map_location=device, weights_only=False)
    model.to(device)
    model.eval()

    # ===== 打印模型结构 =====
    print("\n===== 模型结构 (FLAT-MSRA) =====")
    print(model)

    # 6. 调用带详细报告的 evaluate_with_report（实体级）
    print("\n===== 使用 FLAT 评估 MSRA 测试集 =====")
    metrics = evaluate_with_report(
        model,
        test_loader,
        device,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        use_bert=use_bert,
        idx2label=processor.idx2label,
    )

    print("\n===== 测试集结果 (MSRA, 实体级) =====")
    print(f"Loss={metrics['loss']:.4f}")

    macro = metrics["macro"]
    micro = metrics["micro"]

    # 打印宏观 / 微观 F1（实体级）
    print("\n[Macro]")
    print(f"P={macro['precision']:.4%}  R={macro['recall']:.4%}  F1={macro['f1']:.4%}")
    print("[Micro]")
    print(f"P={micro['precision']:.4%}  R={micro['recall']:.4%}  F1={micro['f1']:.4%}")

    # 各实体类型指标：使用真实标签名（例如 MSRA: NR/NS/NT）
    print("\n===== 各实体类型指标 (MSRA) =====")
    # 统一列宽：Type 6 列，P/R/F1 9 列（含4位小数），计数列 8 列
    print(f"{'Type':<6} {'P':>9} {'R':>9} {'F1':>9} {'Gold':>8} {'Pred':>8} {'TP':>8}")

    scores = metrics["scores"]

    # 若想按固定顺序显示 NR/NS/NT，可以在这里排序
    order = ["NR", "NS", "NT"]
    types_in_scores = list(scores.keys())
    ordered_types = [t for t in order if t in types_in_scores] + [
        t for t in sorted(types_in_scores) if t not in order
    ]

    for t in ordered_types:
        s = scores[t]
        print(
            f"{t:<6} "
            f"{s['precision']:>9.4f} {s['recall']:>9.4f} {s['f1']:>9.4f} "
            f"{s['n_gold']:>8d} {s['n_pred']:>8d} {s['n_true_positive']:>8d}"
        )


if __name__ == "__main__":
    main()