#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSRA FLAT 模型测试脚本（用于测试 cache/flat_msra_expert_bert/...）

- 加载训练时保存的 config.pth（包含 args / char_vocab / label_vocab）
- 加载 best_model.pth（完整模型对象）
- 按 BMES MSRA 格式读取 test.char.bmes
- 重建 FLAT 输入特征，调用 train_flat_complete.py 里的 evaluate 计算 P/R/F1
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
from _5TRAIN.train_flat_complete import evaluate  # 复用原来的评估逻辑


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

    # 6. 调用原始 evaluate 计算 P/R/F1
    print("\n===== 使用 FLAT 评估 MSRA 测试集 =====")
    metrics = evaluate(
        model,
        test_loader,
        device,
        bert_model=bert_model,
        bert_tokenizer=bert_tokenizer,
        use_bert=use_bert,
        idx2label=processor.idx2label,
    )

    print("\n===== 测试集结果 (MSRA, Micro-F1) =====")
    print(f"Loss={metrics['loss']:.4f}")
    print(f"P={metrics['precision']:.4%}")
    print(f"R={metrics['recall']:.4%}")
    print(f"F1={metrics['f1']:.4%}")


if __name__ == "__main__":
    main()