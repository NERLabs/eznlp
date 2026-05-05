#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【实验】完整 FLAT 模型训练脚本 - RedJujube 数据集

使用自己构建的 FLAT 模型组件进行 NER 训练

使用方式:
    python train_flat_complete.py --data_dir _2DATA/RedJujube --word_file assets/vectors/ctb.50d.vec
"""
import argparse
import os
import sys
import datetime
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# 添加项目路径（与 text2text/train_redjujube_ner 保持一致）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "_8TOOL"))

# 复用 eznlp 的训练工具和评估指标（不改内部结构）
from utils import add_base_arguments, parse_to_args
from eznlp.training import LRLambda, collect_params, check_param_groups
from eznlp.metrics import precision_recall_f1_report

# 导入 FLAT 组件
from _4MODELS.models.flat_data_processor import FLATDataProcessor, FLATDataset, load_word_list
from _4MODELS.models.flat_extractor import FLATModel, FLATModelWithBERT, FLATWithInterAttention
from eznlp.model.bert_like import BertLikeConfig
import transformers


def parse_args():
    """解析命令行参数（统一使用 eznlp 的基础训练超参）"""
    parser = argparse.ArgumentParser(
        description='完整 FLAT 模型训练',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars='@',
    )
    # 训练超参：seed / num_epochs / batch_size / lr / optimizer / scheduler / num_grad_acc_steps / grad_clip / use_amp 等
    parser = add_base_arguments(parser)

    # 数据参数
    group_data = parser.add_argument_group("data")
    group_data.add_argument(
        '--data_dir',
        type=str,
        default='_2DATA/RedJujube',
        help='数据目录（包含 *.bmes）'
    )
    group_data.add_argument(
        '--word_file',
        type=str,
        default='assets/vectors/ctb.50d.vec',
        help='词表文件路径（用于构建 Trie）'
    )
    group_data.add_argument(
        '--save_dir',
        type=str,
        default='cache/flat_complete',
        help='模型保存目录'
    )

    # 模型参数（FLAT 专用）
    group_model = parser.add_argument_group("FLAT model")
    group_model.add_argument('--hidden_size', type=int, default=256, help='隐藏层维度')
    group_model.add_argument('--embed_size', type=int, default=50, help='嵌入维度')
    group_model.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    group_model.add_argument('--ff_size', type=int, default=-1, help='前馈网络维度（-1 表示 hidden_size * 4）')
    group_model.add_argument('--max_seq_len', type=int, default=256, help='最大序列长度')
    group_model.add_argument('--dropout', type=float, default=0.15, help='Dropout 率')
    group_model.add_argument(
        '--four_pos_fusion',
        type=str,
        default='ff',
        choices=['ff', 'attn', 'gate', 'ff_linear', 'ff_two'],
        help='四位置融合方式'
    )
    group_model.add_argument('--use_bert', action='store_true', help='是否使用 BERT 嵌入')
    group_model.add_argument(
        '--bert_model',
        type=str,
        default='bert-base-chinese',
        help='BERT 模型名称或本地路径'
    )
    group_model.add_argument(
        '--model_type',
        type=str,
        default='flat',
        choices=['flat', 'flat_inter'],
        help='模型类型: flat=原始FLAT, flat_inter=Inter-Attention改进版'
    )

    # 设备与训练配置
    group_train = parser.add_argument_group("training config")

    # 其他参数
    group_misc = parser.add_argument_group("misc")
    group_misc.add_argument('--device', type=str, default='cuda', help='设备')
    group_misc.add_argument('--eval_every', type=int, default=200, help='每 N 步评估一次')
    group_misc.add_argument('--weight_decay', type=float, default=0.0, help='权重衰减')

    args = parse_to_args(parser)
    return args


def load_bmes_data(file_path):
    """加载 BMES 格式的 NER 数据"""
    data = []
    chars = []
    labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if chars:
                    data.append({'chars': chars, 'labels': labels})
                    chars = []
                    labels = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    chars.append(parts[0])
                    labels.append(parts[1])
    
    if chars:
        data.append({'chars': chars, 'labels': labels})
    
    return data


def evaluate(model, dataloader, device, bert_model=None, bert_tokenizer=None, use_bert=False, idx2label=None, model_type='flat'):
    """评估模型（使用与 eznlp 相同的实体级 F1 计算）"""
    model.eval()
    if bert_model is not None:
        bert_model.eval()
    
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            lattice = batch['lattice'].to(device)
            pos_s = batch['pos_s'].to(device)
            pos_e = batch['pos_e'].to(device)
            seq_len = batch['seq_len'].to(device)
            lex_num = batch['lex_num'].to(device)
            target = batch['target'].to(device)
            
            if model_type == 'flat_inter':
                # BERT 内嵌模式：直接传 chars
                # 计算 loss（临时切换到训练模式以获取 loss）
                was_training = model.training
                model.train()
                output_loss = model(lattice, seq_len, lex_num, pos_s, pos_e, target, chars=batch['chars'])
                total_loss += output_loss['loss'].item()
                num_batches += 1
                
                # 推理
                model.eval()
                output_pred = model(lattice, seq_len, lex_num, pos_s, pos_e, chars=batch['chars'])
                preds = output_pred['pred']
                
                # 恢复之前状态
                if was_training:
                    model.train()
            else:
                # 原始模式：外部处理 BERT
                bert_embed = None
                if use_bert and bert_model is not None:
                    chars_list = batch['chars']
                    batch_size = len(chars_list)
                    max_char_len = max(len(chars) for chars in chars_list)
                    bert_embeds = []
                    
                    for chars in chars_list:
                        text = ''.join(chars)
                        inputs = bert_tokenizer(
                            text,
                            return_tensors='pt',
                            add_special_tokens=False,
                            truncation=True,
                            max_length=len(chars)
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
                        
                        bert_embeds.append(char_embeds[:len(chars)])
                    
                    bert_embed = torch.zeros(batch_size, max_char_len, 768, device=device)
                    for i, emb in enumerate(bert_embeds):
                        bert_embed[i, :len(emb)] = emb
                
                # 计算 loss（临时切换到训练模式以获取 loss）
                was_training = model.training
                model.train()
                output_loss = model(lattice, seq_len, lex_num, pos_s, pos_e, target, bert_embed=bert_embed)
                total_loss += output_loss['loss'].item()
                num_batches += 1
                
                # 推理
                model.eval()
                output_pred = model(lattice, seq_len, lex_num, pos_s, pos_e, bert_embed=bert_embed)
                preds = output_pred['pred']
                
                # 恢复之前状态
                if was_training:
                    model.train()
            
            # 收集预测和目标（按字符位置）
            for i, (pred, length) in enumerate(zip(preds, seq_len)):
                all_preds.append(pred[:length.item()])
                all_targets.append(target[i, :length.item()].cpu().tolist())
    
    # 将 BMES 标签转换为实体 span，与 eznlp 的 ER 评估保持一致
    set_y_pred = []
    set_y_gold = []
    for pred, gold in zip(all_preds, all_targets):
        pred_entities = list(extract_entities(pred, idx2label))
        gold_entities = list(extract_entities(gold, idx2label))
        set_y_pred.append(pred_entities)
        set_y_gold.append(gold_entities)
    
    # 使用 eznlp 的 precision_recall_f1_report 计算 Micro P/R/F1
    scores, ave_scores = precision_recall_f1_report(set_y_gold, set_y_pred)
    precision = ave_scores['micro']['precision']
    recall = ave_scores['micro']['recall']
    f1 = ave_scores['micro']['f1']
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    return {
        'loss': avg_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def extract_entities(labels, idx2label=None):
    """从 BMES 标签中提取实体（实体级别评估）
    
    Args:
        labels: 标签 ID 列表
        idx2label: ID 到标签字符串的映射
        
    Returns:
        set of (start, end, entity_type) tuples
    """
    entities = []
    current_entity = None
    current_start = None
    
    for i, label_id in enumerate(labels):
        # 转换为标签字符串
        if idx2label is not None:
            label = idx2label.get(label_id, 'O')
        else:
            label = str(label_id)
        
        if label == 'O' or label == '<PAD>' or label_id == 0:
            if current_entity:
                entities.append((current_start, i, current_entity))
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entities.append((current_start, i, current_entity))
            current_entity = label[2:]
            current_start = i
        elif label.startswith('M-'):
            if current_entity != label[2:]:
                if current_entity:
                    entities.append((current_start, i, current_entity))
                current_entity = None
        elif label.startswith('E-'):
            if current_entity == label[2:]:
                entities.append((current_start, i + 1, current_entity))
            current_entity = None
        elif label.startswith('S-'):
            if current_entity:
                entities.append((current_start, i, current_entity))
            entities.append((i, i + 1, label[2:]))
            current_entity = None
    
    if current_entity:
        entities.append((current_start, len(labels), current_entity))
    
    return set(entities)


def build_optimizer_and_scheduler(model, num_train_batches: int, args, bert_model=None, model_type='flat'):
    """构建优化器和调度器（对齐 _8TOOL/utils.build_trainer 的策略）"""
    # 构建参数组：区分预训练参数和非预训练参数
    param_groups = []
    
    # 先添加 BERT 参数（使用低学习率）
    if model_type == 'flat_inter' and args.use_bert:
        # BERT 内嵌模式：通过 model.pretrained_parameters() 获取 BERT 参数
        bert_params = model.pretrained_parameters()
        if bert_params:
            bert_lr = getattr(args, 'finetune_lr', 2e-5)
            param_groups.append({"params": bert_params, "lr": bert_lr})
            print(f"  BERT fine-tuning 学习率: {bert_lr}")
    
    # 添加模型其他参数（使用正常学习率）
    param_groups.append({"params": collect_params(model, param_groups), "lr": args.lr})
    print(f"  模型学习率: {args.lr}")
    
    # 检查参数组
    assert check_param_groups(model, param_groups)
    
    # 如果使用外部 BERT（原始模式），把 BERT 参数加入优化器
    if args.use_bert and bert_model is not None:
        bert_lr = getattr(args, 'finetune_lr', args.lr * 0.1)
        bert_params = list(bert_model.parameters())
        param_groups.append({"params": bert_params, "lr": bert_lr})

    optimizer = getattr(torch.optim, args.optimizer)(param_groups)

    schedule_by_step = "warmup" in args.scheduler.lower()
    if args.scheduler == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )
    elif args.scheduler == "LinearDecayWithWarmup":
        # 计算总步数与 warmup 步数
        num_total_steps = num_train_batches * args.num_epochs
        # 修改：warmup 只需要 1 个 epoch 或 1000 步
        num_warmup_steps = min(num_train_batches, 1000)
        lr_lambda = LRLambda.linear_decay_lr_with_warmup(
            num_warmup_steps=num_warmup_steps,
            num_total_steps=num_total_steps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == "PowerDecayWithWarmup":
        num_total_steps = num_train_batches * args.num_epochs
        num_warmup_epochs = max(2, args.num_epochs // 5)
        num_warmup_steps = num_train_batches * num_warmup_epochs
        if num_total_steps < num_warmup_steps:
            num_warmup_steps = num_total_steps
        lr_lambda = LRLambda.power_decay_lr_with_warmup(
            num_warmup_steps=num_warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler, schedule_by_step


def train(args):
    """训练主函数"""
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 创建保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"{args.save_dir}/flat_{timestamp}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 日志文件路径（JSON Lines，一行一条记录）
    log_file = os.path.join(save_dir, 'training_log.jsonl')
    
    # TensorBoard 日志目录与 writer
    tb_log_dir = os.path.join(save_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    # 检测数据集类型
    if os.path.exists(os.path.join(args.data_dir, 'redjujube_train.bmes')):
        dataset_name = 'RedJujube'
    elif os.path.exists(os.path.join(args.data_dir, 'train.char.bmes')):
        dataset_name = 'MSRA'
    else:
        dataset_name = 'Unknown'
    
    print("="*70)
    print(f"【实验】完整 FLAT 模型训练 - {dataset_name} 数据集")
    print("="*70)
    
    # 加载词表
    print("\n[1/6] 加载词表...")
    if os.path.exists(args.word_file):
        word_list = load_word_list(args.word_file)
    else:
        # 使用默认词表（从 softlexicon 文件加载）
        softlex_file = os.path.join(args.data_dir, 'softlexicon_train.txt')
        if os.path.exists(softlex_file):
            word_list = load_word_list(softlex_file)
        else:
            print("⚠️  未找到词表文件，使用空词表")
            word_list = []
    
    # 初始化 BERT （如果需要）
    bert_model = None
    bert_tokenizer = None
    if args.use_bert:
        print("\n[2/6] 初始化 BERT...")
        from transformers import BertModel, BertTokenizer
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        bert_model = BertModel.from_pretrained(args.bert_model)
        bert_model.eval()
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        bert_model.to(device)
        print(f"  BERT 模型: {args.bert_model}")
        print(f"  设备: {device}")
    
    # 创建数据处理器
    processor = FLATDataProcessor(word_list, max_seq_len=args.max_seq_len)
    
    # 加载数据
    print(f"\n[{3 if args.use_bert else 2}/6] 加载数据...")
    # 数据文件名（根据数据集自动检测）
    if os.path.exists(os.path.join(args.data_dir, 'redjujube_train.bmes')):
        # RedJujube 数据集
        train_file = os.path.join(args.data_dir, 'redjujube_train.bmes')
        dev_file = os.path.join(args.data_dir, 'redjujube_dev.bmes')
        test_file = os.path.join(args.data_dir, 'redjujube_test.bmes')
        dataset_name = 'RedJujube'
    elif os.path.exists(os.path.join(args.data_dir, 'train.char.bmes')):
        # MSRA 数据集
        train_file = os.path.join(args.data_dir, 'train.char.bmes')
        dev_file = os.path.join(args.data_dir, 'dev.char.bmes')
        test_file = os.path.join(args.data_dir, 'test.char.bmes')
        dataset_name = 'MSRA'
    else:
        raise ValueError(f"未找到训练数据文件，请检查 {args.data_dir} 目录")
    
    train_data = load_bmes_data(train_file)
    dev_data = load_bmes_data(dev_file)
    test_data = load_bmes_data(test_file)
    
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(dev_data)} 条")
    print(f"  测试集: {len(test_data)} 条")
    
    # 构建词汇表
    print(f"\n[{4 if args.use_bert else 3}/6] 构建词汇表...")
    all_chars = set()
    all_labels = set(['O'])
    
    for data in train_data + dev_data + test_data:
        all_chars.update(data['chars'])
        all_labels.update(data['labels'])
    
    # 字符词汇表
    processor.char_vocab = {'<PAD>': 0, '<UNK>': 1}
    processor.idx2char = {0: '<PAD>', 1: '<UNK>'}
    for i, char in enumerate(sorted(all_chars)):
        idx = len(processor.char_vocab)
        processor.char_vocab[char] = idx
        processor.idx2char[idx] = char
    
    # 标签词汇表
    processor.label_vocab = {'<PAD>': 0}
    processor.idx2label = {0: '<PAD>'}
    for label in sorted(all_labels):
        if label not in processor.label_vocab:
            idx = len(processor.label_vocab)
            processor.label_vocab[label] = idx
            processor.idx2label[idx] = label
    
    print(f"  字符数量: {len(processor.char_vocab)}")
    print(f"  标签数量: {len(processor.label_vocab)}")
    
    # 处理数据
    def process_data(data_list):
        processed = []
        for data in data_list:
            result = processor.process_sentence(data['chars'], data['labels'])
            # 转换标签为索引
            result['target'] = [processor.label_vocab.get(l, 0) for l in data['labels'][:result['seq_len']]]
            # 转换字符为索引
            result['lattice_ids'] = []
            for item in result['lattice']:
                if len(item) == 1:
                    result['lattice_ids'].append(processor.char_vocab.get(item, 1))
                else:
                    # 对于词汇，使用第一个字符的索引（简化处理）
                    result['lattice_ids'].append(processor.char_vocab.get(item[0], 1))
            processed.append(result)
        return processed
    
    train_processed = process_data(train_data)
    dev_processed = process_data(dev_data)
    test_processed = process_data(test_data)
    
    # 创建数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    def collate_fn(batch):
        batch_size = len(batch)
        max_seq_len = max(d['seq_len'] for d in batch)
        max_lattice_len = max(d['seq_len'] + d['lex_num'] for d in batch)
        
        lattice_ids = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_s = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        pos_e = torch.zeros(batch_size, max_lattice_len, dtype=torch.long)
        seq_len = torch.zeros(batch_size, dtype=torch.long)
        lex_num = torch.zeros(batch_size, dtype=torch.long)
        target = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        
        for i, d in enumerate(batch):
            cur_len = d['seq_len'] + d['lex_num']
            lattice_ids[i, :cur_len] = torch.tensor(d['lattice_ids'][:cur_len])
            pos_s[i, :cur_len] = torch.tensor(d['pos_s'][:cur_len])
            pos_e[i, :cur_len] = torch.tensor(d['pos_e'][:cur_len])
            seq_len[i] = d['seq_len']
            lex_num[i] = d['lex_num']
            target[i, :d['seq_len']] = torch.tensor(d['target'])
        
        return {
            'lattice': lattice_ids,
            'pos_s': pos_s,
            'pos_e': pos_e,
            'seq_len': seq_len,
            'lex_num': lex_num,
            'target': target,
            'chars': [d['chars'] for d in batch],  # 用于 BERT
        }
    
    train_dataset = SimpleDataset(train_processed)
    dev_dataset = SimpleDataset(dev_processed)
    test_dataset = SimpleDataset(test_processed)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # 创建模型
    print(f"\n[{5 if args.use_bert else 4}/6] 创建模型...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    model_type = getattr(args, 'model_type', 'flat')
    
    if model_type == 'flat_inter':
        print("  模型类型: FLAT + Inter-Attention (NFLAT 风格)")
        
        # 使用框架的 BertLikeConfig 将 BERT 集成到模型内部
        bert_config = None
        if args.use_bert:
            import transformers
            from eznlp.model.bert_like import BertLikeConfig
            
            bert_like = transformers.AutoModel.from_pretrained(args.bert_model)
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model)
            
            bert_config = BertLikeConfig(
                tokenizer=tokenizer,
                bert_like=bert_like,
                freeze=False,  # fine-tune
                mix_layers="top",
            )
            print(f"  BERT 模型: {args.bert_model} (内嵌)")
        
        model = FLATWithInterAttention(
            vocab_size=len(processor.char_vocab),
            label_size=len(processor.label_vocab),
            hidden_size=768 if args.use_bert else args.hidden_size,
            embed_size=args.embed_size,
            num_heads=8 if args.use_bert else args.num_heads,
            num_inter_layers=args.num_layers,
            ff_size=args.ff_size if args.ff_size > 0 else 768 * 4,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_bert=args.use_bert,
            bert_config=bert_config,  # 传入 BertLikeConfig
            use_ffn=True,
        ).to(device)
        
        # BERT 已集成在模型内部，不需要外部 bert_model
        bert_model = None
        bert_tokenizer = None
    else:
        print("  模型类型: 原始 FLAT")
        model = FLATModel(
            vocab_size=len(processor.char_vocab),
            label_size=len(processor.label_vocab),
            hidden_size=args.hidden_size,
            embed_size=args.embed_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_size=args.ff_size,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_bigram=False,
            use_bert=args.use_bert,
            bert_hidden_size=768 if args.use_bert else 0,
            four_pos_fusion=args.four_pos_fusion,
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  设备: {device}")
    
    # 构建优化器与调度器（对齐 eznlp 的策略）
    optimizer, scheduler, schedule_by_step = build_optimizer_and_scheduler(
        model, len(train_loader), args, bert_model=bert_model, model_type=model_type
    )

    # AMP 梯度缩放器
    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    
    # 训练
    print(f"\n[6/6] 开始训练...")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print()
    
    best_f1 = 0
    global_step = 0
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            lattice = batch['lattice'].to(device)
            pos_s = batch['pos_s'].to(device)
            pos_e = batch['pos_e'].to(device)
            seq_len = batch['seq_len'].to(device)
            lex_num = batch['lex_num'].to(device)
            target = batch['target'].to(device)
            
            # 提取 BERT embeddings（Fine-tuning 模式，保留梯度）
            bert_embed = None
            if args.use_bert and bert_model is not None:
                bert_model.train()  # 确保 BERT 在训练模式
                chars_list = batch['chars']  # List[List[str]]
                batch_size = len(chars_list)
                max_char_len = max(len(chars) for chars in chars_list)
                
                bert_embeds = []
                for chars in chars_list:
                    text = ''.join(chars)
                    inputs = bert_tokenizer(
                        text,
                        return_tensors='pt',
                        add_special_tokens=False,
                        truncation=True,
                        max_length=len(chars)
                    ).to(device)
                    outputs = bert_model(**inputs)
                    char_embeds = outputs.last_hidden_state[0]  # [seq_len, 768]
                    
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
                    
                    bert_embeds.append(char_embeds[:len(chars)])
                
                # 使用 stack 保持梯度连接
                padded_embeds = []
                for emb in bert_embeds:
                    if len(emb) < max_char_len:
                        pad = torch.zeros(max_char_len - len(emb), 768, device=device)
                        padded_emb = torch.cat([emb, pad], dim=0)
                    else:
                        padded_emb = emb[:max_char_len]
                    padded_embeds.append(padded_emb)
                bert_embed = torch.stack(padded_embeds, dim=0)
            
            # 前向传播
            with torch.amp.autocast(device_type=device.type if device.type != "cpu" else "cpu", enabled=args.use_amp):
                if model_type == 'flat_inter':
                    # BERT 内嵌模式：传递 chars
                    output = model(lattice, seq_len, lex_num, pos_s, pos_e, target, chars=batch['chars'])
                else:
                    # 外部 BERT 模式
                    output = model(lattice, seq_len, lex_num, pos_s, pos_e, target, bert_embed=bert_embed)
                raw_loss = output['loss']
            
            # 累积显示用
            epoch_loss += raw_loss.item()
            num_batches += 1
            
            # 梯度累积：loss 平均到 num_grad_acc_steps
            loss = raw_loss / args.num_grad_acc_steps
            
            scaler.scale(loss).backward()
            global_step += 1
            
            # 到达一个“真实 batch”再 step
            if global_step % args.num_grad_acc_steps == 0:
                if args.grad_clip is not None and args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None and schedule_by_step:
                    scheduler.step()
            
            # 当前学习率
            if scheduler is not None:
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]['lr']
            
            # 记录训练 loss 和学习率到 TensorBoard
            writer.add_scalar("train/loss", raw_loss.item(), global_step)
            writer.add_scalar("train/lr", current_lr, global_step)
            
            pbar.set_postfix({'loss': f"{raw_loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            
            # 定期评估（保持原逻辑）
            if global_step % args.eval_every == 0:
                dev_metrics = evaluate(model, dev_loader, device, bert_model, bert_tokenizer, args.use_bert, processor.idx2label, model_type)
                print(f"\n  Step {global_step}: Dev Loss={dev_metrics['loss']:.4f}, "
                      f"P={dev_metrics['precision']:.2%}, R={dev_metrics['recall']:.2%}, "
                      f"F1={dev_metrics['f1']:.2%}")
                
                # 将本次 dev 评估结果写入日志文件
                dev_log_entry = {
                    "type": "dev_eval",
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "loss": dev_metrics["loss"],
                    "precision": dev_metrics["precision"],
                    "recall": dev_metrics["recall"],
                    "f1": dev_metrics["f1"],
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dev_log_entry, ensure_ascii=False) + "\n")
                
                # 同步写入 TensorBoard
                writer.add_scalar("dev/loss", dev_metrics["loss"], global_step)
                writer.add_scalar("dev/precision", dev_metrics["precision"], global_step)
                writer.add_scalar("dev/recall", dev_metrics["recall"], global_step)
                writer.add_scalar("dev/f1", dev_metrics["f1"], global_step)
                
                if dev_metrics['f1'] > best_f1:
                    best_f1 = dev_metrics['f1']
                    # 保存模型 state_dict 和配置
                    best_model_path = os.path.join(save_dir, 'best_model.pth')
                    config_path = os.path.join(save_dir, 'config.pth')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'model_type': model_type,
                        },
                        best_model_path,
                    )
                    torch.save(
                        {
                            'args': vars(args),
                            'char_vocab': processor.char_vocab,
                            'label_vocab': processor.label_vocab,
                            'idx2label': processor.idx2label,
                        },
                        config_path,
                    )
                    print(f"  ✅ 保存最佳模型 (F1={best_f1:.2%}) 到 {best_model_path}")
                    print(f"  ✅ 保存配置到 {config_path}")
                    
                    # 同时保存 BERT 权重（如果使用）
                    if args.use_bert and bert_model is not None:
                        bert_save_path = os.path.join(save_dir, 'best_bert.pth')
                        torch.save(bert_model.state_dict(), bert_save_path)
                        print(f"  ✅ 保存 BERT 权重到 {bert_save_path}")
                
                model.train()
        
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1} 完成: 平均损失={avg_loss:.4f}")
        
        # 记录 epoch 级别训练损失
        epoch_log_entry = {
            "type": "epoch_end",
            "epoch": epoch + 1,
            "avg_train_loss": avg_loss,
            "global_step": global_step,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_log_entry, ensure_ascii=False) + "\n")
        
        writer.add_scalar("epoch/avg_train_loss", avg_loss, epoch + 1)
    
    # 最终测试
    print("\n" + "="*70)
    print("最终测试")
    print("="*70)
    
    # 加载配置
    config_path = os.path.join(save_dir, 'config.pth')
    config = torch.load(config_path, map_location=device, weights_only=False)
    
    # 重新初始化 BERT（如果使用）
    test_bert_model = None
    test_bert_tokenizer = None
    if args.use_bert:
        from transformers import BertModel, BertTokenizer
        test_bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model)
        test_bert_model = BertModel.from_pretrained(args.bert_model)
        test_bert_model.eval()
        test_bert_model.to(device)
        
        # 加载最佳 BERT 权重
        bert_save_path = os.path.join(save_dir, 'best_bert.pth')
        if os.path.exists(bert_save_path):
            test_bert_model.load_state_dict(torch.load(bert_save_path, map_location=device))
            print(f"  ✅ 已加载 BERT 最佳权重")
    
    # 重新创建模型并加载权重
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    # 根据模型类型重建
    if model_type == 'flat_inter':
        # BERT 内嵌模式需要从 config 获取 bert_config
        bert_config = None
        if args.use_bert:
            import transformers
            from eznlp.model.bert_like import BertLikeConfig
            bert_like = transformers.AutoModel.from_pretrained(args.bert_model)
            tokenizer = transformers.AutoTokenizer.from_pretrained(args.bert_model)
            bert_config = BertLikeConfig(
                tokenizer=tokenizer,
                bert_like=bert_like,
                freeze=False,
                mix_layers="top",
            )
        
        model = FLATWithInterAttention(
            vocab_size=len(config['char_vocab']),
            label_size=len(config['label_vocab']),
            hidden_size=768 if args.use_bert else args.hidden_size,
            embed_size=args.embed_size,
            num_heads=8 if args.use_bert else args.num_heads,
            num_inter_layers=args.num_layers,
            ff_size=args.ff_size if args.ff_size > 0 else 768 * 4,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_bert=args.use_bert,
            bert_config=bert_config,
            use_ffn=True,
        ).to(device)
    else:
        model = FLATModel(
            vocab_size=len(config['char_vocab']),
            label_size=len(config['label_vocab']),
            hidden_size=args.hidden_size,
            embed_size=args.embed_size,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            ff_size=args.ff_size,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            use_bigram=False,
            use_bert=args.use_bert,
            bert_hidden_size=768 if args.use_bert else 0,
            four_pos_fusion=args.four_pos_fusion,
        ).to(device)
    
    # 加载最佳权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"  ✅ 已加载最佳模型权重")
    
    # 测试集评估
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        bert_model=test_bert_model,
        bert_tokenizer=test_bert_tokenizer,
        use_bert=args.use_bert,
        idx2label=config['idx2label'],
        model_type=model_type,
    )
    
    # 将最终测试结果写入日志
    test_log_entry = {
        "type": "test",
        "loss": test_metrics["loss"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(test_log_entry, ensure_ascii=False) + "\n")
    
    # 同步到 TensorBoard（用最后的 global_step 作为 step）
    writer.add_scalar("test/loss", test_metrics["loss"], global_step)
    writer.add_scalar("test/precision", test_metrics["precision"], global_step)
    writer.add_scalar("test/recall", test_metrics["recall"], global_step)
    writer.add_scalar("test/f1", test_metrics["f1"], global_step)
    
    print(f"\n📊 测试结果:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.2%}")
    print(f"  Recall: {test_metrics['recall']:.2%}")
    print(f"  F1: {test_metrics['f1']:.2%}")
    
    # 保存结果
    results = {
        'model_config': {
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'num_heads': args.num_heads,
            'dropout': args.dropout,
        },
        'test_metrics': test_metrics,
        'best_dev_f1': best_f1,
    }
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {save_dir}")
    
    # 关闭 TensorBoard writer
    writer.close()
    
    return test_metrics


if __name__ == '__main__':
    args = parse_args()
    train(args)