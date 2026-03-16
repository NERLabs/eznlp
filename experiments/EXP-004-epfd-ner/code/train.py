# -*- coding: utf-8 -*-
"""
EXP-004-epfd-ner 训练脚本
基于 eznlp 框架
"""

import os
import sys
import yaml
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from eznlp.dataset import Dataset
from eznlp.metrics import precision_recall_f1_report
from eznlp.nn.modules.crf import CRF
from eznlp.training import Trainer
from transformers import AutoModel, AutoTokenizer

from model import EPFDNERConfig, EPFDNERModel


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(config: dict):
    """加载数据集"""
    from eznlp.io import ConllIO
    
    data_config = config['data']
    io = ConllIO(
        text_col_id=0,
        tag_col_id=1,
        scheme="BMES",
        encoding="utf-8",
        token_sep="",
        pad_token="<pad>",
    )
    
    train_data = io.read(os.path.join(data_config['data_dir'], data_config['train_file']))
    dev_data = io.read(os.path.join(data_config['data_dir'], data_config['dev_file']))
    test_data = io.read(os.path.join(data_config['data_dir'], data_config['test_file']))
    
    logger.info(f"训练集：{len(train_data)} 句")
    logger.info(f"验证集：{len(dev_data)} 句")
    logger.info(f"测试集：{len(test_data)} 句")
    
    return train_data, dev_data, test_data, io


def build_model(config: dict, num_labels: int, device: torch.device):
    """构建模型"""
    model_config = config['model']
    
    # 创建模型配置
    epfd_config = EPFDNERConfig(
        bert_pretrained=model_config['encoder']['pretrained'],
        bert_freeze=model_config['encoder']['freeze'],
        res_bilstm=model_config.get('res_bilstm', {}),
        self_gate=model_config.get('self_gate', {}),
        entity_feature=model_config.get('entity_feature', {}),
        num_labels=num_labels,
        ablation=config.get('ablation', {}).get('mode', 'full')
    )
    
    # 实例化模型
    model = epfd_config.instantiate()
    
    # 加载 BERT
    logger.info(f"加载预训练模型: {model_config['encoder']['pretrained']}")
    bert = AutoModel.from_pretrained(model_config['encoder']['pretrained'])
    model.set_bert(bert)
    
    # 使用 CRF 解码器
    # 设置 pad_idx=0，因为 'O' 标签的 ID 是 0
    crf = CRF(num_labels, batch_first=True)
    model.set_crf(crf)
    logger.info("使用 CRF 解码器")
    
    model = model.to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,} (可训练: {trainable_params:,})")
    
    return model


def build_optimizer(model, config: dict, num_training_steps: int):
    """构建优化器"""
    training_config = config['training']
    
    # 分层学习率
    bert_params = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters() if 'bert' not in n]
    
    optimizer = AdamW([
        {'params': bert_params, 'lr': training_config['learning_rate']['bert']},
        {'params': other_params, 'lr': training_config['learning_rate']['other']}
    ], weight_decay=training_config['weight_decay'])
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[training_config['learning_rate']['bert'], training_config['learning_rate']['other']],
        total_steps=num_training_steps,
        pct_start=training_config['warmup_ratio']
    )
    
    return optimizer, scheduler


def evaluate(model, dataloader, device, id2label, debug=False):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            predictions = outputs['predictions']
            
            # 将 ID 转换为标签
            id2label_list = list(id2label.values()) if isinstance(id2label, dict) else id2label
            
            for pred, label, mask in zip(predictions, labels, attention_mask):
                seq_len = mask.sum().item()
                # CRF.decode() 返回的是列表的列表，pred 已经是整数列表
                if isinstance(pred, list):
                    pred_tags = [id2label.get(p, 'O') if isinstance(id2label, dict) else id2label[p] 
                                for p in pred[:seq_len]]
                else:
                    # 如果是 tensor
                    pred_tags = [id2label.get(p.item(), 'O') if isinstance(id2label, dict) else id2label[p.item()] 
                                for p in pred[:seq_len]]
                label_tags = [id2label.get(l.item(), 'O') if isinstance(id2label, dict) else id2label[l.item()] 
                             for l in label[:seq_len] if l.item() != -100]
                
                # Debug: 打印前几个样本
                if debug and batch_idx == 0 and len(all_preds) < 2:
                    print(f"\nSample {len(all_preds)}:")
                    print(f"  Pred tags: {pred_tags[:20]}")
                    print(f"  Label tags: {label_tags[:20]}")
                
                # 转换为 BIO 格式用于评估
                all_preds.append(pred_tags)
                all_labels.append(label_tags)
    
    # 使用 precision_recall_f1_report 计算指标
    from eznlp.metrics import precision_recall_f1_report
    
    # 将标签序列转换为 chunks 格式用于评估
    def tags_to_chunks(tags):
        """将 BIO 标签转换为 (type, start, end) 元组列表"""
        chunks = []
        current_type = None
        start = None
        
        for i, tag in enumerate(tags):
            if tag.startswith('B-'):
                if current_type is not None:
                    chunks.append((current_type, start, i))
                current_type = tag[2:]
                start = i
            elif tag.startswith('I-'):
                if current_type is None or tag[2:] != current_type:
                    # I-标签不匹配，视为新实体开始
                    if current_type is not None:
                        chunks.append((current_type, start, i))
                    current_type = tag[2:]
                    start = i
            else:  # O
                if current_type is not None:
                    chunks.append((current_type, start, i))
                current_type = None
                start = None
        
        # 处理最后一个 chunk
        if current_type is not None:
            chunks.append((current_type, start, len(tags)))
        
        return chunks
    
    # 转换预测和标签为 chunks 格式
    pred_chunks = [tags_to_chunks(tags) for tags in all_preds]
    gold_chunks = [tags_to_chunks(tags) for tags in all_labels]
    
    # 计算指标
    _, ave_scores = precision_recall_f1_report(gold_chunks, pred_chunks)
    
    return {
        'precision': ave_scores['micro']['precision'],
        'recall': ave_scores['micro']['recall'],
        'f1': ave_scores['micro']['f1']
    }


def train(config_path: str, ablation: str = None):
    """训练主函数"""
    # 加载配置
    config = load_config(config_path)
    
    # 覆盖消融模式
    if ablation:
        config['ablation']['mode'] = ablation
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config['training']['seed'])
    
    # 加载数据
    train_data, dev_data, test_data, io = load_data(config)
    
    # 构建 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['encoder']['pretrained'])
    
    # 获取标签集合（从训练数据构建，确保所有数据使用相同的映射）
    all_chunk_types = set()
    for data in train_data:
        for chunk_type, start, end in data['chunks']:
            all_chunk_types.add(chunk_type)
    
    # 构建 BIO 标签映射（简化版，更容易训练）
    tags_list = ['O']
    for chunk_type in sorted(all_chunk_types):
        tags_list.extend([f'B-{chunk_type}', f'I-{chunk_type}'])
    label2id = {tag: i for i, tag in enumerate(tags_list)}
    id2label = {v: k for k, v in label2id.items()}
    num_labels = len(label2id)
    logger.info(f"标签数量：{num_labels} (实体类型：{len(all_chunk_types)}种，使用 BIO 格式)")
    
    # 构建模型
    model = build_model(config, num_labels, device)
    
    # 创建数据加载器
    from torch.utils.data import DataLoader
    from eznlp.dataset import Dataset as NERDataset
    
    # 将 chunks 标签转换为 BIO 标签序列
    def convert_chunks_to_tags(data_list, label2id):
        """将 chunks 标注转换为 BIO 标签序列"""
        for data in data_list:
            tags = ['O'] * len(data['tokens'])
            for chunk_type, start, end in data['chunks']:
                tags[start] = f'B-{chunk_type}'
                for i in range(start + 1, end):
                    tags[i] = f'I-{chunk_type}'
            data['tags'] = tags
        return data_list
    
    # 转换标签
    train_data = convert_chunks_to_tags(train_data, label2id)
    dev_data = convert_chunks_to_tags(dev_data, label2id)
    test_data = convert_chunks_to_tags(test_data, label2id)
    
    # 构建模型配置（用于 Dataset）
    from eznlp.config import Config
    from eznlp.model.decoder import SequenceTaggingDecoderConfig
    
    decoder_config = SequenceTaggingDecoderConfig(
        scheme="BIOES",
        use_crf=True,
        tag2id=label2id,
        id2tag=id2label,
    )
    
    class SimpleModelConfig(Config):
        def __init__(self, decoder_config):
            self.decoder = decoder_config
        
        def build_vocabs_and_dims(self, data, *others):
            pass
        
        def exemplify(self, entry, training=True):
            # Tokenize
            tokens = entry['tokens']
            # 确保 tokens 是字符串列表
            if hasattr(tokens, 'text'):
                tokens_text = tokens.text  # TokenSequence 对象
            else:
                tokens_text = [str(t) for t in tokens]
            
            encoding = tokenizer(
                tokens_text,
                is_split_into_words=True,
                padding=False,
                truncation=True,
                max_length=512
            )
            
            # 获取 'O' 标签的 ID（用于 [CLS] 和 [SEP]）
            o_tag_id = self.decoder.tag2id.get('O', 0)
            
            # 使用 word_ids() 对齐 subword 和原始 token 的标签
            word_ids = encoding.word_ids()
            tag_ids = []
            for word_id in word_ids:
                if word_id is None:
                    # [CLS] 或 [SEP] 或 padding
                    tag_ids.append(o_tag_id)
                else:
                    # 使用原始 token 的标签
                    tag_ids.append(self.decoder.tag2id.get(entry['tags'][word_id], o_tag_id))
            
            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': tag_ids
            }
        
        def collate(self, batch_examples):
            import torch
            from eznlp.nn.functional import seq_lens2mask
            
            input_ids = [ex['input_ids'] for ex in batch_examples]
            attention_mask = [ex['attention_mask'] for ex in batch_examples]
            labels = [ex['labels'] for ex in batch_examples]
            max_len = max(len(ids) for ids in input_ids)
            
            # Padding
            input_ids_padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
            attention_mask_padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
            labels_padded = torch.zeros(len(labels), max_len, dtype=torch.long)  # 使用 0 填充，而不是 -100
            
            for i, (ids, mask, lab) in enumerate(zip(input_ids, attention_mask, labels)):
                input_ids_padded[i, :len(ids)] = torch.tensor(ids)
                attention_mask_padded[i, :len(mask)] = torch.tensor(mask)
                labels_padded[i, :len(lab)] = torch.tensor(lab)
                # CRF 不支持 -100，需要用 mask 来忽略 padding 位置
            
            return {
                'input_ids': input_ids_padded,
                'attention_mask': attention_mask_padded,
                'labels': labels_padded
            }
    
    model_config = SimpleModelConfig(decoder_config)
    
    train_dataset = NERDataset(train_data, model_config, training=True)
    dev_dataset = NERDataset(dev_data, model_config, training=False)
    test_dataset = NERDataset(test_data, model_config, training=False)
    
    # 使用自定义 collate_fn
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=model_config.collate)
    dev_loader = DataLoader(dev_dataset, batch_size=config['training']['batch_size'], collate_fn=model_config.collate)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], collate_fn=model_config.collate)
    
    # 构建优化器
    num_training_steps = len(train_loader) * config['training']['epochs']
    optimizer, scheduler = build_optimizer(model, config, num_training_steps)
    
    # 训练循环
    best_f1 = 0
    patience = 0
    output_dir = Path(config['output']['save_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(config['training']['epochs']):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, labels=labels)
            loss = outputs['loss']
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        # 验证
        metrics = evaluate(model, dev_loader, device, id2label, debug=(epoch==1))
        logger.info(f"验证集 - P: {metrics['precision']:.2%}, R: {metrics['recall']:.2%}, F1: {metrics['f1']:.2%}")
            
        # 保存最佳模型
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            logger.info(f"保存最佳模型 (F1: {best_f1:.2%})")
        else:
            patience += 1
            if patience >= config['evaluation']['early_stop']:
                logger.info(f"早停：{patience} 轮无提升")
                break
        
    # 加载最佳模型并测试
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, device, id2label)
    logger.info(f"测试集 - P: {test_metrics['precision']:.2%}, R: {test_metrics['recall']:.2%}, F1: {test_metrics['f1']:.2%}")
    
    # 保存结果
    import json
    results = {
        'epoch': epoch + 1,
        'best_dev_f1': best_f1,
        'test_metrics': test_metrics,
        'config': config
    }
    with open(output_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return test_metrics


def main():
    parser = argparse.ArgumentParser(description='EXP-004-epfd-ner 训练脚本')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--ablation', type=str, default=None, 
                        choices=['full', 'no_res_bilstm', 'no_self_gate', 'no_entity_feature'],
                        help='消融实验模式')
    args = parser.parse_args()
    
    train(args.config, args.ablation)


if __name__ == '__main__':
    main()
