### 边界平滑用于命名实体识别  
本文档说明《边界平滑用于命名实体识别》项目所需材料与代码。

---

## 环境配置
### 安装依赖
根据 [README](../README.md) 创建环境并安装依赖项及 `eznlp` 库。

### 下载与处理数据集
* **英文数据集**
    * CoNLL 2003
    * OntoNotes 5：从 https://catalog.ldc.upenn.edu/LDC2013T19 下载，按 Pradhan 等人 (2013) 方法处理
    * ACE 2004：从 https://catalog.ldc.upenn.edu/LDC2005T09 下载，按 Lu 和 Roth (2015) 方法处理
    * ACE 2005：从 https://catalog.ldc.upenn.edu/LDC2006T06 下载，按 Lu 和 Roth (2015) 方法处理

* **中文数据集**
    * OntoNotes 4：从 https://catalog.ldc.upenn.edu/LDC2011T03 下载，按 Che 等人 (2013) 方法处理
    * MSRA：从 https://github.com/v-mipeng/LexiconAugmentedNER 下载
    * 微博命名实体识别：从 https://github.com/hltcoe/golden-horse 下载
    * 简历命名实体识别：从 https://github.com/jiesutd/LatticeLSTM 下载

### 下载预训练语言模型
通过 `transformers` 下载预训练模型并保存至 `assets/transformers`：
```bash
git clone https://huggingface.co/google-bert/bert-base-uncased  assets/transformers/bert-base-uncased
git clone https://huggingface.co/google-bert/bert-base-cased    assets/transformers/bert-base-cased
git clone https://huggingface.co/google-bert/bert-large-uncased assets/transformers/bert-large-uncased
git clone https://huggingface.co/google-bert/bert-large-cased   assets/transformers/bert-large-cased
git clone https://huggingface.co/FacebookAI/roberta-base        assets/transformers/roberta-base
git clone https://huggingface.co/FacebookAI/roberta-large       assets/transformers/roberta-large
git clone https://huggingface.co/hfl/chinese-bert-wwm-ext       assets/transformers/hfl/chinese-bert-wwm-ext
git clone https://huggingface.co/hfl/chinese-macbert-base       assets/transformers/hfl/chinese-macbert-base
git clone https://huggingface.co/hfl/chinese-macbert-large      assets/transformers/hfl/chinese-macbert-large
```

---

## 运行代码
**英文数据集命令**：
```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --num_epochs 50 \
    --batch_size 48 \
    --num_grad_acc_steps 1 \
    --dataset {conll2003 | conll2012 | ace2004 | ace2005} \
    --ck_decoder boundary_selection \
    --sb_epsilon {0.0 | 0.1 | 0.2 | 0.3} \
    --sb_size {1 | 2} \
    --bert_arch {RoBERTa_base | RoBERTa_large | BERT_base | BERT_large} \
    --use_interm2 \
    [其他参数]
```

**中文数据集命令**：
```bash
$ python scripts/entity_recognition.py @scripts/options/with_bert.opt \
    --num_epochs 50 \
    --batch_size 48 \
    --num_grad_acc_steps 1 \
    --dataset {ontonotesv4_zh | SIGHAN2006 | WeiboNER | ResumeNER} \
    --ck_decoder boundary_selection \
    --sb_epsilon {0.0 | 0.1 | 0.2 | 0.3} \
    --sb_size {1 | 2} \
    --bert_arch {BERT_base_wwm | MacBERT_base | MacBERT_large} \
    --use_interm2 \
    [其他参数]
```

**查看完整参数说明**：
```bash
$ python scripts/entity_recognition.py --help
```
