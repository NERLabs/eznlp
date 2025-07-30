# 简易自然语言处理工具包

过参数化神经网络常被描述为"惰性"（Chizat等人，2019），这一特性启发我们设计更易优化的架构与目标函数。

`eznlp`是基于`PyTorch`的神经网络自然语言处理工具包，当前支持以下任务：

* 文本分类（[实验结果](docs/text-classification.md)）
* 命名实体识别（[实验结果](docs/entity-recognition.md)）
* 关系抽取（[实验结果](docs/relation-extraction.md)）
* 属性抽取
* 机器翻译
* 图像描述生成

本代码库还包含已发表论文的配套代码：
* [深度跨度表示在命名实体识别中的应用](docs/deep-span.md)（ACL 2023会议）
* [边界平滑技术在命名实体识别中的应用](docs/boundary-smoothing.md)（ACL 2022会议）
* 《人工智能在医学》期刊论文《中文临床文本医学信息标注与抽取统一框架》中描述的[标注规范](publications/framework/scheme.pdf)与[华美-500](publications/framework/HwaMei-500.md)数据集

## 安装指南

### 环境配置

推荐使用Docker环境。最新验证镜像为`pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`：
```bash
$ docker run --rm -it --gpus=all --mount type=bind,source=${PWD},target=/workspace/eznlp --workdir /workspace/eznlp pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
```

也可创建虚拟环境：
```bash
$ conda create --name eznlp python=3.11
$ conda activate eznlp
```

### 安装工具包

常规安装：
```bash
$ pip install eznlp
```

开发者模式安装：
```bash
$ pip install -e .
```

## 执行代码

### 文本分类
```bash
$ python scripts/text_classification.py --dataset <数据集> [选项]
```

### 实体识别
```bash
$ python scripts/entity_recognition.py --dataset <数据集> [选项]
```

### 关系抽取
```bash
$ python scripts/relation_extraction.py --dataset <数据集> [选项]
```

### 属性抽取
```bash
$ python scripts/attribute_extraction.py --dataset <数据集> [选项]
```

## 引用说明

若使用本代码，请引用以下文献：

（引用格式保持原文不变）

## 参考文献
* Chizat, L., Oyallon, E., and Bach, F. 可微分编程中的惰性训练研究. 发表于*NeurIPS 2019*会议.