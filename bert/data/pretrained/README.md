# 预训练模型存储目录

此目录用于存储各模型预训练文件



## 预训练模型下载

- BERT 中文预训练模型: `bert-base-chinese`  [下载地址](https://github.com/huggingface/transformers)

- ROBERTA 中文预训练模型: `RoBERTa-wwm-ext, Chinese` [下载地址](https://awesomeopensource.com/project/ymcui/Chinese-BERT-wwm)

- ERNIE1.0 中文预训练模型: `ERNIE 1.0 Base for Chinese` [下载地址](https://github.com/nghuyong/ERNIE-Pytorch)



## 目录格式

下载解压后的**预训练文件目录重命名为模型名，内部共包含3个文件，文件名不同请重命名为以下文件名**

- `config.json` 超参配置
- `pytorch_model.bin` 模型参数权重
- `vocab.txt` 词表

例如BERT预训练模型应有以下文件树

```
├── bert
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
```

