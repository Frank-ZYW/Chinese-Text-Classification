# BERT

包含BERT及其衍生的ROBERTA、ERNIE1.0等一些经典的深度网络模型



## 依赖

- python 3.7
- pytorch 1.5
- tqdm
- sklearn
- filelock
- pytorch-pretrained-bert
- tokenizers



## 目录结构

```
├── config.py
├── data
│   ├── dataset
│   │   ├── dev.json
│   │   ├── labels.json
│   │   ├── test.json
│   │   └── train.json
│   └── pretrained
├── dataloader.py
├── README.md
├── results
├── run.py
├── test.py
├── train.py
├── transformers
└── utils.py
```

#### 主要目录

- `data/dataset:` 存放原始数据集，原始数据集已随项目上传
- `data/pretrained:` 存放预训练词向量模型文件
- `results:` 模型训练结果存储
- `transformers:` 模型定义、配置及文本标记工具，代码实现参考项目[Transformers](https://github.com/huggingface/transformers)



## 预训练模型

- BERT 中文预训练模型: `bert-base-chinese`  [下载地址](https://github.com/huggingface/transformers)

- ROBERTA 中文预训练模型: `RoBERTa-wwm-ext, Chinese` [下载地址](https://awesomeopensource.com/project/ymcui/Chinese-BERT-wwm)

- ERNIE1.0 中文预训练模型: `ERNIE 1.0 Base for Chinese` [下载地址](https://github.com/nghuyong/ERNIE-Pytorch)




## 使用

先下载所需模型对应的预训练模型

```bash
# BERT
python run.py --model bert

# ROBERTA
python run.py --model roberta

# ERNIE1.0
python run.py --model ernie
```

- `--cuda:` 使用GPU加速训练，不添加默认使用CPU训练



## 参数

部分超参定义在模型文件内，公用超参定义在`config.py`内。



## 相关论文

[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [论文链接](https://arxiv.org/pdf/1810.04805.pdf)

[2] RoBERTa: A Robustly Optimized BERT Pretraining Approach [论文链接](https://arxiv.org/pdf/1907.11692.pdf)

[3] ERNIE: Enhanced Representation through Knowledge Integration [论文链接](https://arxiv.org/pdf/1904.09223.pdf)

