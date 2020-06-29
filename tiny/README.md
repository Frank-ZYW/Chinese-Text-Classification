# Tiny

包含TextCNN、TextRCNN、FastText等一些经典的轻量网络模型。



## 依赖

- python 3.7
- pytorch 1.5
- tqdm
- sklearn
- fasttext
- pkuseg



## 目录结构

```
├── config.py
├── data
│   ├── dataset
│   │   ├── dev.json
│   │   ├── labels.json
│   │   ├── test.json
│   │   └── train.json
│   ├── pretrained
│   └── pretreatment
├── dataloader.py
├── fast_text.py
├── models
│   ├── TextCNN.py
│   └── TextRCNN.py
├── README.md
├── results
├── run.py
├── test.py
├── train.py
└── utils.py
```

#### 主要目录

- `data/dataset:` 存放原始数据集，原始数据集已随项目上传
- `data/pretrained:` 存放预训练词向量模型文件，可从[Chinese Word Vectors 中文词向量](https://github.com/Embedding/Chinese-Word-Vectors)项目下载
- `data/pretreatment:` 存放数据预处理结果文件
- `results:` 模型训练结果存储
- `models:` 模型定义及部分超参



## 使用

***FastText使用官方库实现，参阅[FastText官网](https://fasttext.cc/)***

```bash
# FastText
python fast_text.py --word

# TextCNN
python run.py --model TextCNN

# TextRCNN
python run.py --model TextRCNN
```

- `--word:` 使用词分割，不添加则使用字符分割

对TextCNN、TextRCNN有额外参数选项:

- `--emb:` 词向量，默认random(随机词向量)，使用预训练词向量请放置在相应路径并填写文件名，例如`embedding_SougouNews.npz`
- `--cuda:` 使用GPU加速训练，不添加默认使用CPU训练



## 参数

部分超参定义在模型文件内，公用超参定义在`config.py`内。



## 相关论文

[1] Convolutional Neural Networks for Sentence Classification [论文链接](https://arxiv.org/pdf/1408.5882.pdf)
[2] Recurrent Convolutional Neural Networks for Text Classification [论文链接](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)
[3] Bag of Tricks for Efficient Text Classification [论文链接](https://arxiv.org/pdf/1607.01759.pdf)