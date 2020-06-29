# Chinese-Text-Classification
基于Pytorch实现的一些经典自然语言处理模型中文短文本分类任务，包含TextCNN，TextRCNN，FastText，BERT，ROBERT以及ERNIE



## 介绍

本项目共包含两个子项目

- [Tiny](https://github.com/Frank-ZYW/Chinese-Text-Classification/tree/master/tiny): 包含TextCNN、TextRCNN、FastText等一些经典的轻量网络模型
- [BERT](https://github.com/Frank-ZYW/Chinese-Text-Classification/tree/master/bert): 包含BERT及其衍生的ROBERTA、ERNIE1.0等一些经典的深度网络模型



## 依赖

参见各子项目要求



## 数据集

数据集来自[CLUE benchmark(中文语言理解测评基准)](https://github.com/CLUEbenchmark/CLUE)提供的TNEW数据集(今日头条中文新闻)的子集

数据集划分:

| 数据集 | 条目  |
| ------ | ----- |
| 训练集 | 53360 |
| 测试集 | 5000  |
| 开发集 | 5000  |

**每条条目包含标签、新闻原句和关键词共3个字段**

```
{"label": "115", "label_desc": "news_agriculture", "sentence": "60岁农村大叔为挣百十元钱工地干的这活，看完眼睛湿润", "keywords": "大叔,农村"}
```

**共15个不同分类**

数据集已随子项目上传，无需下载

#### 更换自己的数据集

- 请依照本项目原始数据集格式整理数据



## 模型效果

各模型在不同数据预处理情况下的最佳得分，取5次测试的最高值，评判标准为Accuracy

|             Type              |  TextCNN   |  TextRCNN  | FastText  |    BERT    |   ROBERT   |   ERNIE    |
| :---------------------------: | :--------: | :--------: | :-------: | :--------: | :--------: | :--------: |
|        sentence(char)         |   49.26%   |   50.51%   |   47.9%   |     -      |     -      |     -      |
|    sentence+keywords(char)    | **58.52%** | **60.30%** |   62.4%   |     -      |     -      |     -      |
|        sentence(word)         |   48.07%   |   49.53%   |   54.7%   |     -      |     -      |     -      |
|    sentence+keywords(word)    |   55.65%   |   56.04%   | **63.9%** |     -      |     -      |     -      |
|     sentence(pretrained)      |     -      |     -      |     -     |   57.19%   |   58.02%   |   57.43%   |
| sentence+keywords(pretrained) |     -      |     -      |     -     | **65.98%** | **66.74%** | **67.20%** |

- Type表示使用的数据段组合以及分词方式，s代指sentence，k代指keywords
- 括号内的文字表示采用的分词方式，char表示按字符分词，word表示按词汇分词，pretrain表示按预训练模型提供的词表分词
- 使用的中文分词工具: 北京大学语言计算与机器学习研究组开源的多领域中文分词工具包 [Pkuseg](https://github.com/lancopku/pkuseg-python)

[**CLUE中文任务基准测评分类排行榜**](https://www.cluebenchmarks.com/classification.html)



## 使用

模型使用及参数配置请参见各子项目使用指导



## 相关论文

[1] Convolutional Neural Networks for Sentence Classification [论文链接](https://arxiv.org/pdf/1408.5882.pdf)

[2] Recurrent Convolutional Neural Networks for Text Classification [论文链接](http://www.nlpr.ia.ac.cn/cip/~liukang/liukangPageFile/Recurrent%20Convolutional%20Neural%20Networks%20for%20Text%20Classification.pdf)

[3] Bag of Tricks for Efficient Text Classification [论文链接](https://arxiv.org/pdf/1607.01759.pdf)

[4] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding [论文链接](https://arxiv.org/pdf/1810.04805.pdf)

[5] RoBERTa: A Robustly Optimized BERT Pretraining Approach [论文链接](https://arxiv.org/pdf/1907.11692.pdf)

[6] ERNIE: Enhanced Representation through Knowledge Integration [论文链接](https://arxiv.org/pdf/1904.09223.pdf)