# coding: UTF-8
import torch
import json
import numpy as np
from os.path import join
from tqdm import tqdm


def get_class_list(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]


def get_pretrained(emb, file_path):
    return torch.tensor(np.load(file_path)["embeddings"].astype('float32')) if emb != 'random' else None


class BaseConfig(object):

    """配置参数"""

    def __init__(self, emb, word, cuda):
        # dataset
        dataset_dir = 'data/dataset'
        self.ori_train_path = join(dataset_dir, 'train.json')                       # 原始训练集
        self.ori_dev_path = join(dataset_dir, 'dev.json')                           # 原始验证集
        self.ori_test_path = join(dataset_dir, 'test.json')                         # 原始测试集
        self.ori_label_path = join(dataset_dir, 'labels.json')                      # 原始标签

        # pretreatment
        data_dir = 'data/pretreatment'
        self.train_path = join(data_dir, 'train.txt')                               # 预处理训练集
        self.dev_path = join(data_dir, 'dev.txt')                                   # 预处理验证集
        self.test_path = join(data_dir, 'test.txt')                                 # 预处理测试集
        self.class_list = get_class_list(self.ori_label_path)                       # 标签列表
        self.num_classes = len(self.class_list)                                     # 标签数

        # vocab
        self.vocab_path = 'results/vocab.pkl'                                       # 词表保存路径
        self.min_freq = 5                                                           # 最小词频
        self.use_word = word                                                        # 使用词/字分句
        self.pad_size = 100                                                         # 每句话处理成的长度(短填长切)
        self.n_vocab = 0                                                            # 词表大小，运行时赋值
        self.pretrained = get_pretrained(emb, 'data/pretrained/' + emb)             # 预训练词向量
        self.emb_dim = 300 if self.pretrained is None else self.pretrained.size(1)  # 词向量维度

        # train
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available and cuda else 'cpu')    # 使用cpu/gpu训练
        self.require_improvement = 1000                                             # 超过1000batch效果没提升，提前结束训练
