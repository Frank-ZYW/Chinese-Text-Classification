# coding: UTF-8
import torch
import json
import os
from os.path import join
from tqdm import tqdm


def get_class_list(file_path):
    with open(file_path, 'r', encoding='UTF-8') as f:
        return [json.loads(line.strip())['label_desc'] for line in tqdm(f) if line.strip()]


def create_dir(dirs):
    if not os.path.isdir(dirs):
        os.makedirs(dirs)
    return dirs


def best_learning_rate(model_name):
    return 5e-5 if model_name == 'ernie' else 2e-5


class BaseConfig(object):

    """配置参数"""

    def __init__(self, cuda, model_name):

        # train
        cuda_available = torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_available and cuda else 'cpu')  # 使用cpu/gpu训练
        self.require_improvement = 1000                                           # 超过1000batch效果没提升，提前结束训练
        self.num_epochs = 3                                                       # epoch数
        self.batch_size = 16                                                      # mini-batch大小
        self.learning_rate = best_learning_rate(model_name)                       # 学习率
        self.max_seq_length = 128                                                 # 每句话处理成的长度(短填长切)
        self.pretrained_path = 'data/pretrained/' + model_name                    # 预训练模型路径
        self.output_dir = create_dir('results/' + model_name)                     # 结果保存路径

        # dataset
        self.data_dir = 'data/dataset'
        self.train_path = join(self.data_dir, 'train.json')                       # 训练集
        self.dev_path = join(self.data_dir, 'dev.json')                           # 验证集
        self.test_path = join(self.data_dir, 'test.json')                         # 测试集
        self.label_path = join(self.data_dir, 'labels.json')                      # 标签
        self.class_list = get_class_list(self.label_path)                         # 标签列表
        self.label_number = len(self.class_list)                                  # 标签个数
