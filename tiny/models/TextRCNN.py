# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BaseConfig


class Config(BaseConfig):

    """ RCNN 配置参数 """

    def __init__(self, emb, word, cuda):
        super(Config, self).__init__(emb, word, cuda)

        # network
        self.save_path = 'results/RCNN_Module.pth'                                  # 模型训练结果保存路径
        self.hidden_size = 256                                                      # lstm隐藏层
        self.num_layers = 1                                                         # lstm层数

        # train
        self.dropout = 1.0                                                          # 随机失活
        self.num_epochs = 10                                                        # epoch数
        self.batch_size = 128                                                       # mini-batch大小
        self.learning_rate = 5e-4                                                   # 学习率


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        if config.pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.emb_dim, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(
            config.emb_dim, config.hidden_size, config.num_layers,
            bidirectional=True, batch_first=True, dropout=config.dropout
        )
        self.max_pool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.emb_dim, config.num_classes)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.max_pool(out).squeeze()
        return self.fc(out)
