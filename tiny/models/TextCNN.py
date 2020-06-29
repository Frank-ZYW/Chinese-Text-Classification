# coding: UTF-8
import torch
import torch.nn as nn

from config import BaseConfig


class Config(BaseConfig):

    """ TextCNN 配置参数 """

    def __init__(self, emb, word, cuda):
        super(Config, self).__init__(emb, word, cuda)

        # network
        self.save_path = 'results/TextCNN_Module.pth'                               # 模型训练结果保存路径
        self.filter_sizes = (2, 3, 4)                                               # 卷积核尺寸
        self.num_filters = 16                                                       # 卷积核数量(channels数)

        # train
        self.dropout = 0.5                                                          # 随机失活
        self.num_epochs = 10                                                        # epoch数
        self.batch_size = 128                                                       # mini-batch大小
        self.learning_rate = 1e-3                                                   # 学习率


class Model(nn.Module):

    """ TextCNN Module """

    def __init__(self, config):
        super(Model, self).__init__()
        # use pretrained emb or random emb
        if config.pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.emb_dim, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.emb_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * 3, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = [torch.relu(conv(out)).squeeze(3) for conv in self.convs]  # relu activation func
        out = [torch.max_pool1d(h, h.size(2)).squeeze(2) for h in out]   # max pooling
        out = torch.cat(out, 1)                                          # to vector
        out = self.dropout(out)                                          # dropout
        return self.fc(out)                                              # linear layer
