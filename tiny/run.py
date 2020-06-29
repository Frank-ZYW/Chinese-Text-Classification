# coding: UTF-8
import time
import torch
import numpy as np
import argparse

from dataloader import DatasetIterator, DatasetBuilder, Vocab
from test import test
from train import train, init_network
from utils import get_time_dif
from importlib import import_module

parser = argparse.ArgumentParser(description='FastText Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRCNN')
parser.add_argument('--emb', default='random', type=str, help='random or pre_trained model path')
parser.add_argument('--word', action='store_true', help='True for word, False for char')
parser.add_argument('--cuda', action='store_true', help='True use GPU, False use CPU')
args = parser.parse_args()


if __name__ == '__main__':
    x = import_module('models.' + args.model)
    config = x.Config(args.emb, args.word, args.cuda)
    vocab = Vocab(config).getVocab()
    np.random.seed(1)
    torch.manual_seed(1)

    start_time = time.time()
    print("Loading data...")
    dataset_builder = DatasetBuilder(config, vocab)
    train_data = dataset_builder.getDataSet('train')
    test_data = dataset_builder.getDataSet('test')
    dev_data = dataset_builder.getDataSet('dev')
    train_iter = DatasetIterator(train_data, config)
    dev_iter = DatasetIterator(dev_data, config)
    test_iter = DatasetIterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    print('\nStart training...')
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    init_network(model)
    print('Network: \n', model.parameters)
    train(config, model, train_iter, dev_iter)

    # test
    test(config, model, test_iter)
