# coding: UTF-8
import torch
import time
from dataloader import load_and_cache_examples
from test import test
from train import train
from config import BaseConfig
import numpy as np
import argparse
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification


'''
Models: bert, ernie, roberta
'''


parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', default='bert', type=str, help='choose a model in bert, ernie, roberta')
parser.add_argument('--cuda', action='store_true', help='True use GPU, False use CPU')
args = parser.parse_args()


if __name__ == '__main__':

    model_name = args.model
    base_config = BaseConfig(args.cuda, model_name)
    model_config = BertConfig.from_pretrained(
        base_config.pretrained_path + '/config.json',
        num_labels=base_config.label_number
    )
    tokenizer = BertTokenizer.from_pretrained(base_config.pretrained_path, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(base_config.pretrained_path, config=model_config)
    model.to(base_config.device)

    np.random.seed(42)
    torch.manual_seed(42)

    start_time = time.time()
    print("Loading data...")
    train_dataset = load_and_cache_examples(base_config, model_name, tokenizer, data_type='train')
    dev_dataset = load_and_cache_examples(base_config, model_name, tokenizer, data_type='dev')

    # train
    print('\nStart training...')
    train(base_config, model, tokenizer, train_dataset, dev_dataset)

    # test
    # Load a trained model and vocabulary that you have fine-tuned
    tokenizer = BertTokenizer.from_pretrained(base_config.output_dir, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(base_config.output_dir)
    test_dataset = load_and_cache_examples(base_config, model_name, tokenizer, data_type='test')
    test(base_config, model, test_dataset)
