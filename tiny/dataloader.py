# coding: UTF-8
import os
import json
import torch
import pkuseg
import pickle as pkl
from tqdm import tqdm

UNK, PAD = '<UNK>', '<PAD>'


def label_map(label):
    """
    将不连续label变为连续
    """
    label = int(label) - 100
    if 5 < label < 11:
        label = label - 1
    elif 11 < label:
        label = label - 2
    return str(label)


class Vocab(object):

    """ Vocab """

    def __init__(self, config):
        self.config = config
        self.seg = pkuseg.pkuseg() if self.config.use_word else None

        # load vocab if exist else create
        vocab_path = config.vocab_path
        print('Loading vocab from', vocab_path, ' ...')
        self.vocab = pkl.load(open(vocab_path, 'rb')) if os.path.exists(vocab_path) else self.build_vocab()
        print('Complete! Vocab size: {}'.format(len(self.vocab)))

    def build_vocab(self):
        """
        build vocab
        """
        print("Vocab isn't exist, creating...")
        vocab_dic = {}
        with open(self.config.ori_train_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                json_item = line.strip()
                if not json_item:
                    continue
                json_item = json.loads(json_item)
                content = json_item['sentence']
                keywords = json_item['keywords']
                # word-level:以空格隔开 char-level:单个字符隔开
                words = self._sentence_segment(content, keywords)
                # 统计词频
                for word in words:
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1
            # 删除不到最小词频的词语，按词频降序
            vocab_list = [item for item in vocab_dic.items() if item[1] >= self.config.min_freq]
            # 制作成 {词:索引号} 的字典
            vocab_dic = {word_count[0]: index for index, word_count in enumerate(vocab_list)}
            vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
            # save to disk
            pkl.dump(vocab_dic, open(self.config.vocab_path, 'wb'))
        # dataset pretreatment
        self._dataset_pretreatment(self.config.ori_train_path, self.config.train_path)
        self._dataset_pretreatment(self.config.ori_test_path, self.config.test_path)
        self._dataset_pretreatment(self.config.ori_dev_path, self.config.dev_path)
        print('Dataset pretreatment complete')
        return vocab_dic

    def _sentence_segment(self, sentence, keywords):
        if self.config.use_word:
            # 句子分词后添加,关键词直接添加
            return self.seg.cut(sentence) + keywords.split(',')
        else:
            return [char for char in sentence + keywords]

    def _dataset_pretreatment(self, file_path, save_path):
        contents = []
        with open(file_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                json_item = line.strip()
                if not json_item:
                    continue
                json_item = json.loads(json_item)
                label = '__label__' + label_map(json_item['label'])
                words = self._sentence_segment(json_item['sentence'], json_item['keywords'])
                contents.append(label + ' ' + ' '.join(words))
        with open(save_path, 'w', encoding='UTF-8') as w:
            for item in contents:
                w.writelines(item + '\n')

    def getVocab(self):
        """
        获取词表
        """
        return self.vocab


class DatasetBuilder(object):

    def __init__(self, config, vocab):
        self.config = config
        self.vocab = vocab

    def load_dataset(self, path):
        """
        加载指定路径数据集
        """
        pad_size = self.config.pad_size
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                lin = lin.split(' ')
                words = lin[1:]
                label = lin[0][9:]
                words_line = []
                seq_len = len(words)
                # 将每个句子划分的结果截长补短成定长
                if pad_size:
                    if len(words) < pad_size:
                        words.extend([PAD] * (pad_size - len(words)))
                    else:
                        words = words[:pad_size]
                        seq_len = pad_size
                # word to id
                for word in words:
                    words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
                contents.append((words_line, int(label), seq_len))
        return contents

    def getDataSet(self, dataset_type):
        """
        获取数据集
        """
        if dataset_type == 'train':
            return self.load_dataset(self.config.train_path)
        elif dataset_type == 'test':
            return self.load_dataset(self.config.test_path)
        elif dataset_type == 'dev':
            return self.load_dataset(self.config.dev_path)
        return None


class DatasetIterator(object):

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.batch_size = config.batch_size
        self.device = config.device

        self.n_batches = len(dataset) // self.batch_size                      # number of batch
        self.residue = True if len(dataset) % self.n_batches != 0 else False  # if dataset is divisible
        self.index = 0

    def _to_tensor(self, data):
        """
        build numpy data into torch Tensor
        """
        x = torch.tensor([item[0] for item in data], dtype=torch.long, device=self.device)
        y = torch.tensor([item[1] for item in data], dtype=torch.long, device=self.device)
        seq_len = torch.tensor([item[2] for item in data], dtype=torch.long, device=self.device)
        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]
            self.index += 1
            return self._to_tensor(batches)
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self._to_tensor(batches)

    def __iter__(self):
        return self

    def __len__(self):
        return self.n_batches + 1 if self.residue else self.n_batches
