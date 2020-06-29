from fasttext import train_supervised
from dataloader import label_map
from tqdm import tqdm
import pkuseg
import json
import argparse

seg = pkuseg.pkuseg()  # 加载中文分词器


# 显示模型测试结果
def print_results(n, p, r):
    print("N\t" + str(n))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))


# 选择数据段及分词方式
def sentence_segment(use_word, sentence, keywords):
    if use_word:
        return seg.cut(sentence) + keywords.split(',')
    else:
        return [char for char in sentence + keywords]


# 文件读取
def preslove_dataset(use_word, path, save_path):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            json_item = line.strip()
            if not json_item:
                continue
            json_item = json.loads(json_item)
            label = '__label__' + label_map(json_item['label'])
            words = sentence_segment(use_word, json_item['sentence'], json_item['keywords'])
            contents.append(label + ' ' + ' '.join(words))

    with open(save_path, 'w', encoding='UTF-8') as w:
        for item in contents:
            w.writelines(item + '\n')


parser = argparse.ArgumentParser(description='FastText Text Classification')
parser.add_argument('--word', action='store_true', help='True for word, False for char')
args = parser.parse_args()


if __name__ == "__main__":
    train_data = 'data/pretreatment/fasttext_train.txt'
    valid_data = 'data/pretreatment/fasttext_test.txt'

    # pretreatment
    preslove_dataset(args.word, 'data/dataset/test.json', 'data/pretreatment/fasttext_test.txt')
    preslove_dataset(args.word, 'data/dataset/train.json', 'data/pretreatment/fasttext_train.txt')

    # train
    model = train_supervised(
        input=train_data, epoch=25, lr=0.1, wordNgrams=2, verbose=2, minCount=5, minn=3, maxn=6, dim=300)
    # test & save
    print_results(*model.test(valid_data))
    model.save_model("results/model.bin")
