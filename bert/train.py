# coding: UTF-8
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataloader import collate_fn


def train(config, model, tokenizer, train_data, dev_data):

    data_loader = DataLoader(
        train_data,
        sampler=RandomSampler(train_data),
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )

    start_time = time.time()

    # Prepare optimizer and schedule
    params = model.named_parameters()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        warmup=0.05,
        t_total=len(data_loader) * config.num_epochs
    )

    total_batch = 0                # 记录进行到多少batch
    last_improve = 0               # 记录上次验证集loss下降的batch数
    flag = False                   # 记录是否很久没有效果提升
    dev_best_loss = float('inf')
    model.zero_grad()

    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for step, batch in enumerate(data_loader):
            model.train()
            batch = tuple(batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'token_type_ids': batch[2]}
            outputs = model(**inputs)

            loss = outputs[0]  # model outputs are always tuple
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            model.zero_grad()

            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                label = batch[3].data.cpu()
                predic = torch.max(outputs[1].data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(label, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_data)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(config.output_dir)
                    tokenizer.save_vocabulary(vocab_path=config.output_dir)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def evaluate(config, model, data, test=False):
    """
    评价与损失函数
    """
    data_loader = DataLoader(
        data,
        sampler=SequentialSampler(data),
        batch_size=config.batch_size,
        collate_fn=collate_fn
    )
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            batch = tuple(batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3], 'token_type_ids': batch[2]}
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs[1], batch[3])
            loss_total += loss
            labels = batch[3].data.cpu().numpy()
            predic = torch.max(outputs[1].data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_loader), report, confusion
    return acc, loss_total / len(data_loader)
