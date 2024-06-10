# coding: UTF-8
import math
from importlib import import_module

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from run import basic_path
from utils import get_time_dif


def init_network(model):
    for name, param in model.named_parameters():
        if name.startswith('token_embedding'):
            continue  # 跳过词嵌入层

        if isinstance(param, nn.Linear):
            if 'attn' in name:
                # 注意力层的Linear层
                nn.init.xavier_uniform_(param.weight)
            else:
                # FFN层
                nn.init.kaiming_uniform_(param.weight, a=math.sqrt(5))

            if 'bias' in name:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(param.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(param.bias, -bound, bound)
        # Layer Norm参数？
        elif isinstance(param, nn.LayerNorm):
            nn.init.ones_(param.weight)
            nn.init.zeros_(param.bias)


def train(config, model, train_loader, dev_loader, test_loader):
    print("Start training...")
    start_time = time.time()
    model.train()
    model.apply(init_network)

    model = model.to('cuda')
    # 设置emb
    # model.token_embedding.weight.requires_grad = False
    # model.token_embedding.weight.data.copy_(config.embedding_pretrained)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to('cuda'), target.to('cuda')
            outputs = model(data)
            model.zero_grad()
            loss = F.cross_entropy(outputs, target.long())  # 64 * 10  64
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = target.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_loader)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = ('Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {'
                       '4:>6.2%},  Time: {5} {6}')
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
        epoch_end_time = time.time()
        # 记录这一个epoch花费时间:
        print("number: " + str(epoch + 1) + "cost: " + str(epoch_end_time - epoch_start_time))
    test(config, model, test_loader)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    try:
        with torch.no_grad():
            for texts, labels in data_iter:
                texts, labels = texts.to('cuda'), labels.to('cuda')
                outputs = model(texts)
                if test:
                    print('test eval')
                    print("------------------------------")
                    print(outputs.shape, labels.shape)
                    print("------------------------------")
                loss = F.cross_entropy(outputs, labels.long())
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)
    except ValueError:
        pass
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)


def test(config, model, test_iter):
    print("Start testing...")
    # test加载最好
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

