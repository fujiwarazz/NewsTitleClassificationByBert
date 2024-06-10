# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta

UNK, PAD = '<UNK>', '<PAD>'


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def build_vocab(config):
    vocab = pkl.load(open('THUCNews/data/vocab.pkl', 'rb'))
    return vocab, len(vocab)


class TextDataset(Dataset):
    def __init__(self, config, vocab, tokenizer=None, phase="train"):
        self.config = config
        self.tokenizer = lambda x: [y for y in x] if tokenizer is None else tokenizer
        self.phase = phase
        self.vocab, _ = build_vocab(config)
        self.tokens, self.labels = self.getContentLabels(self.config.pad_size)

    def getContentLabels(self, pad_size=32):
        tokens = []
        labels = []
        if self.phase == "train":
            path = os.path.join(self.config.dataset_path, self.config.train_path)
        elif self.phase == "dev":
            path = os.path.join(self.config.dataset_path, self.config.dev_path)
        elif self.phase == "test":
            path = os.path.join(self.config.dataset_path, self.config.test_path)
        else:
            path = os.path.join(self.config.dataset_path, self.config.inference_path)

        with open(path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                if not line:
                    continue
                line = line.strip()
                words_line = []
                sep = '\t' if self.phase != 'infer' else ','
                content, label = line.split(sep)
                token = self.tokenizer(content)  # word,word,word
                seq_len = len(token)
                if pad_size is not None:
                    if len(token) < pad_size:
                        token += [PAD] * (pad_size - seq_len)
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                for word in token:
                    words_line.append(self.vocab.get(word, self.vocab.get(UNK)))
                tokens.append(words_line)
                labels.append(label)
        return tokens, labels

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        tokens = torch.from_numpy(np.array(self.tokens[idx], dtype=int))
        labels = torch.from_numpy(np.array(self.labels[idx], dtype=int))
        return tokens, labels
