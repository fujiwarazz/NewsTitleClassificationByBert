import os
import pickle as pkl
from importlib import import_module

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import build_vocab, UNK, PAD

basic_path = {
    'dataset_root': 'THUCNews/data',
    'save_path': 'THUCNews/data/save',
    'embedding': 'THUCNews/data/embedding_SougouNews.npz'
}

x = import_module('models.Transformer')
config = x.Config(basic_path)


def build_vocab(config):
    vocab = pkl.load(open('THUCNews/data/vocab.pkl', 'rb'))
    return vocab, len(vocab)


class TextDataset(Dataset):
    def __init__(self, cc, tokenizer=None, phase="train"):
        self.config = cc
        self.tokenizer = lambda x: [y for y in x] if tokenizer is None else tokenizer
        self.phase = phase
        self.vocab, _ = build_vocab(cc)
        self.tokens, self.labels = self.getContentLabels(self.config.pad_size)

    def getContentLabels(self, pad_size=32):
        tokens = []
        labels = []
        if self.phase == "train":
            path = os.path.join(self.config.dataset_path, self.config.train_path)
        elif self.phase == "dev":
            path = os.path.join(self.config.dataset_path, self.config.dev_path)
        else:
            path = os.path.join(self.config.dataset_path, self.config.test_path)

        with open(path, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                if not line:
                    continue
                line = line.strip()
                words_line = []
                content, label = line.split('\t')

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

        tokens = torch.from_numpy(np.array(self.tokens[idx], dtype=float))
        labels = torch.from_numpy(np.array(self.labels[idx], dtype=float))
        return tokens, labels


# test_dataset = TextDataset(config, phase="test")
# test_dataLoader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

dataset = TextDataset(config, phase="sss")
dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

for data, label in dataloader:
    print(data.shape)
    print(label.shape)
    print(data)
    print(label)
    break

