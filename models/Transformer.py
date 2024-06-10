# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config:
    def __init__(self, basic_path):
        self.dataset_path = basic_path['dataset_root']
        self.vocab_path = 'vocab.pkl'
        self.train_path = 'train.txt'
        self.dev_path = 'dev.txt'
        self.test_path = 'test.txt'
        self.inference_path = 'inference.txt'
        self.save_path = basic_path['save_path']
        self.embedding = basic_path['embedding']

        self.embedding_pretrained = torch.tensor(
            np.load(self.embedding)["embeddings"].astype('float32'))  # 4762 * 300
        self.embedding_size = self.embedding_pretrained.size(1) \
            if self.embedding_pretrained is not None else 300
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_list = [x.strip() for x in open(
            self.dataset_path + '/class.txt').readlines()]  # 类别名单
        self.pad_size = 32
        self.num_classes = len(self.class_list)

        self.batch_size = 64
        self.num_epochs = 10
        self.lr = 1e-3
        self.vocab_size = 0
        self.dropout = 0.5
        self.head_num = 5
        self.depth = 6
        self.out_dim = 10
        self.require_improvement = 1000


class AddNorm(nn.Module):
    """
    Transformer的ADD & NORM LAYER
    """

    def __init__(self, normalized_shape, dropout_rate, eps=1e-6):
        super(AddNorm, self).__init__()
        self.eps = eps
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x, y):
        return self.layer_norm(x + self.dropout(y))


class PositionWiseFFN(nn.Module):
    """
    前馈神经
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(PositionWiseFFN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer Encoder Block
    """

    def __init__(self, embed_dim, heads_num, hidden_size, dropout_rate):
        super(TransformerEncoderBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, heads_num, dropout=dropout_rate)
        self.add_norm_1 = AddNorm(embed_dim, dropout_rate)
        self.ffn = PositionWiseFFN(embed_dim, hidden_size, embed_dim)
        self.add_norm_2 = AddNorm(embed_dim, dropout_rate)

    def forward(self, x):
        y = self.add_norm_1(x, self.attn(x, x, x, need_weights=False)[0])
        return self.add_norm_2(y, self.ffn(y))


class BERT(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, heads_num, block_num, dropout_rate, embedding_pretrained,
                 max_len=32, **kwargs):
        super(BERT, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding.from_pretrained(embedding_pretrained, freeze=False)
        # self.token_embedding = nn.Embedding(vocab_size, embed_dim) # 64 * 32
        print("vocab_size", vocab_size)
        self.position_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.1)
        self.Trm_encoder_blocks = nn.Sequential()
        for i in range(block_num):
            self.Trm_encoder_blocks.add_module(f"{i}", TransformerEncoderBlock(
                embed_dim=embed_dim, heads_num=heads_num, hidden_size=hidden_size, dropout_rate=dropout_rate
            ))
        self.output = nn.Linear(embed_dim, 10)

    def forward(self, tokens):
        x = self.token_embedding(tokens) + self.position_embedding  # x： (batch_size,max_length,hidden_size)
        for block in self.Trm_encoder_blocks:
            x = block(x)
        x = self.output(torch.mean(x, dim=1))
        return x
