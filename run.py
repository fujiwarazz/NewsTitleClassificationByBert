import time
import torch
import numpy as np
from importlib import import_module
import argparse
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


# --model TextRNN
parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True,
                    help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()

basic_path = {
    'dataset_root': 'THUCNews/data',
    'save_path': 'THUCNews/data/save',
    'embedding': 'THUCNews/data/embedding_SougouNews.npz'
}

if __name__ == '__main__':
    # 获取配置
    model_name = args.model
    if model_name == 'Transformer':
        from utils import TextDataset, build_vocab
    else:
        pass

    x = import_module('models.' + model_name)
    config = x.Config(basic_path)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # load dataset
    vocab, config.vocab_size = build_vocab(config)
    train_dataset = TextDataset(config, vocab, phase="train")
    dev_dataset = TextDataset(config, vocab, phase="dev")
    test_dataset = TextDataset(config, vocab, phase="test")
    # get dataloader
    train_dataLoader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataLoader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataLoader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)
    # train
    #  vocab_size, embed_dim, hidden_size, heads_num, block_num, dropout_rate
    model = x.BERT(config.vocab_size, config.embedding_size, 128, config.head_num, 2, config.dropout,
                   config.embedding_pretrained)

    train(config, model, train_dataLoader, dev_dataLoader, test_dataLoader)
