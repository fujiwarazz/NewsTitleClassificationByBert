import time
from importlib import import_module

import torch
from torch.utils.data import DataLoader

from run import basic_path
from utils import get_time_dif, TextDataset, build_vocab


def inference():
    x = import_module('models.Transformer')
    config = x.Config(basic_path)
    model = x.BERT(config.vocab_size, config.embedding_size, 128, config.head_num, 2, config.dropout,
                   config.embedding_pretrained)
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    vocab, config.vocab_size = build_vocab(config)
    dataset = TextDataset(config, vocab, phase="infer")
    dataLoader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    for input, _ in dataLoader:
        output = model(input)
        print(output)
        predict = torch.max(output.data, 1)[1].cpu().numpy()
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        print(predict,_)


if __name__ == '__main__':
    inference()
