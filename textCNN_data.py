from torch.utils.data import Dataset
import random
import numpy as np
from tqdm import tqdm
import torch

import sen2inds


class textCNN_data(Dataset):
    def __init__(self, trainDataFile):
        trainData = open(trainDataFile, 'r').read().split('\n')
        trainData = list(filter(None, trainData))

        res = []
        for data in tqdm(trainData, desc='index to tensor'):
            data = list(filter(None, data.split(',')))
            data = [int(x) for x in data]
            cla = torch.tensor(data[0], dtype=torch.long)
            sentence = torch.tensor(data[1:], dtype=torch.long)
            temp = []
            temp.append(cla)
            temp.append(sentence)
            res.append(temp)

        self.trainData = res

    def __len__(self):
        return len(self.trainData)

    def __getitem__(self, idx):
        data = self.trainData[idx]
        cla = data[0]
        sentence = data[1]

        return cla, sentence


word2ind, ind2word = sen2inds.get_worddict('wordLabel.txt')
label_w2n, label_n2w = sen2inds.read_labelFile('data/label.txt')

textCNN_param = {
    'vocab_size': len(word2ind) + 1,
    'embed_dim': 128,  # 1 x 128 vector
    'class_num': len(label_w2n),
    "kernel_num": 16,
    "kernel_size": [3, 4, 5],
    "dropout": 0.5,
}
dataLoader_param = {
    'batch_size': 64,
    'shuffle': True,
}
