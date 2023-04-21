import torch
import torch.nn as nn
import time

import torch.optim as optim

from transformer import Transformer
import sen2inds
from textCNN_data import textCNN_data, textCNN_param, dataLoader_param
from torch.utils.data import DataLoader
from multihead_attention import my_model
import os
from torch.nn import functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"



def validation(model, val_dataLoader, device):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for i, (clas, sentences) in enumerate(val_dataLoader):
            try:
                sentences = sentences.type(torch.LongTensor).to(device)
                clas = clas.type(torch.LongTensor).to(device)
                out = model(sentences)
                # out = F.relu(out.squeeze(-3))
                # out = F.max_pool1d(out, out.size(2)).squeeze(2)
                pred = torch.argmax(out, dim=-1)
                correct += (pred == clas).sum()
                total += clas.size()[0]
            except IndexError as e:
                print(i)
                print(clas)
                print(sentences)

    acc = correct / total
    return acc


seed = 1
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



# # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

# init dataset
print('init dataset...')
trainDataFile = 'traindata_vec.txt'
valDataFile = 'devdata_vec.txt'
train_dataset = textCNN_data(trainDataFile)
train_dataLoader = DataLoader(train_dataset,
                              batch_size=dataLoader_param['batch_size'],
                              shuffle=True)

val_dataset = textCNN_data(valDataFile)
val_dataLoader = DataLoader(val_dataset,
                            batch_size=dataLoader_param['batch_size'],
                            # batch size 64
                            shuffle=False)

if __name__ == "__main__":
    # 设置随机种子，保证结果可复现

    # init net
    print('init net...')
    model = my_model()
    print(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("training...")

    model.train()
    best_dev_acc = 0
    embed = nn.Embedding(textCNN_param['vocab_size'],
                         textCNN_param['embed_dim'], padding_idx=1)
    embed.train()
    for epoch in range(100):
        for i, (clas, sentences) in enumerate(train_dataLoader):
            sentences = embed(sentences) # sentences: batch size 64 x sentence length 20 x embed dimension 128
            # 一个字是个128维vector 一句话是个 20x128的2D tensor 一个batch有64句话是个 64x20x128的3D tensor
            out = model(sentences) # out: batch size 64 x class num 4
            out = torch.mean(out, dim=1)
            loss = criterion(out, clas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 10 == 0:
                print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
        model.eval()
        dev_acc = validation(model=model, val_dataLoader=val_dataLoader,
                            device=device)

        # dev_acc = validation(model=multi_atten, val_dataLoader=val_dataLoader,
        #                      device=device)
        # if best_dev_acc < dev_acc:
        #     best_dev_acc = dev_acc
        #     print("save model...")
        #     torch.save(multi_atten.state_dict(), "textcnn.bin")
        #     print("epoch:", epoch + 1, "step:", i + 1, "loss:", loss.item())
        # print("best dev acc %.4f  dev acc %.4f" % (best_dev_acc, dev_acc))
