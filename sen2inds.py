#-*- coding: utf_8 -*-
from tqdm import tqdm
import jieba
import random


trainFile = 'data/train.txt'
trainDataVecFile = 'traindata_vec.txt'


devFile = 'data/dev.txt'
devDataVecFile = 'devdata_vec.txt'

labelFile = 'data/label.txt'
stopwordFile = 'data/stopword.txt'

wordLabelFile = 'wordLabel.txt'

maxLen = 20


def read_labelFile(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')
    label_w2n = {}
    label_n2w = {}
    for line in tqdm(data,desc='read label'):
        line = line.split(' ')
        name_w = line[0]
        name_n = int(line[1])
        label_w2n[name_w] = name_n
        label_n2w[name_n] = name_w

    return label_w2n, label_n2w


def read_stopword(file):
    data = open(file, 'r', encoding='utf_8').read().split('\n')

    return data


def get_worddict(file):
    datas = open(file, 'r', encoding='utf_8').read().split('\n')
    datas = list(filter(None, datas))
    word2ind = {}
    for line in tqdm(datas,desc="get_worddict"):
        line = line.split(' ')
        word2ind[line[0]] = int(line[1])

    ind2word = {word2ind[w]:w for w in word2ind}
    return word2ind, ind2word


def json2txt():
    label_dict, label_n2w = read_labelFile(labelFile)
    word2ind, ind2word = get_worddict(wordLabelFile)
    stoplist = read_stopword(stopwordFile)

    #train data to vec
    traindataTxt = open(trainDataVecFile, 'w')
    datas = open(trainFile, 'r', encoding='utf_8').readlines()
    datas = list(filter(None, datas))
    random.shuffle(datas)
    for line in tqdm(datas,desc="traindata to vec"):
        line = line.replace('\n','').split('\t')
        cla = line[1]
        cla_ind = label_dict[cla]
        title_seg = jieba.cut(line[0], cut_all=False)
        title_ind = [cla_ind]
        for w in title_seg:
            if w in stoplist:
                continue
            title_ind.append(word2ind[w])
        length = len(title_ind)
        if length > maxLen + 1:
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')

    #dev data to vec
    traindataTxt = open(devDataVecFile, 'w')
    datas = open(devFile, 'r', encoding='utf_8').readlines()
    datas = list(filter(None, datas))
    random.shuffle(datas)
    for line in tqdm(datas, desc="dev to vec"):
        line = line.replace('\n', '').split('\t')
        cla = line[1]
        cla_ind = label_dict[cla]
        title_seg = jieba.cut(line[0], cut_all=False)
        title_ind = [cla_ind]
        for w in title_seg:
            if w in stoplist:
                continue
            title_ind.append(word2ind[w])
        length = len(title_ind)
        if length > maxLen + 1:
            title_ind = title_ind[0:21]
        if length < maxLen + 1:
            title_ind.extend([0] * (maxLen - length + 1))
        for n in title_ind:
            traindataTxt.write(str(n) + ',')
        traindataTxt.write('\n')


def main():
    json2txt()


if __name__ == "__main__":
    main()
