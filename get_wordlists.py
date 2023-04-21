# -*- coding: utf-8 -*-
'''
将训练数据使用jieba分词工具进行分词。并且剔除stopList中的词。
得到词表：
        词表的每一行的内容为：词 词的序号 词的频次
'''

import json
import jieba
from tqdm import tqdm

trainFile = 'data/train.txt'
devFile = "data/dev.txt"
stopwordFile = 'data/stopword.txt'
wordLabelFile = 'wordLabel.txt'
lengthFile = 'length.txt'


def read_stopword(file):
    datas = open(file, 'r', encoding='utf_8').readlines()
    datas = [data.replace('\n', '') for data in datas]
    return datas


def main():
    worddict = {}
    stoplist = read_stopword(stopwordFile)
    len_dic = {}
    data_num = 0
    # trainFile
    datas = open(trainFile, 'r', encoding='utf_8').readlines()
    data_num += len(datas)
    datas = list(filter(None, datas))
    for line in tqdm(datas, desc='traindata word to label'):
        line = line.replace('\n', '').split('\t')
        title_seg = jieba.cut(line[0], cut_all=False)
        length = 0
        for w in title_seg:
            if w in stoplist:
                continue
            length += 1
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1
        if length in len_dic:
            len_dic[length] += 1
        else:
            len_dic[length] = 1

    # devFile
    datas = open(devFile, 'r', encoding='utf_8').readlines()
    datas = list(filter(None, datas))
    data_num += len(datas)
    for line in tqdm(datas, desc='devdata word to label'):
        line = line.replace('\n', '').split('\t')
        title_seg = jieba.cut(line[0], cut_all=False)
        length = 0
        for w in title_seg:
            if w in stoplist:
                continue
            length += 1
            if w in worddict:
                worddict[w] += 1
            else:
                worddict[w] = 1
        if length in len_dic:
            len_dic[length] += 1
        else:
            len_dic[length] = 1

    wordlist = sorted(worddict.items(), key=lambda item: item[1], reverse=True)
    f = open(wordLabelFile, 'w', encoding='utf_8')
    # ind = 0
    ind = 1
    for t in wordlist:
        d = t[0] + ' ' + str(ind) + ' ' + str(t[1]) + '\n'
        ind += 1
        f.write(d)

    for k, v in len_dic.items():
        len_dic[k] = round(v * 1.0 / data_num, 3)
    len_list = sorted(len_dic.items(), key=lambda item: item[0], reverse=True)
    f = open(lengthFile, 'w')
    for t in len_list:
        d = str(t[0]) + ' ' + str(t[1]) + '\n'
        f.write(d)


if __name__ == "__main__":
    main()
