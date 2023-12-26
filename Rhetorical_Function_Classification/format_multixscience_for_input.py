#!/usr/bin/python
# -*- coding: utf-8 -*-
### 本代码统计标准summary中的句子修辞角色是否存在一定的转移规律


import json
import nltk
from tqdm import tqdm 
from collections import Counter
import pdb
import pandas as pd
import csv
import re

def get_label_dict(path):
    label_dict = {}
    lines = open(path,'r',encoding='utf-8').readlines()
    for index,line in enumerate(lines):
        line = json.loads(line)
        label_dict[line['type']] = index
    return label_dict

if __name__ == '__main__':
    path_json = r'/data/run01/scv7414/wpc/Other_Dataset/tas2/test.json'
    path_w = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/tas2/test_input.json'

    total_dict = []
    with open(path_json,'r',encoding='utf-8') as f:
        lines_json = f.readlines()
        for line in tqdm(lines_json):
            d = {}
            line = json.loads(line)
            summ = line['summary']
            summ = nltk.sent_tokenize(summ)
            d['sentences'] = summ
            d['label'] = ["General descriptions of the topic" for i in range(len(summ))]

            total_dict.append(d)
        
    with open(path_w,'w',encoding='utf-8') as f:
        for i in total_dict:
            json.dump(i,f)
            f.write('\n')