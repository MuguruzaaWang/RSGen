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
    path_txt = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/my_data/40testexamples.txt'
    path_json = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/my_data/40testexamples.json'
    path_src = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/my_data/40testsources.json'
    path_w = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/my_data/40testmydata.json'

    labels = ["General descriptions of the topic","Reference to current state of knowledge","General reference to previous research or scholarship: research objective","General reference to previous research or scholarship: approaches taken","General reference to previous research or scholarship: about results","Reference to single investigations in the past:  about objective","Reference to single investigations in the past: about method","Reference to single investigations in the past: about result","Summarize the above references","Other reference purpose","Describing the objective","Describing the motivation","Describing used methods","Describing the results","Explaining the method relationship between own work and references","Explaining the objective relationship between own work and references","Explaining the result relationship between own work and references","Explaining the inadequacies of previous studies","Explain the significance of references","Other comments","Signalling Transition","Other functional sentences","Not sure"]

    num_dicts = {l:0   for l in labels}

    total_label = []
    with open(path_txt,'r',encoding='utf-8') as f, open(path_json, 'r', encoding='utf-8') as g, open(path_src, 'r', encoding='utf-8') as h:
        lines_txt = f.readlines()
        lines_json = g.readlines()
        lines_src =h.readlines()
        index = 1000
        index_pre = 0
        dict_write = []

        txt_list = []
        for line_json in tqdm(lines_json):
            line_json = json.loads(line_json)
            index_now = int(line_json['id'].split('###')[0])
            #text = re.sub(r'@cite\_\d{1,3}','@cite',line_json['text'])
            if index_now != index:
                j = {}
                j['sentences'] = txt_list
                line_txt = lines_txt[index-1000].strip('\n').split('\t')
                label = [i for i in line_txt if i != '']
                j['label'] = label

                for l in label:
                    if l not  in total_label:
                        total_label.append(l)

                line_src = json.loads(lines_src[index-1000])
                j['target_paper'] = line_src['target_paper']
                j['reference'] = line_src['reference']
                dict_write.append(j)

                assert len(txt_list)==len(label), "length not equal {} {}  {}".format(index,len(txt_list),len(label))
                index = index_now
                txt_list = []
            txt_list.append(line_json['text'])

        j = {}
        j['sentences'] = txt_list
        line_txt = lines_txt[index-1000].strip('\n').split('\t')
        label = [i for i in line_txt if i != '']
        j['label'] = label
        assert len(txt_list)==len(label), "length not equal {} {}  {}".format(index,len(txt_list),len(label))

        line_src = json.loads(lines_src[index-1000])
        j['target_paper'] = line_src['target_paper']
        j['reference'] = line_src['reference']
        dict_write.append(j)           

        with open(path_w,'w',encoding='utf-8') as f:
            for i in dict_write:
                json.dump(i,f)
                f.write('\n')