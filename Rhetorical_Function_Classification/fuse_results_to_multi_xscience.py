#!/usr/bin/python
# -*- coding: utf-8 -*-

import json
import nltk
from tqdm import tqdm 
from collections import Counter

if __name__ == '__main__':
    path_xscience = r'/data/run01/scv7414/wpc/Other_Dataset/tas2/train_wcite.json'
    path_input = r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/tas2/train_output.json'
    path_write = r'/data/run01/scv7414/wpc/Other_Dataset/tas2/train_wcite_rhelabel.json'

    with open(path_xscience,'r',encoding='utf-8') as f, open(path_input,'r',encoding='utf-8') as g:
        lines_x = f.readlines()
        lines_in = g.readlines()
        
        dics = []
        for line_x, line_in in tqdm(zip(lines_x,lines_in)):
            line_x = json.loads(line_x)
            line_in = json.loads(line_in)

            line_x['sum_rhetorical_role'] = line_in['label']
            sens = line_x['summary']
            sens = nltk.sent_tokenize(sens)
            if len(sens) != len(line_in['label']):
                sens = sens[:len(line_in['label'])]
                sens = ' '.join(sens)
                line_x['summary'] = sens

            dics.append(line_x)

    with open(path_write,'w',encoding='utf-8') as t:
        for dic in dics:
            json.dump(dic,t)
            t.write('\n')
