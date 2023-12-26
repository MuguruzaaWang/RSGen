import sys 
sys.path.append("..")

import torch
import dgl
import numpy as np
import json
import pickle
import random
from itertools import combinations
from collections import Counter
from tools.logger import *
from utils.logging import init_logger, logger
from module import vocabulary
from module import data
from torch.autograd import Variable
from collections import Counter
from nltk.corpus import stopwords
import nltk
import re
from transformers import  AutoTokenizer, AutoModel

import time

import pdb

FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

oracle_labels = ["General descriptions of the topic","Reference to current state of knowledge","General reference to previous research or scholarship: research objective","General reference to previous research or scholarship: approaches taken","General reference to previous research or scholarship: about results","Reference to single investigations in the past:  about objective","Reference to single investigations in the past: about method","Reference to single investigations in the past: about result","Summarize the above references","Other reference purpose","Describing the objective","Describing the motivation","Describing used methods","Describing the results","Explaining the method relationship between own work and references","Explaining the objective relationship between own work and references","Explaining the result relationship between own work and references","Explaining the inadequacies of previous studies","Explain the significance of references","Other comments","Signalling Transition","Other functional sentences","Not sure"]


def load_to_cuda(batch,device):
    batch = {'sentences': batch['sentences'].to(device, non_blocking=True), 'segs':batch['segs'].to(device, non_blocking=True), 'label': batch['label'].to(device, non_blocking=True), 'raw_sen':batch['raw_sen']}

    return batch 

def readJson(fname):
    data = []
    with open(fname, 'r',encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readText(fname):
    data = []
    with open(fname, 'r', encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def write_txt(batch, seqs, w_file, args):
    # converting the prediction to real text.
    ret = []
    for b, seq in enumerate(seqs):
        txt = []
        for token in seq:
            if int(token) not in [args.wordvocab.word2id(x) for x in ['<PAD>', '<BOS>', '<EOS>']]:
                txt.append(args.wordvocab.id2word(int(token)))
            if int(token) == args.wordvocab.word2id('<EOS>'):
                break
        w_file.write(' '.join([str(x) for x in txt])+'\n')
        ret.append([' '.join([str(x) for x in txt])])
    return ret 


def replace_ent(x, ent, V):
    # replace the entity
    mask = x>=V
    if mask.sum()==0:
        return x
    nz = mask.nonzero()
    fill_ent = ent[nz, x[mask]-V]
    x = x.masked_scatter(mask, fill_ent)
    return x


###这是建立长度为lens的mask矩阵的操作
def len2mask(lens, device):
    #得到最大的长度n
    max_len = max(lens)
    #构造维度为[len(lens),maxlen]的矩阵
    mask = torch.arange(max_len, device=device).unsqueeze(0).expand(len(lens), max_len)
    ####最终会得到类似[[ 0, 0, 0，1，1],
        #[0, 0, 1, 1, 1],
        #[0, 1, 1, 1, 1]]的矩阵
    #作者这里用0来表示实际的单词，用1来填充
    mask = mask >= torch.LongTensor(lens).to(mask).unsqueeze(1)
    return mask


### for roberta, use pad_id = 1 to pad tensors.
def pad(var_len_list, out_type='list', flatten=False):
    if flatten:
        lens = [len(x) for x in var_len_list]
        var_len_list = sum(var_len_list, [])
    max_len = max([len(x) for x in var_len_list])
    if out_type=='list':
        if flatten:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list], lens
        else:
            return [x+['<PAD>']*(max_len-len(x)) for x in var_len_list]
    if out_type=='tensor':
        if flatten:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0), lens
        else:
            return torch.stack([torch.cat([x, \
            torch.ones([max_len-len(x)]+list(x.shape[1:])).type_as(x)], 0) for x in var_len_list], 0)

def pad_sent_entity(var_len_list, pad_id,bos_id,eos_id, flatten=False):
    def _pad_(data,height,width,pad_id,bos_id,eos_id):
        rtn_data = []
        for para in data:
            if torch.is_tensor(para):
                para = para.numpy().tolist()
            if len(para) > width:
                para = para[:width]
            else:
                para += [pad_id] * (width - len(para))
            rtn_data.append(para)
        rtn_length = [len(para) for para in data]
        x = []
        '''
        x.append(bos_id)
        x.append(eos_id)
        '''
        x.extend([pad_id] * (width))
        rtn_data = rtn_data + [x] * (height - len(data))
        # rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        if len(rtn_data) == 0:
            rtn_data.append([])
        return rtn_data, rtn_length
    
    if flatten:
        var_len = [len(x) for x in var_len_list]
        max_nsent = max(var_len)
        max_ntoken = max([max([len(p) for p in x]) for x in var_len_list])
        _pad_var_list = [_pad_(ex, max_nsent, max_ntoken, pad_id, bos_id, eos_id) for ex in var_len_list]
        pad_var_list = torch.stack([torch.tensor(e[0]) for e in _pad_var_list])
        return pad_var_list, var_len

    else:
        max_nsent = len(var_len_list)
        max_ntoken = max([len(x) for x in var_len_list])
        
        _pad_var_list = _pad_(var_len_list, max_nsent,max_ntoken, pad_id, bos_id, eos_id)
        pad_var_list = torch.tensor(_pad_var_list[0])
        return pad_var_list

def pad_edges(batch_example):
    max_nsent = max([len(ex.raw_sent_input) for ex in batch_example])
    max_nent = max([len(ex.raw_ent_text) for ex in batch_example])
    edges = torch.zeros(len(batch_example),max_nsent,max_nent)
    for index,ex in enumerate(batch_example):
        for key in ex.entities:
            if int(key) >= ex.doc_max_len:
                break
            if ex.entities[key] != []:
                for x in ex.entities[key]:
                    #e = at_least(x.lower().split())
                    e = at_least(x.lower())
                    entNo = ex.raw_ent_text.index(e)
                    sentNo = int(key)

                    edges[index][sentNo][entNo] = 1
    return edges

def at_least(x):
    # handling the illegal data
    if len(x) == 0:
        return ['<UNK>']
    else:
        return x

class Example(object):
    def __init__(self,sentences, label, sent_max_len, doc_max_len, docset_max_len, tokenizer):
        self.tokenizer = tokenizer
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len
        self.max_pos = 512
        self.raw_sen = sentences

        self.tgt_ori = ' '.join(sentences)
        self.tgt_ori = re.sub(r'@cite\_\d{1,3}','@cite',self.tgt_ori)
        self.sentences_idxs, self.seg_ids = self.token_sent(self.tgt_ori)
        
        label_dict = { lab:index for index,lab in enumerate(oracle_labels)}
        self.label = [label_dict[i]  for i in label]
        # self.src_subtoken_idxs, self.src_segments_ids = self.token_sent(target)

        # self.tgt_str = self.tgt_subtoken_idxs
        # self.src_str = target
        # self.refpapers = []
        # for key in references:
        #     ref = references[key]['abstract']
        #     if ref != "":
        #         self.refpapers.append(ref)
        #         if len(self.refpapers) >= self.docset_max_len:
        #             break
        # self.refs_ids = []
        # self.refs_segs = []
        # for ref in self.refpapers:
        #     src_subtoken_idxs, segments_ids = self.token_sent(ref)
        #     self.refs_ids.append(src_subtoken_idxs)
        #     self.refs_segs.append(segments_ids)

        # end_id = [self.refs_ids[0][-1]]

        # if len(self.tgt_subtoken_idxs) > self.max_pos:
        #     self.tgt_subtoken_idxs = self.tgt_subtoken_idxs[:self.max_pos-1] + [2]

        # if len(self.src_subtoken_idxs) > self.max_pos:
        #     self.src_subtoken_idxs = self.src_subtoken_idxs[:self.max_pos-1] + end_id
        #     self.src_segments_ids = self.src_segments_ids[:self.max_pos]

        # self.refs_ids = [each[:self.max_pos - 1] + end_id if len(each) > self.max_pos else each for each in self.refs_ids]
        # self.refs_segs = [each[:self.max_pos] if len(each) > self.max_pos else each for each in self.refs_segs]

    def __str__(self):
        return '\n'.join([str(k)+':\t'+str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_text)

    def token_sent(self, ref):
        ref = nltk.sent_tokenize(ref)
        ref = ref[:self.doc_max_len]
        ref = [each.lower().split()[:50] for each in ref]
        ref_txt = [' '.join(sent) for sent in ref]
        text = ' {} {} '.format(self.sep_token, self.cls_token).join(ref_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:500]
        src_subtokens = [self.cls_token] + src_subtokens + [self.sep_token]

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        # while len(src_subtoken_idxs) < 200:
        #     src_subtoken_idxs.append(0)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        # while len(segments_ids) < 200:
        #     segments_ids.append(0)
        return src_subtoken_idxs, segments_ids


class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, text_path, tokenizer, sent_max_len, doc_max_len, docset_max_len, device=None):
        super(ExampleSet, self).__init__()
        self.device = device
        self.tokenizer = tokenizer

        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.pad_token = '[PAD]'
        self.tgt_bos = '[unused0]'
        self.tgt_eos = '[unused1]'
        self.tgt_sent_split = '[unused2]'
        self.sep_vid = self.tokenizer.vocab[self.sep_token]
        self.cls_vid = self.tokenizer.vocab[self.cls_token]
        self.pad_vid = self.tokenizer.vocab[self.pad_token]
        self.tgt_bosid = self.tokenizer.vocab[self.tgt_bos]
        self.tgt_eosid = self.tokenizer.vocab[self.tgt_eos]

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.json_text_list = readJson(text_path) ###将训练数据读出来（text: , sentences: ,label:）
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小

    def get_example(self, index):
        json_text = self.json_text_list[index]
        example = Example(json_text['sentences'], json_text['label'], self.sent_max_len, self.doc_max_len, self.docset_max_len, self.tokenizer)
        return example
    
    def pad_label_m(self, label_matrix):
        label_m = label_matrix[:self.doc_max_len, :self.doc_max_len]
        N, m = label_m.shape
        if m < self.doc_max_len:
            pad_m = np.zeros((N, self.doc_max_len - m))
            return np.hstack([label_m, pad_m])
        return label_m

    def __getitem__(self, index):
        item = self.get_example(index)
        #enc_sent_input_pad是包含所有经过pad后的句子列表，这一步是对句子进行裁剪，只取前max个句子
        ex_data = self.get_tensor(item)
        return ex_data

    def __len__(self):
        return self.size

    def get_tensor(self, ex):
        _cached_tensor = {'sentences':torch.LongTensor(ex.sentences_idxs), 'segs':torch.LongTensor(ex.seg_ids), 'label':torch.LongTensor(ex.label),'raw_sen':ex.raw_sen}
        return _cached_tensor

    def batch_fn(self, samples):
        batch_sentences, batch_segs, batch_label, batch_rawsen = [], [], [], []
        #batch_ex = map(list, zip(*samples))
        for ex_data in samples:
            if ex_data != {}:
                batch_sentences.append(ex_data['sentences'])
                batch_segs.append(ex_data['segs'])
                batch_label.append(ex_data['label'])
                batch_rawsen.append(ex_data['raw_sen'])
        batch_sentences = pad_sent_entity(batch_sentences, self.pad_vid,self.tgt_bosid,self.tgt_eosid, flatten = False)
        batch_segs = pad_sent_entity(batch_segs, self.pad_vid,self.tgt_bosid,self.tgt_eosid, flatten = False)
        batch_label = pad_sent_entity(batch_label, 104,self.tgt_bosid,self.tgt_eosid, flatten = False)

        
        return {'sentences': batch_sentences, 'segs':batch_segs, 'label': batch_label, 'raw_sen':batch_rawsen}
        
if __name__ == '__main__' :
    pass

