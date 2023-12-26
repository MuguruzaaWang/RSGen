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
from module.sampler import MPerClassSampler

import time

import pdb

NODE_TYPE = {'word':0, 'augment_func': 1, 'reference':2, 'target': 3, 'global':4}
FILTERWORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '\/']
FILTERWORD.extend(punctuations)

oracle_labels = ["General descriptions of the topic","Reference to current state of knowledge","General reference to previous research or scholarship: research objective","General reference to previous research or scholarship: approaches taken","General reference to previous research or scholarship: about results","Reference to single investigations in the past:  about objective","Reference to single investigations in the past: about method","Reference to single investigations in the past: about result","Summarize the above references","Other reference purpose","Describing the objective","Describing the motivation","Describing used methods","Describing the results","Explaining the method relationship between own work and references","Explaining the objective relationship between own work and references","Explaining the result relationship between own work and references","Explaining the inadequacies of previous studies","Explain the significance of references","Other comments","Signalling Transition","Other functional sentences","Not sure"]
oracle_dict = {l:index   for index,l in enumerate(oracle_labels)}
rhe_dict = {l:'<type'+str(index+1)+'>' for index,l in enumerate(oracle_labels)}


def load_to_cuda(batch,device):
    batch = {'tarpaper': batch['tarpaper'].to(device,non_blocking=True), 'tarpaper_extend': batch['tarpaper_extend'].to(device,non_blocking=True), \
            'refpaper': batch['refpaper'].to(device,non_blocking=True), 'refpaper_extend': batch['refpaper_extend'].to(device,non_blocking=True),  'article_len':batch['article_len'], 'graph': batch['graph'],\
             'raw_sent_input': batch['raw_sent_input'], 'words':batch['words'].to(device, non_blocking=True),\
             'raw_tgt_text': batch['raw_tgt_text'], 'examples':batch['examples'], 'tgt': batch['tgt'].to(device, non_blocking=True), \
             'sent_num':batch['sent_num'], 'extra_zeros':batch['extra_zeros'], \
             'article_oovs':batch['article_oovs'],'tgt_extend': batch['tgt_extend'].to(device, non_blocking=True),\
             'text': batch['text'].to(device, non_blocking=True),'text_extend': batch['text_extend'].to(device,non_blocking=True),\
             'ref_func': batch['ref_func'].to(device, non_blocking=True),'tar_func': batch['tar_func'].to(device, non_blocking=True)}
    batch['extra_zeros'] = batch['extra_zeros'].to(device, non_blocking=True) if batch['extra_zeros'] != None else batch['extra_zeros']
    
    return batch 




def readJson(fname):
    data = []
    with open(fname, 'r',encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def readJson_cls(fname,label_to_id):
    data = []
    labels = []
    with open(fname, 'r',encoding="utf-8") as f:
        for line in f:
            l = json.loads(line)
            data.append(l)
            labels.append(label_to_id[l['label']])
    return data, labels

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

def find_sentence_boundaries(symbols, input_ids):
    """ Finds the sentence boundaries in summary sentences. """
    assert input_ids.dim() == 2
    input_ids = input_ids.transpose(0,1)
    starts, ends, masks = [], [], []
    for ids in input_ids.tolist():
        end = []
        for i, id in enumerate(ids):
            if id == symbols['EOS'] or id == symbols['EOD']:
                end.append(i)
        ends.append(end)
        starts.append([0] + list(map(lambda x: x + 1, end[:-1])))

    # pad to same length
    starts = pad2(starts, symbols['PAD'])
    ends = pad2(ends, symbols['PAD'])

    return starts, ends


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
def pad_ref_state(refsent_state, ref_context, ref_func,refpaper):
    batch_size, num_doc, num_sent = ref_func.size()
    num_words = ref_context.size(2)
    new_tensor = torch.zeros((batch_size, num_doc, num_sent,refsent_state.size(-1)), \
        dtype=refsent_state.dtype, device=refsent_state.device)
    new_tensor2 = torch.zeros((batch_size, num_doc, num_sent, num_words, refsent_state.size(-1)), \
        dtype=refsent_state.dtype, device=refsent_state.device)
    # paper_tensor = torch.full((batch_size, num_doc, num_sent,num_words), False, dtype=torch.bool\
    #     , device=refsent_state.device)
    paper_tensor = torch.zeros((batch_size, num_doc, num_sent,num_words), \
        dtype=ref_func.dtype, device=refsent_state.device)
    for index in range(batch_size):
        split_list = ref_func[index].ne(0).sum(dim=1).tolist()
        split_list = [x for x in split_list if x > 0]
        refsent = refsent_state[index][:sum(split_list)]
        refsent_tuple  = torch.split(refsent, split_list)
        refcont = ref_context[index][:sum(split_list)]
        refcont_tuple  = torch.split(refcont, split_list)
        paper = refpaper[index][:sum(split_list)]
        refpaper_tuple  = torch.split(paper, split_list)
        for i,size in enumerate(split_list):
            new_tensor[index,i,:size] = refsent_tuple[i]
            new_tensor2[index,i,:size] = refcont_tuple[i]
            paper_tensor[index,i,:size] = refpaper_tuple[i]

    ref_mask = ref_func.ne(0)
    return new_tensor, new_tensor2, ref_mask, paper_tensor

def unpad_ref_state(tar_context,ref_context,tar_func, ref_func, tar_paper, ref_paper):
    tarlen_list = tar_func.ne(0).sum(dim=1).reshape(-1,1)
    reflen_list = ref_func.ne(0).sum(dim=2)
    len_list = torch.cat((tarlen_list,reflen_list), dim=1)

    new_list = []
    wordsum = []
    paper_list = []

    batch_size = tar_context.size(0)
    for index in range(batch_size):
        new_list.append([])
        wordsum.append([])
        paper_list.append([])
        l_list = len_list[index]
        tar_context2 = tar_context[index,:l_list[0]]
        tar_context2 = tar_context2.reshape(-1,tar_context2.size(-1))
        new_list[index].append(tar_context2)
        wordsum[index].append(tar_context2.size(0))
        tar_paper2 = tar_paper[index,:l_list[0]].reshape(-1)
        paper_list[index].append(tar_paper2)
        for doc_ind, l in enumerate(l_list[1:]):
            if l == 0:
                continue
            context = ref_context[index,doc_ind,:l].reshape(-1,tar_context2.size(-1))
            wordsum[index].append(context.size(0))
            new_list[index].append(context)
            paper2 = ref_paper[index,doc_ind,:l].reshape(-1)
            paper_list[index].append(paper2)
    
    total_len = [sum(x) for x in wordsum]
    max_len = max(total_len)
    new_tensor = torch.zeros((batch_size, max_len,ref_context.size(-1)), \
        dtype=ref_context.dtype, device=ref_context.device)
    paper_tensor = torch.zeros((batch_size, max_len), \
        dtype=tar_paper.dtype, device=ref_context.device)
    for index in range(batch_size):
        new_tensor[index,:total_len[index]] = torch.cat(new_list[index],dim=0)
        paper_tensor[index,:total_len[index]] = torch.cat(paper_list[index],dim=0)
    return new_tensor, paper_tensor, wordsum

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

def pad2(data, pad_id):
    """ Pad all lists in data to the same length. """
    width = max(len(d) for d in data)
    return [d + [pad_id] * (width - len(d)) for d in data]


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
        pad_var_list = torch.tensor(_pad_var_list[0]).transpose(0, 1)
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
    def __init__(self, target, tar_function, references, summary, sum_labels,  sent_max_len, doc_max_len, docset_max_len, wordvocab):
        #data format is as follows:
        # text: [[],[],[]] list(list(string)) for multi-document; one per article sentence. each token is separated by a single space
        # entities: {"0":[],"1":[]...}, a dict correponding to all the sentences, one list per sentence
        # relations: list(list()) the inner list correspongding to a 3-element tuple, [ent1, relation, ent2]
        # types: list  one type per entity
        # clusters: list(list) the inner list is the set of all the co-reference lists

        # filterwords are only used in graph building process
        # all the text words should be in the range of word_vocab, or it will be [UNK]

        self.wordvocab = wordvocab
        start_decoding = wordvocab.word2id(vocabulary.START_DECODING)
        stop_decoding = wordvocab.word2id(vocabulary.STOP_DECODING)
        start_sen = wordvocab.word2id(vocabulary.START_DOCUMENT)

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len

        self.func_types = {'background':1,'objective':2,'method':3,'result':4,'other':5}
        self.tarpaper = nltk.sent_tokenize(target)
        self.refpaper = []
        self.ref_func = []
        for key in references:
            ref = references[key]['abstract']
            func = references[key]['ref_function']
            if ref != "":
                self.refpaper.append(nltk.sent_tokenize(ref))
                self.ref_func.append([self.func_types[x] for x in func])
                if len(self.refpaper) >= self.docset_max_len:
                    break
                if len(sum(self.refpaper,[])) >= self.doc_max_len:
                    break

        self.summary = summary
        self.summary = re.sub(r'@cite\_\d{1,3}','@cite',self.summary)

        self.sum_sent = nltk.sent_tokenize(self.summary)

        self.sum_label = [ oracle_dict[i]  for i in sum_labels]
        rhe_words = [rhe_dict[i]  for i in sum_labels]
        rhe_ids = [wordvocab.word2id(w) for w in rhe_words]

        abstract_words = []
        abs_ids = []
        for sent in self.sum_sent:
            words = sent.lower().split()
            abstract_words.append(words)
            abs_ids.append([wordvocab.word2id(w) for w in words])

        self.raw_sent_len = []
        self.raw_sent_input = []

        # process target paper
        self.enc_input = []
        self.raw_input = []
        self.article_len = []
        self.enc_tarpaper_input = []
        self.raw_tarpaper_input = []
        self.tarpaper_func = [self.func_types[x] for x in tar_function]
        self.article_len.append(len(self.tarpaper))
        for sent in self.tarpaper:
            article_words = sent.lower().split()
            if len(article_words) > sent_max_len:
                article_words = article_words[:sent_max_len]
            self.raw_tarpaper_input.append(article_words)
            self.raw_input.append(article_words)
            self.enc_input.append([wordvocab.word2id(w) for w in article_words])
            self.enc_tarpaper_input.append([wordvocab.word2id(w) for w in article_words]) # list of word ids; OOVs are represented by the id for UNK token
            if len(self.raw_tarpaper_input) >= self.doc_max_len:
                break

        assert len(self.tarpaper_func) == len(self.enc_tarpaper_input), "number of targert function %d not equal to that of sentences %d"\
                    %(len(self.tarpaper_func,len(self.enc_tarpaper_input)))

        # process reference papers
        self.enc_refpaper_input = []
        self.raw_refpaper_input = []
        for index,doc in enumerate(self.refpaper):
            docLen = len(doc)
            self.enc_refpaper_input.append([])
            self.article_len.append(docLen)
            for sent in doc:
                article_words = sent.lower().split()
                if len(article_words) > sent_max_len:
                    article_words = article_words[:sent_max_len]
                self.raw_refpaper_input.append(article_words)
                self.raw_input.append(article_words)
                self.enc_input.append([wordvocab.word2id(w) for w in article_words])
                self.enc_refpaper_input[index].append([wordvocab.word2id(w) for w in article_words]) # list of word ids; OOVs are represented by the id for UNK token
            if len(self.raw_refpaper_input) >= self.doc_max_len:
                break

        assert len(sum(self.ref_func,[])) == len(sum(self.enc_refpaper_input,[])), "number of ref function %d not equal to that of sentences %d"%(len(sum(self.ref_func,[])),len(sum(self.enc_refpaper_input,[])))

        self.dec_input, self.target = self.get_dec_inp_targ_seqs(abs_ids, 500, start_sen, start_decoding, stop_decoding, rhe_ids)
        self.dec_len = len(self.dec_input)
        
        self.enc_tarpaper_input_extend = []
        self.enc_refpaper_input_extend = []
        self.article_oovs = []
        
        ###add pointer-generator mode
        for enc_sent in self.raw_tarpaper_input:
            enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_tarpaper_input_extend.append(enc_input_extend_vocab)

        for enc_sent in self.raw_refpaper_input:
            enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_refpaper_input_extend.append(enc_input_extend_vocab)

        self.enc_input_extend = []
        self.article_oovs = []
        for enc_sent in self.raw_input:
            enc_input_extend_vocab, self.article_oovs = data.article2ids(enc_sent, wordvocab, self.article_oovs)
            #self.article_oovs.extend(oovs)
            self.enc_input_extend.append(enc_input_extend_vocab)
        
        # Get a verison of the reference summary where in-article OOVs are represented by their temporary article OOV id
        abs_ids_extend_vocab = data.abstract2ids(abstract_words, wordvocab, self.article_oovs)

        # Overwrite decoder target sequence so it uses the temp article OOV ids
        _, self.target = self.get_dec_inp_targ_seqs(abs_ids_extend_vocab, 500, start_sen, start_decoding, stop_decoding, rhe_ids)
    
    def __str__(self):
        return '\n'.join([str(k)+':\t'+str(v) for k, v in self.__dict__.items()])

    def __len__(self):
        return len(self.raw_text)
    
    def get_dec_inp_targ_seqs(self, sequence, max_len, start_sen, start_id, stop_id, rhe_ids):
        inp = []
        inp.append([start_id])
        target = []
        inp.append(rhe_ids + [start_sen])
        target.append(rhe_ids + [start_sen])
        #target.append([start_decoding] + sequence[0]+[start_sen])
        # for r_id,sent in zip(rhe_ids,sequence):
        #     inp.append([r_id] + sent)
        #     target.append([r_id] + sent)
        inp += sequence
        target += sequence
        target[-1] = target[-1] + [stop_id]

        target = sum(target,[])
        inp = sum(inp,[])

        return inp, target

class ExampleSet(torch.utils.data.Dataset):
    def __init__(self, text_path, wordvocab, sent_max_len, doc_max_len, docset_max_len, device=None):
        super(ExampleSet, self).__init__()
        self.device = device
        self.wordvocab = wordvocab

        self.sent_max_len = sent_max_len
        self.doc_max_len = doc_max_len
        self.docset_max_len = docset_max_len

        logger.info("[INFO] Start reading %s", self.__class__.__name__)
        start = time.time()
        self.json_text_list = readJson(text_path) ###将训练数据读出来（text: , summary: ,label:）
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小

    def get_example(self, index):
        json_text = self.json_text_list[index]
        #e["summary"] = e.setdefault("summary", [])
        example = Example(json_text['target_paper'], json_text['target_function'], json_text['reference'], json_text['summary'],json_text['sum_rhetorical_role'], self.sent_max_len, self.doc_max_len, self.docset_max_len, self.wordvocab)
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
        graph,word_list = self.build_graph(item)
        ex_data = self.get_tensor(item,word_list)
        return graph, ex_data

    def __len__(self):
        return self.size

    def AddWordNode(self, graph,ex): 
        word2nid = {}
        nid2word = []
        nid = 0
        for sent in ex.enc_tarpaper_input:
            for word in sent:
                if word not in word2nid.keys() and self.wordvocab.id2word(word) not in FILTERWORD:
                    word2nid[word] = nid
                    nid2word.append(word)
                    nid += 1
        temp_sents = sum(ex.enc_refpaper_input,[])
        for sent in temp_sents:
            for word in sent:
                if word not in word2nid.keys() and self.wordvocab.id2word(word) not in FILTERWORD:
                    word2nid[word] = nid
                    nid2word.append(word)
                    nid += 1
        w_nodes = len(nid2word)

        ###增加w_nodes个新节点
        graph.add_nodes(w_nodes, {'type': torch.ones(w_nodes) * NODE_TYPE['word']})
        ###对节点的所有特征进行初始化，0初始化，但是不太合理呀
        #graph.set_n_initializer(dgl.init.zero_initializer)

        return word2nid, nid2word

    def MapSent2Aug(self,ex):
        Aug = []
        Aug.append([])
        tartype = set(ex.tarpaper_func)
        Augx = {t:[] for t in tartype}
        for sent,value in zip(ex.enc_tarpaper_input,ex.tarpaper_func):
            Augx[value].extend(sent) 
        for value in Augx.values():
            Aug[0].append(value)

        for index,(reffunc,ref_input) in enumerate(zip(ex.ref_func,ex.enc_refpaper_input)):
            Aug.append([])
            reftype = set(reffunc)
            Augx = {t:[] for t in reftype}
            for sent,value in zip(ref_input,reffunc):
                Augx[value].extend(sent) 
            for value in Augx.values():
                Aug[index+1].append(value)

        return Aug

    def word_edge_aug(self, G, aug_group, aug2nid, word2nid, ex):
        aug_group2 = sum(aug_group,[])
        noempty_aug = [False for i in range(len(aug_group2))]
        for index,sent in enumerate(aug_group2):
            c = Counter(sent) #计数器,统计列表中的单词出现次数
            aug_nid = aug2nid[index] #当前句子的nodeid

            for word, cnt in c.items():###cnt is not needed
                if word in word2nid.keys():
                    noempty_aug[index] = True
                    G.add_edges(word2nid[word], aug_nid)
                    G.add_edges(aug_nid, word2nid[word])
            if noempty_aug[index] == False:
                G.add_edges(0, aug_nid)
                G.add_edges(aug_nid, 0)

        return G

    def aug_edge_paper(self, G, aug_group, aug2nid, tar2nid, ref2nid,ex):
        for i in range(len(aug_group[0])):
            aug_nid = aug2nid[i]
            G.add_edges(tar2nid, aug_nid)
            G.add_edges(aug_nid, tar2nid)
        global_index = i + 1
        for index,group in enumerate(aug_group[1:]):
            for x in group:
                aug_nid = aug2nid[global_index]
                G.add_edges(aug_nid, ref2nid[index])
                G.add_edges(ref2nid[index], aug_nid)
                global_index += 1
        return G 

    def aug_edge(self, G, aug2nid, tarpaper_func, ref_func):
        def get_index1(lst=None, item=''):
            return [index for (index,value) in enumerate(lst) if value == item]
        def find_ngrams(input_list):
            ngrams = []
            for index, ele1 in enumerate(input_list[:-1]):
                for ele2 in input_list[index+1:]:
                    ngrams.append((ele1,ele2))
            return ngrams

        aug_list = []
        aug_list.extend(list(set(tarpaper_func)))
        for x in ref_func:
            aug_list.extend(list(set(x)))

        for label in range(1,6):
            lst = get_index1(aug_list,label)
            if len(lst) < 2:
                continue
            bigrams = find_ngrams(lst)
            for bi in bigrams:
                G.add_edges(aug2nid[bi[0]], aug2nid[bi[1]])
                G.add_edges(aug2nid[bi[1]], aug2nid[bi[0]])

        return G

    def build_graph(self,ex):
        graph = dgl.DGLGraph()
        #graph = dgl.graph()

        graph.set_n_initializer(dgl.init.zero_initializer)

        word2nid, nid2word = self.AddWordNode(graph,ex)
        w_nodes = len(nid2word)

        aug_nodes = len(set(ex.tarpaper_func)) + sum([len(set(x)) for x in ex.ref_func])

        graph.add_nodes(aug_nodes, {'type': torch.ones(aug_nodes) * NODE_TYPE['augment_func']})
        
        aug2nid = [i+w_nodes for i in range(aug_nodes)]
        #taraug2nid = [i+w_nodes for i in range(len(set(ex.tarpaper_func)))]

        graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['target']})
        ref_len = len(ex.enc_refpaper_input)
        #graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['root']})
        graph.add_nodes(ref_len, {'type': torch.ones(ref_len) * NODE_TYPE['reference']})

        graph.add_nodes(1, {'type': torch.ones(1) * NODE_TYPE['global']})
        #graph.add_nodes(len(ex.types), {'type': torch.ones(len(ex.types)) * NODE_TYPE['type']})

        tar2nid = w_nodes + aug_nodes

        ref2nid = [i + w_nodes + aug_nodes + 1 for i in range(ref_len)]
        root2nid = w_nodes + aug_nodes + 1 + ref_len

        aug_group = self.MapSent2Aug(ex)
        assert len(sum(aug_group,[]))==aug_nodes, "num of aug_group not equal to aug_nodes"
        graph = self.word_edge_aug(graph, aug_group, aug2nid, word2nid,ex)
        graph = self.aug_edge_paper(graph, aug_group, aug2nid, tar2nid, ref2nid,ex)
        graph = self.aug_edge(graph, aug2nid, ex.tarpaper_func, ex.ref_func)
        graph.add_edges(tar2nid, ref2nid)
        graph.add_edges(ref2nid, tar2nid)

        graph.add_edges(root2nid, torch.arange(root2nid))

        graph.add_edges(torch.arange(root2nid), root2nid)

        word_list = list(word2nid.keys())
        
        return graph,word_list

    def get_tensor(self, ex, word_list):
        
        ex.enc_refpaper_input2 = sum(ex.enc_refpaper_input,[])
        _cached_tensor = {'tarpaper': [torch.LongTensor(x) for x in ex.enc_tarpaper_input], 'tarpaper_extend':[torch.LongTensor(x) for x in ex.enc_tarpaper_input_extend], \
        'refpaper': [torch.LongTensor(x) for x in ex.enc_refpaper_input2], 'refpaper_extend':[torch.LongTensor(x) for x in ex.enc_refpaper_input_extend], 'summary':ex.summary, 'tgt':torch.LongTensor(ex.dec_input), \
                            'raw_sent_input': ex.raw_sent_input, 'example':ex, 'tgt_extend': torch.LongTensor(ex.target), 'oovs':ex.article_oovs, 'article_len':ex.article_len, 'ref_func':[torch.LongTensor(x) for x in ex.ref_func],\
                            'tar_func':torch.LongTensor(ex.tarpaper_func),\
                            'text': [torch.LongTensor(x) for x in ex.enc_input], 'text_extend': [torch.LongTensor(x) for x in ex.enc_input_extend], 'words':torch.LongTensor(word_list)}
        return _cached_tensor

    def batch_fn(self, samples):
        batch_tarpaper, batch_refpaper, batch_refpaper_extend, batch_raw_sent_input, batch_examples, \
            batch_tgt, batch_tgt_extend,batch_raw_tgt_text,batch_oovs, batch_tarpaper_extend, \
            batch_art_len, batch_text,batch_text_extend,batch_words,batch_graph2, \
            batch_reffunc,batch_tarfunc  =  [], [], [], [], [], [], [], [],[],[],[],[],[],[],[],[],[]
        #batch_ex = map(list, zip(*samples))
        batch_graph, batch_ex = map(list, zip(*samples))
        for graph,ex_data in zip(batch_graph, batch_ex):
            if ex_data != {}:
                batch_tarpaper.append(ex_data['tarpaper'])
                batch_raw_sent_input.append(ex_data['raw_sent_input'])
                batch_examples.append(ex_data['example'])
                batch_tgt.append(ex_data['tgt'])
                batch_raw_tgt_text.append(ex_data['summary'])
                batch_oovs.append(ex_data['oovs'])
                batch_tgt_extend.append(ex_data['tgt_extend'])
                batch_tarpaper_extend.append(ex_data['tarpaper_extend'])
                batch_refpaper.append(ex_data['refpaper'])
                batch_refpaper_extend.append(ex_data['refpaper_extend'])
                batch_art_len.append(ex_data['article_len'])
                batch_text.append(ex_data['text'])
                batch_graph2.append(graph)
                batch_text_extend.append(ex_data['text_extend'])
                batch_words.append(ex_data['words'])
                batch_reffunc.append(ex_data['ref_func'])
                batch_tarfunc.append(ex_data['tar_func'])


        pad_id = self.wordvocab.word2id('<PAD>')
        bos_id = self.wordvocab.word2id('<BOS>')
        eos_id = self.wordvocab.word2id('<EOS>')
        
        batch_tgt = pad_sent_entity(batch_tgt, pad_id,bos_id,eos_id, flatten = False)
        batch_tgt_extend = pad_sent_entity(batch_tgt_extend, pad_id,bos_id,eos_id, flatten = False)
        batch_tarpaper,_sent_num = pad_sent_entity(batch_tarpaper, pad_id,bos_id,eos_id, flatten = True)
        batch_tarpaper_extend,_sent_num = pad_sent_entity(batch_tarpaper_extend, pad_id,bos_id,eos_id, flatten = True)
        batch_refpaper,sent_num = pad_sent_entity(batch_refpaper, pad_id,bos_id,eos_id, flatten = True)
        batch_refpaper_extend,sent_num = pad_sent_entity(batch_refpaper_extend, pad_id,bos_id,eos_id, flatten = True)
        batch_text,sent_num = pad_sent_entity(batch_text, pad_id,bos_id,eos_id, flatten = True)
        batch_text_extend,sent_num = pad_sent_entity(batch_text_extend, pad_id,bos_id,eos_id, flatten = True)
        batch_reffunc,_ = pad_sent_entity(batch_reffunc, 0,bos_id,eos_id, flatten = True)
        batch_tarfunc = pad_sent_entity(batch_tarfunc, 0,bos_id,eos_id, flatten = False).transpose(0, 1)

        batch_words = pad(batch_words, out_type='tensor')
        
        max_art_oovs = max([len(oovs) for oovs in batch_oovs])
        extra_zeros = None
        batch_size = batch_tgt.shape[1]
        if max_art_oovs > 0:
            #extra_zeros = Variable(torch.zeros((batch_size, max_art_oovs)))
            extra_zeros = torch.zeros((batch_size, max_art_oovs))

        batch_graph = dgl.batch(batch_graph2)
        return {'tarpaper': batch_tarpaper, 'tarpaper_extend':batch_tarpaper_extend, 'extra_zeros':extra_zeros, \
             'raw_sent_input': batch_raw_sent_input, 'refpaper': batch_refpaper, 'refpaper_extend':batch_refpaper_extend,\
            'examples':batch_examples, 'tgt': batch_tgt, 'sent_num':sent_num, 'raw_tgt_text': batch_raw_tgt_text, \
            'article_oovs':batch_oovs, 'tgt_extend': batch_tgt_extend, 'article_len':batch_art_len,\
            'text':batch_text,'text_extend':batch_text_extend,'words': batch_words,'graph': batch_graph,\
            'ref_func': batch_reffunc,'tar_func':batch_tarfunc}

class ExampleSet_Classification(torch.utils.data.Dataset):
    def __init__(self, text_path, wordvocab, args, device=None):
        super(ExampleSet_Classification, self).__init__()
        self.device = device
        self.wordvocab = wordvocab
        self.args = args

        start = time.time()
        label_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '30', '31', '32', '33', '34', '35', '36', '37', '38', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']
        ###no 15,29,39
        self.label_to_id = {v: i for i, v in enumerate(label_list)}
        self.json_text_list, self.labels = readJson_cls(text_path, self.label_to_id)
        logger.info("[INFO] Finish reading %s. Total time is %f, Total size is %d", self.__class__.__name__,
                    time.time() - start, len(self.json_text_list))
        self.size = len(self.json_text_list) ###训练集的大小

    def get_example(self, index):
        json_text = self.json_text_list[index]
        sent = json_text['text']
        label = json_text['label']
        label = self.label_to_id[label]

        article_words = sent.lower().split()
        self.enc_sent = [self.wordvocab.word2id(w) for w in article_words]
        if len(self.enc_sent) > self.args.max_cls_length:
            self.enc_sent = self.enc_sent[:self.args.max_cls_length]
        self.enc_sent = torch.LongTensor(self.enc_sent)
        item = {'input':self.enc_sent, 'label':label}
        return item

    def __getitem__(self, index):
        item = self.get_example(index)
        return item

    def __len__(self):
        return self.size

    def sampler(self):
        train_sampler = None
        if self.args.m_per_class_sampler:
            train_sampler = MPerClassSampler(
                self.labels, 
                1,
                batch_size=self.args.cls_batch_size,
                length_before_new_iter=len(self.labels)
            )
        return train_sampler

    def batch_fn(self, samples):
        batch_input, batch_label =  [], []
        #batch_ex = map(list, zip(*samples))
        for ex_data in samples:
            if ex_data != {}:
                batch_input.append(ex_data['input'])
                batch_label.append(ex_data['label'])
        
        pad_id = self.wordvocab.word2id('<PAD>')
        bos_id = self.wordvocab.word2id('<BOS>')
        eos_id = self.wordvocab.word2id('<EOS>')
        batch_input= pad_sent_entity(batch_input, pad_id,bos_id,eos_id, flatten = False).transpose(0,1)
        batch_label = torch.LongTensor(batch_label)
        return {'input':batch_input, 'label':batch_label}
        
if __name__ == '__main__' :
    pass

