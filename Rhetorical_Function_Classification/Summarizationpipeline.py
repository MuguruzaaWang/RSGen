#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

###这个版本的代码利用roberta的BPE编码方式来tokenize句子

import argparse
import datetime
import os
import shutil
import time
import numpy as np
import torch
import pdb
import signal
import glob

from Tester import SLTester

from module.embedding import Word_Embedding
import torch.nn.functional as F
from module.vocabulary import WordVocab
from module.ERTvocabulary import Vocab
import torch.nn as nn

import time
from tqdm import tqdm

#from module.utils import get_datasets
from module.opts import vocab_config, fill_config, get_args
from ertsumgraph_transformer import ERTSumGraph,build_optim_bert,build_optim_dec

from tools.logger import *
from tools import utils

from module.utlis_dataloader import *
from utils import distributed

from transformers import RobertaTokenizer, RobertaModel
from module.trainer_builder import build_trainer
from module.predictor_builder2 import build_predictor
from utils.logging import init_logger, logger
from module.optimizer import Optimizer
import re

_DEBUG_FLAG_ = False
global val_loss
val_loss = 2**31

model_flags = [ 'emb_size', 'enc_hidden_size', 'dec_hidden_size', 'enc_layers', 'dec_layers', 'block_size',  'heads', 'ff_size', 'hier',
               'inter_layers', 'inter_heads', 'block_size']

class Summarizationpipeline():
    def __init__(self,text_list,vocab_path):
        self.args = get_args()
        self.args.batch_size,self.args.seed,self.args.visible_gpus,self.args.dec_dropout,self.args.enc_dropout = 4,666,-1,0.1,0.1
        self.args.inter_layers = '6,7'
        self.args.inter_heads,self.args.doc_max_timesteps,self.args.alpha,self.args.no_repeat_ngram_size,self.args.prop  = 8,50,0.4,3,3
        self.args.inter_layers = [int(i) for i in self.args.inter_layers.split(',')]
        self.args.vocab_size = 50000
        self.args.use_bert = False

        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        logger.info("Pytorch %s", torch.__version__)

        VOCAL_FILE = vocab_path
        logger.info("[INFO] Create Vocab, vocab path is %s", VOCAL_FILE)
        wordvocab = WordVocab(VOCAL_FILE, self.args.vocab_size)

        word_padding_idx = wordvocab.word2id('<PAD>')
        
        sentences_list = []
        for text in text_list:
            sentences = re.split('(。|！|\!|\.|？|\?)',text) 
            new_sents = []
            for i in range(int(len(sentences)/2)):
                sent = sentences[2*i] + sentences[2*i+1]
                new_sents.append(sent)
            sentences_list.append(new_sents)
        temp_path = r'./temp_test.json'
        with open(temp_path,'w',encoding='utf-8') as f:
            for sents in sentences_list:
                dic = {'summary':'','text':sents}
                json.dump(dic,f)
                f.write('\n')

        test_text_file = temp_path
        
        self.args.wordvocab = wordvocab
        self.args.word_padding_idx = word_padding_idx
        device = "cpu"
        self.args.device =  device
        
        test_dataset = ExampleSet(test_text_file,wordvocab,self.args.sent_max_len, self.args.doc_max_timesteps, self.args.device)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.args.valid_batch_size, shuffle=False, num_workers=self.args.num_workers, \
                            pin_memory = True, collate_fn=test_dataset.batch_fn)
        self.args.test_dataloader = test_dataloader

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        

    def __call__(self,max_length,min_length,model_path):
        device = "cpu"
        device_id = -1
        self.args.max_length = max_length
        self.args.min_length = min_length
        self.args.test_from = model_path
        step = int(self.args.test_from.split('.')[-2].split('_')[-1])
        # validate(args, device_id, args.test_from, step)

        test_from = self.args.test_from
        logger.info('Loading checkpoint from %s' % test_from)
        checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])

        for k in opt.keys():
            if (k in model_flags):
                setattr(self.args, k, opt[k])
        print(self.args)

        # vocab = spm
        model = ERTSumGraph(self.args, self.args.word_padding_idx, self.args.vocab_size, device, checkpoint)
        model.to(device)

        model.eval()
        symbols = {'BOS':2,'EOS':3,'PAD':1,'UNK':0}
        predictor = build_predictor(self.args, self.args.wordvocab, symbols, model, device, logger=logger)
        candidates = predictor.translate(self.args.test_dataloader, step)

        return candidates






