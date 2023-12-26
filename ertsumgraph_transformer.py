import torch
from modules import BiLSTM, GraphTrans, GAT_Hetersum, MSA 
from torch import nn
from transformers import RobertaTokenizer, RobertaModel
from torch.nn.init import xavier_uniform_

import dgl
from module.embedding import Word_Embedding
from module.EMDecoder import TransformerDecoder as Hier_TransformerDecoder
from module.transformer_encoder import TgtTransformerEncoder
from module.transformer_decoder import TransformerDecoder as Sin_TransformerDecoder
from module.roberta import RobertaEmbedding
from module.EMNewEncoder import EMEncoder
from module.optimizer import Optimizer
from module.utlis_dataloader import find_sentence_boundaries
from module.transformer_sendecoder import Sen_TransformerDecoder
import pdb
import time

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][0]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_bert, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_bert,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optims'][1]
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr_dec, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps_dec,
            model_size=args.enc_hidden_size)

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('bert_model')]
    optim.set_parameters(params)


    return optim

def get_generator(dec_hidden_size, vocab_size, emb_dim, device):
    gen_func = nn.Softmax(dim=-1)
    ### nn.Sequential内部实现了forward函数
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, emb_dim),
        nn.LeakyReLU(),
        nn.Linear(emb_dim, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator

class Roberta_model(nn.Module):
    def __init__(self, roberta_path, finetune=False):
        super(Roberta_model, self).__init__()
        print('Roberta initialized')
        self.model = RobertaModel.from_pretrained(roberta_path)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
        self.pad_id = self.tokenizer.pad_token_id
        self._embedding = self.model.embeddings.word_embeddings

        self.finetune = finetune

    def forward(self, input_ids):
        attention_mask = (input_ids != self.pad_id).float()
        if(self.finetune):
            return self.model(input_ids, attention_mask=attention_mask)
        else:
            self.eval()
            with torch.no_grad():
                return self.model(input_ids, attention_mask=attention_mask)

class ERTSumGraph(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, symbols, checkpoint=None):
        super(ERTSumGraph, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.padding_idx = word_padding_idx
        self.device = device
        self.symbols = symbols
        # need to encode the following nodes:
        # word embedding : use glove embedding
        # sentence encoder: bilstm (word)
        # doc encoder: bilstm (sentence)
        # entity encoder: bilstm (word)
        # relation embedding: initial embedding
        # type embedding: initial embedding 

        # use roberta
        
        self.src_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=word_padding_idx)
        if self.args.share_embeddings:
            tgt_embeddings.weight = self.src_embeddings.weight
        self.encoder = EMEncoder(self.args, self.device, self.src_embeddings, self.padding_idx, None)
        emb_dim = tgt_embeddings.weight.size(1)
        
        self.tgt_encoder = TgtTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                      self.args.ff_size, self.args.enc_dropout, self.src_embeddings, self.device)

        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, emb_dim, self.device)
        if self.args.share_decoder_embeddings:
            self.generator[2].weight = tgt_embeddings.weight

        if args.hier_decoder:
            self.decoder = Hier_TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings)
        
        else:
            self.decoder = Sin_TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, generator=self.generator)

        self.dropout = torch.nn.Dropout(self.args.enc_dropout)

        #self.tw_embedding = nn.Parameter(embeddings)
        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            print('keys为:'+str(keys))
            for k in keys:
                if ('a_2' in k):
                    checkpoint['model'][k.replace('a_2', 'weight')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])
                if ('b_2' in k):
                    checkpoint['model'][k.replace('b_2', 'bias')] = checkpoint['model'][k]
                    del (checkpoint['model'][k])

            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for n, p in self.named_parameters():
                if 'RobertaModel' not in n:
                    if p.dim() > 1:
                        xavier_uniform_(p)


        self.to(device)

    def forward(self,  batch):
        
        tgt = batch['tgt']
        
        tar_ref_context, tarref_word, tar_ref_state, tar_ref_mask, wordsum = self.encoder(batch)

        dec_state = self.decoder.init_decoder_state(tarref_word, tar_ref_context)     # src: num_paras_in_one_batch x max_length
        decoder_outputs = self.decoder(tgt,tarref_word, tar_ref_context, tar_ref_state, tar_ref_mask, wordsum, dec_state)
        
        return decoder_outputs

