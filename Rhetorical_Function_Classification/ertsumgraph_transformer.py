import torch
from modules import BiLSTM, GraphTrans, GAT_Hetersum, MSA 
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.nn.init import xavier_uniform_

import dgl
from module.embedding import Word_Embedding
from module.transformer_decoder import TransformerDecoder
from module.EMNewEncoder import EMEncoder
from module.optimizer import Optimizer
import copy
import pdb

def build_optim_bert(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][0]
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

    params = [(n, p) for n, p in list(model.named_parameters()) if n.startswith('scibert.model')]
    optim.set_parameters(params)


    return optim

def build_optim_dec(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'][1]
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

    params = [(n, p) for n, p in list(model.named_parameters()) if not n.startswith('scibert.model')]
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

class Scibert(nn.Module):
    def __init__(self, path, finetune=True):
        super(Scibert, self).__init__()
        self.model = AutoModel.from_pretrained(path)
        self.finetune = finetune

    def forward(self, x, segs, mask):
        if (self.finetune):
            top_vec = self.model(x, attention_mask=mask,  token_type_ids=segs)
        else:
            self.eval()
            with torch.no_grad():
                top_vec  = self.model(x, attention_mask=mask,  token_type_ids=segs)
        return top_vec

class ERTSumGraph(nn.Module):
    def __init__(self, args, padding_idx, vocab_size, device, checkpoint=None):
        super(ERTSumGraph, self).__init__()
        self.args = args
        self.device = device
        self.padding_idx = padding_idx
        self.dropout = nn.Dropout(0.1)
        self.vocab_size = vocab_size
        
        self.scibert = Scibert(r'/data/run01/scv7414/wpc/rhetorical_aspect_embeddings_myexp/scibert-base', finetune=True)

        self.classifier = nn.Linear(768, 23)

        tgt_embeddings = torch.nn.Embedding(self.vocab_size, self.args.emb_size, padding_idx=padding_idx)
        self.generator = get_generator(self.args.dec_hidden_size, self.vocab_size, 256, self.device)
        self.decoder = TransformerDecoder(
                self.args.dec_layers,
                self.args.dec_hidden_size, heads=self.args.heads,
                d_ff=self.args.ff_size, dropout=self.args.dec_dropout, embeddings=tgt_embeddings, generator=self.generator)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            pass
            # for module in self.decoder.modules():
            #     if isinstance(module, (nn.Linear, nn.Embedding)):
            #         module.weight.data.normal_(mean=0.0, std=0.02)
            #     elif isinstance(module, nn.LayerNorm):
            #         module.bias.data.zero_()
            #         module.weight.data.fill_(1.0)
            #     if isinstance(module, nn.Linear) and module.bias is not None:
            #         module.bias.data.zero_()

        self.to(device)

    def forward(self,  batch):
        sentences = batch['sentences']
        segs = batch['segs']
        mask_sentences = ~sentences.eq(self.padding_idx)

        src_vec = self.scibert(sentences, segs, mask_sentences)

        cls_mask = sentences.eq(102)
        sen_emb = src_vec[0][cls_mask]
        sen_emb = self.dropout(sen_emb)

        label_logits = self.classifier(sen_emb)

        return label_logits