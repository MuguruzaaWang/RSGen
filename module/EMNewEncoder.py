from module.transformer_encoder import NewTransformerEncoder, BertLSTMEncoder
from module.neural import PositionwiseFeedForward, sequence_mask
from module.roberta import RobertaEmbedding
from modules import GraphTrans,GAT
from module.GAT import WSWGAT

import torch.nn as nn
import pdb
import dgl
import time

from module.utlis_dataloader import *
func_types = {'background':1,'objective':2,'method':3,'result':4,'other':5}

class EMEncoder(nn.Module):
    def __init__(self, args, device, src_embeddings, padding_idx, bert_model):
        super(EMEncoder, self).__init__()
        self.args = args
        self.padding_idx = padding_idx
        self.device = device
        self.src_embeddings = src_embeddings

        self.sent_encoder = NewTransformerEncoder(self.args.enc_layers, self.args.enc_hidden_size, self.args.heads,
                                                  self.args.ff_size, self.args.enc_dropout, src_embeddings, self.device)

        self.graph_enc = GraphTrans(args)

        self.layer_norm = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        # self.layer_norm2 = nn.LayerNorm(self.args.emb_size, eps=1e-6)
        self.feed_forward = PositionwiseFeedForward(self.args.enc_hidden_size, self.args.ff_size, self.args.enc_dropout)

        if args.graph_enc == "gat":
            # we only support gtrans, don't use this one
            ###有4个头，每个头的输出就是输入维度除以4
            self.gat = nn.ModuleList([GAT(args.enc_hidden_size, args.enc_hidden_size//args.n_head, args.n_head, attn_drop=args.attn_drop, trans=False) for _ in range(args.prop)]) #untested
        else:
            self.gat = nn.ModuleList([GAT(args.nhid, args.nhid//args.n_head, args.n_head, attn_drop=args.attn_drop, ffn_drop=args.drop, trans=True) for _ in range(args.prop)])
        self.prop = args.prop

    def forward(self, batch):
        tarpaper = batch['tarpaper']
        refpaper = batch['refpaper']
        tar_func = batch['tar_func']
        ref_func = batch['ref_func']
        graph = batch['graph'].to(self.device)

        batch_size, tar_n_sents, n_tokens = tarpaper.size()
        batch_size, ref_n_docs, ref_n_sents = ref_func.size()
        tarsent_state, tar_context, _ = self.sent_encoder(tarpaper)
        refsent_state, ref_context, _ = self.sent_encoder(refpaper)
        
        ### next need to obtain augmentation function representations os each paper.
        tar_aug_state = torch.zeros((batch_size,5,self.args.enc_hidden_size),dtype=tarsent_state.dtype, device=self.device)
        tar_aug_mask = torch.zeros((batch_size,5),dtype=torch.bool, device=self.device)
        for label in range(5):
            label_mask = tar_func.eq(label+1)
            if label_mask.sum().tolist() == 0:
                continue
            label_num_sum = torch.sum(label_mask,dim=1,keepdim=True)
            tar_aug_mask[:,label] = label_num_sum.ne(0).squeeze(-1)
            temp = label_num_sum.cpu().numpy()
            temp[temp==0] = 10e10
            label_num_sum  = torch.from_numpy(temp).to(self.device)
            label_mask = label_mask.unsqueeze(-1).expand_as(tarsent_state)      
            mask_label_sum = torch.sum(tarsent_state*label_mask, dim=1) 
            aug_rep = torch.div(mask_label_sum,label_num_sum)
            tar_aug_state[:,label,:] = aug_rep
        
        refpaper = batch['refpaper_extend']
        refsent_state, ref_context, ref_mask,refpaper = pad_ref_state(refsent_state,ref_context,ref_func,refpaper)
        ref_token_mask =  ~(refpaper.data.eq(self.padding_idx).bool())
        ref_aug_state = torch.zeros((batch_size,ref_n_docs,5,self.args.enc_hidden_size),dtype=tarsent_state.dtype, device=self.device)
        ref_aug_mask = torch.zeros((batch_size,ref_n_docs,5),dtype=torch.bool, device=self.device)
        for label in range(5):
            label_mask = ref_func.eq(label+1)
            if label_mask.sum().tolist() == 0:
                continue
            label_num_sum =torch.sum(label_mask,dim=2,keepdim=True)
            ref_aug_mask[:,:,label] = label_num_sum.ne(0).squeeze(-1)
            temp = label_num_sum.cpu().numpy()
            temp[temp==0] = 10e10
            label_num_sum  = torch.from_numpy(temp).to(self.device)
            label_mask = label_mask.unsqueeze(-1).expand_as(refsent_state)      
            mask_label_sum = torch.sum(refsent_state*label_mask, dim=2) 
            aug_rep = torch.div(mask_label_sum,label_num_sum)
            ref_aug_state[:,:,label,:] = aug_rep

        word_state = self.src_embeddings(batch['words'])
        word_mask = batch['words'].ne(self.padding_idx)
        #word_state = word_state[word_mask]
        # tar_aug_state = tar_aug_state[tar_aug_mask]
        # ref_aug_state = ref_aug_state[ref_aug_mask]
        
        tar_mask = tar_func.ne(0)
        tar_num_sum = torch.sum(tar_mask,dim=1,keepdim=True)
        tar_mask2 = tar_num_sum.ne(0)
        temp = tar_num_sum.cpu().numpy()
        temp[temp==0] = 10e10
        tar_num_sum  = torch.from_numpy(temp).to(self.device)
        tar_mask = tar_mask.unsqueeze(-1).expand_as(tarsent_state)      
        mask_tar_sum = torch.sum(tarsent_state*tar_mask, dim=1) 
        tarpaper_state = torch.div(mask_tar_sum,tar_num_sum)

        ref_mask = ref_func.ne(0)
        ref_num_sum = torch.sum(ref_mask,dim=2,keepdim=True)
        ref_mask2 = ref_num_sum.ne(0).squeeze(-1)
        temp = ref_num_sum.cpu().numpy()
        temp[temp==0] = 10e10
        ref_num_sum  = torch.from_numpy(temp).to(self.device)
        ref_mask = ref_mask.unsqueeze(-1).expand_as(refsent_state)      
        mask_ref_sum = torch.sum(refsent_state*ref_mask, dim=2) 
        refpaper_state = torch.div(mask_ref_sum,ref_num_sum)
        paper_state = torch.cat((tarpaper_state.unsqueeze(1),refpaper_state),dim=1)
        paper_mask = torch.cat((tar_mask2,ref_mask2),dim=1)

        #paper_state = paper_state[paper_mask]
        #refpaper_state = refpaper_state[ref_mask2]

        pnode_id = graph.filter_nodes(lambda nodes: (nodes.data["type"] == 2) | (nodes.data["type"] == 3))
        rnode_id = graph.filter_nodes(lambda nodes: (nodes.data["type"] == 2))
        tnode_id = graph.filter_nodes(lambda nodes: (nodes.data["type"] == 3))

        tar_mask2 = torch.full(pnode_id.shape, False, dtype=torch.bool, device=self.device)
        for _id in tnode_id:
            tar_mask2 += pnode_id.eq(_id)

        ###目前得到了word_state，tar_aug_state，ref_aug_state，tarpaper_state，refpaper_stateg
        ###下一步开始进行图更新 

        tar_aug_state = tar_aug_state.unsqueeze(1)
        aug_state = torch.cat([tar_aug_state,ref_aug_state],dim=1)
        tar_aug_mask = tar_aug_mask.unsqueeze(1)
        aug_mask = torch.cat([tar_aug_mask,ref_aug_mask],dim=1)
        #aug_state = aug_state[aug_mask]

        init_h = []

        zero_tensor = torch.zeros((1,self.args.enc_hidden_size),dtype=tarsent_state.dtype, device=self.device)
        for i in range(batch_size):
            init_h.append(word_state[i][word_mask[i]])
            init_h.append(aug_state[i][aug_mask[i]])
            init_h.append(paper_state[i][paper_mask[i]])
            init_h.append(zero_tensor)
        
        init_h = torch.cat(init_h, 0)
        feats = init_h
        
        for p in range(self.prop):
            feats = self.gat[p](graph, feats)
        
        paper_len = paper_mask.sum(dim=-1).tolist()
        g_paper = feats.index_select(0, graph.filter_nodes(lambda x: (x.data['type']==NODE_TYPE['reference']) \
                 | (x.data['type']==NODE_TYPE['target']))).split(paper_len)
        tarpaper_state = []
        for i in range(batch_size):
            tarpaper_state.append(g_paper[i][0:1])
        tarpaper_state= torch.cat(tarpaper_state, 0)
        refpaper_state = []
        for i in range(batch_size):
            refpaper_state.append(g_paper[i][1:])
        refpaper_state= torch.cat(refpaper_state, 0)

        _target_state = tarpaper_state.unsqueeze(1).unsqueeze(1)
        tar_context = self.feed_forward(tar_context + _target_state)         # batch_size x n_paras x n_tokens x hidden
        tar_mask =  ~(tarpaper.data.eq(self.padding_idx).bool())

        _ref_state = torch.zeros((batch_size*ref_n_docs,self.args.enc_hidden_size),dtype=tarsent_state.dtype, device=self.device)
        ref_mask2 = ref_mask2.reshape(-1)
        _ref_state[ref_mask2] = refpaper_state
        _ref_state = _ref_state.reshape(batch_size,ref_n_docs,-1)
        _ref_state2 = _ref_state.unsqueeze(2).unsqueeze(2)
        ref_context = self.feed_forward(ref_context + _ref_state2)

        tar_ref_state = torch.cat([tarpaper_state.unsqueeze(1),_ref_state],dim=1)
        tar_ref_mask = paper_mask

        ref_mask =  ~(refpaper.data.eq(self.padding_idx).bool())
        tarpaper = batch['tarpaper_extend']
        tar_ref_context,tarref_word, wordsum = unpad_ref_state(tar_context,ref_context,tar_func, ref_func, tarpaper, refpaper)
        return tar_ref_context, tarref_word, tar_ref_state, tar_ref_mask, wordsum


        # wordaug_state = []
        # for i in range(batch_size):
        #     wordaug_state.append(word_state[i][word_mask[i]])
        #     wordaug_state.append(aug_state[i][aug_mask[i]])
        # wordaug_state = torch.cat(wordaug_state, 0)
        # waedge_id = graph.filter_edges(lambda edges: (edges.src["type"] == 0) & (edges.dst["type"] == 1)).cpu()
        # g_wa = dgl.edge_subgraph(graph.cpu(),waedge_id).to(self.device)
        # feats = wordaug_state

        # word_len = word_mask.sum(dim=-1).tolist()
        # aug_split_len = aug_mask.sum(dim=-1).sum(dim=-1).tolist()
        # paper_len = paper_mask.sum(dim=-1).tolist()

        # apedge_id = graph.filter_edges(lambda edges: (edges.src["type"] == 1) & ((edges.dst["type"] == 2) | (edges.dst["type"] == 3))).cpu()
        # g_ap = dgl.edge_subgraph(graph.cpu(),apedge_id).to(self.device)

        # rtedge_id = graph.filter_edges(lambda edges: (edges.src["type"] == 2) & (edges.dst["type"] == 3)).cpu()
        # g_rt = dgl.edge_subgraph(graph.cpu(),rtedge_id).to(self.device)

        # tredge_id = graph.filter_edges(lambda edges: (edges.src["type"] == 3) & (edges.dst["type"] == 2)).cpu()
        # g_tr = dgl.edge_subgraph(graph.cpu(),tredge_id).to(self.device)

        # paedge_id = graph.filter_edges(lambda edges: ((edges.src["type"] == 2) | (edges.src["type"] == 3)) & (edges.dst["type"] == 1)).cpu()
        # g_pa = dgl.edge_subgraph(graph.cpu(),paedge_id).to(self.device)

        # awedge_id = graph.filter_edges(lambda edges: (edges.src["type"] == 1) & (edges.dst["type"] == 0)).cpu()
        
        # g_aw = dgl.edge_subgraph(graph.cpu(),awedge_id).to(self.device)

        # pdb.set_trace()
        # for p in range(self.prop):
        #     wordaug_state = self.gat[p](g_wa, wordaug_state)
        #     g_word = wordaug_state.index_select(0, g_wa.filter_nodes(lambda x: x.data['type']==NODE_TYPE['word'])).split(word_len)
        #     g_aug =  wordaug_state.index_select(0, g_wa.filter_nodes(lambda x: x.data['type']==NODE_TYPE['augment_func'])).split(aug_split_len)
        #     augpaper_state = []
        #     for i in range(batch_size):
        #         augpaper_state.append(g_aug[i])
        #         if p == 0:
        #             augpaper_state.append(paper_state[i][paper_mask[i]])
        #         else:
        #             augpaper_state.append(g_paper[i])
        #     augpaper_state = torch.cat(augpaper_state, 0)
        #     augpaper_state = self.gat[p](g_ap, augpaper_state)

        #     g_aug =  augpaper_state.index_select(0, g_ap.filter_nodes(lambda x: x.data['type']==NODE_TYPE['augment_func'])).split(aug_split_len)
        #     g_paper = augpaper_state.index_select(0, g_ap.filter_nodes(lambda x: (x.data['type']==NODE_TYPE['reference']) \
        #         | (x.data['type']==NODE_TYPE['target']))).split(paper_len)
            
        #     paper_state = []
        #     for i in range(batch_size):
        #         paper_state.append(g_paper[i][0:1])
        #         paper_state.append(g_paper[i][1:])
        #     paper_state = torch.cat(paper_state, 0)
        #     paper_state = self.gat[p](g_rt, paper_state)

        #     paper_state = paper_state.split(paper_len)
        #     paper_state2 = []
        #     for i in range(batch_size):
        #         paper_state2.append(paper_state[i][0:1])
        #         paper_state2.append(paper_state[i][1:])
        #     paper_state2= torch.cat(paper_state2, 0)
        #     paper_state2 = self.gat[p](g_tr, paper_state2)

        #     g_paper = paper_state2.split(paper_len)
        #     paperaug_state = []
        #     for i in range(batch_size):
        #         paperaug_state.append(g_aug[i])
        #         paperaug_state.append(g_paper[i])
        #     paperaug_state = torch.cat(paperaug_state, 0)
        #     paperaug_state = self.gat[p](g_pa, paperaug_state)

        #     g_aug =  paperaug_state.index_select(0, g_pa.filter_nodes(lambda x: x.data['type']==NODE_TYPE['augment_func'])).split(aug_split_len)
        #     augword_state = []
        #     for i in range(batch_size):
        #         augword_state.append(g_word[i])
        #         augword_state.append(g_aug[i])
        #     augword_state = torch.cat(augword_state, 0)
        #     augword_state = self.gat[p](g_aw, augword_state)

        #     ###开始第二个循环
        #     wordaug_state = augword_state
        # tarpaper_state = []
        # for i in range(batch_size):
        #     tarpaper_state.append(g_paper[i][0:1])
        # tarpaper_state= torch.cat(tarpaper_state, 0)
        # refpaper_state = []
        # for i in range(batch_size):
        #     refpaper_state.append(g_paper[i][1:])
        # refpaper_state= torch.cat(refpaper_state, 0)

        # _target_state = tarpaper_state.unsqueeze(1).unsqueeze(1)
        # tar_context = self.feed_forward(tar_context + _target_state)         # batch_size x n_paras x n_tokens x hidden
        # tar_mask =  ~(tarpaper.data.eq(self.padding_idx).bool())

        # _ref_state = torch.zeros((batch_size*ref_n_docs,self.args.enc_hidden_size),dtype=tarsent_state.dtype, device=self.device)
        # ref_mask2 = ref_mask2.reshape(-1)
        # _ref_state[ref_mask2] = refpaper_state
        # _ref_state = _ref_state.reshape(batch_size,ref_n_docs,-1)
        # _ref_state2 = _ref_state.unsqueeze(2).unsqueeze(2)
        # ref_context = self.feed_forward(ref_context + _ref_state2)

        # tar_ref_state = torch.cat([tarpaper_state.unsqueeze(1),_ref_state],dim=1)
        # tar_ref_mask = paper_mask

        # ref_mask =  ~(refpaper.data.eq(self.padding_idx).bool())
        # tarpaper = batch['tarpaper_extend']
        # tar_ref_context,tarref_word, wordsum = unpad_ref_state(tar_context,ref_context,tar_func, ref_func, tarpaper, refpaper)
        # return tar_ref_context, tarref_word, tar_ref_state, tar_ref_mask, wordsum