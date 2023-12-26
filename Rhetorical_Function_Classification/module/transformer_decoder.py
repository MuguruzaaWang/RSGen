"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import math
import numpy as np

from module.attention import MultiHeadedAttention, MultiAdjstHeadedAttention
from module.neural import PositionwiseFeedForward
from module.transformer_encoder import PositionalEncoding
from module.multi_head_only_attention  import MultiheadOnlyAttention
import pdb

MAX_SIZE = 5000

def get_generator(dec_hidden_size, vocab_size, emb_dim, device):
    gen_func = nn.LogSoftmax(dim=-1)
    ### nn.Sequential内部实现了forward函数
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, emb_dim),
        nn.LeakyReLU(),
        nn.Linear(emb_dim, vocab_size),
        gen_func
    )

class DecoderState(object):
    """Interface for grouping together the current state of a recurrent
    decoder. In the simplest case just represents the hidden state of
    the model.  But can also be used for implementing various forms of
    input_feeding and non-recurrent models.

    Modules need to implement this to utilize beam search decoding.
    """
    def detach(self):
        """ Need to document this """
        self.hidden = tuple([_.detach() for _ in self.hidden])
        self.input_feed = self.input_feed.detach()

    def beam_update(self, idx, positions, beam_size):
        """ Need to document this """
        for e in self._all:
            sizes = e.size()
            br = sizes[1]
            if len(sizes) == 3:
                sent_states = e.view(sizes[0], beam_size, br // beam_size,
                                     sizes[2])[:, :, idx]
            else:
                sent_states = e.view(sizes[0], beam_size,
                                     br // beam_size,
                                     sizes[2],
                                     sizes[3])[:, :, idx]

            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def map_batch_fn(self, fn):
        raise NotImplementedError()



class TransformerDecoderLayer(nn.Module):
    """
    Args:
      d_model (int): the dimension of keys/values/queries in
                       MultiHeadedAttention, also the input size of
                       the first-layer of the PositionwiseFeedForward.
      heads (int): the number of heads for MultiHeadedAttention.
      d_ff (int): the second-layer of the PositionwiseFeedForward.
      dropout (float): dropout probability(0-1.0).
      self_attn_type (string): type of self-attention scaled-dot, average
    """

    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerDecoderLayer, self).__init__()


        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        # self.paper_attn = MultiPaperAttention(
        #     heads, d_model, dropout=dropout)
        self.context_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)
        mask = self._get_attn_subsequent_mask(MAX_SIZE)
        # Register self.mask as a buffer in TransformerDecoderLayer, so
        # it gets TransformerDecoderLayer's cuda behavior automatically.
        self.register_buffer('mask', mask)
        #self.fusion_gate = nn.Sequential(nn.Linear(2 * d_model, 1), nn.Sigmoid())
        self.fusion = nn.Linear(2*d_model,d_model,bias = False)

    def forward(self, inputs, memory_bank, tar_ref_state, tar_ref_mask, src_pad_mask, \
                        tgt_pad_mask, wordsum, sent_states, layer_cache=None, step=None):
        """
        Args:
            inputs (`FloatTensor`): `[batch_size x 1 x model_dim]`
            memory_bank (`FloatTensor`): `[batch_size x src_len x model_dim]`
            src_pad_mask (`LongTensor`): `[batch_size x 1 x src_len]`
            tgt_pad_mask (`LongTensor`): `[batch_size x 1 x 1]`

        Returns:
            (`FloatTensor`, `FloatTensor`, `FloatTensor`):

            * output `[batch_size x 1 x model_dim]`
            * attn `[batch_size x 1 x src_len]`
            * all_input `[batch_size x current_step x model_dim]`

        """
        dec_mask = torch.gt(tgt_pad_mask +
                            self.mask[:, :tgt_pad_mask.size(1),
                                      :tgt_pad_mask.size(1)], 0)
        input_norm = self.layer_norm_1(inputs)
        all_input = input_norm

        query = self.self_attn(all_input, all_input, input_norm,
                                     mask=dec_mask,
                                     layer_cache=layer_cache,
                                     type="self")

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)

        # paper_context, paper_attn = self.paper_attn(tar_ref_state, tar_ref_state, 
        #                                 query_norm, mask = tar_ref_mask)
        # word_context = self.context_attn(memory_bank, memory_bank, 
        #                                 query_norm, mask = src_pad_mask, layer_cache=layer_cache,
        #                                 paper_attn=paper_attn, wordsum = wordsum, type="context")
        word_context = self.context_attn(memory_bank, memory_bank, 
                                        query_norm, mask = src_pad_mask, layer_cache=layer_cache,
                                    type="context")
        output = self.feed_forward(self.drop(word_context) + query + sent_states)

        return output, all_input
        # return output

    def _get_attn_subsequent_mask(self, size):
        """
        Get an attention mask to avoid using the subsequent info.

        Args:
            size: int

        Returns:
            (`LongTensor`):

            * subsequent_mask `[1 x size x size]`
        """
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        subsequent_mask = torch.from_numpy(subsequent_mask)
        return subsequent_mask


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, generator):
        super(TransformerDecoder, self).__init__()

        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])
        
        self.generator = generator

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.copy_or_generate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.copy_attention = MultiheadOnlyAttention(1, d_model, dropout=0)

    def forward(self, tgt, tarref_word, memory_bank, tar_ref_state, tar_ref_mask, wordsum, \
                    sent_outputs, tgt_starts, tgt_ends, state, memory_lengths=None, step=None, cache=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        tarref_word = state.src
        tgt_words = tgt.transpose(0,1)

        tgt_batch, tgt_len = tgt_words.size()

        # Run the forward pass of the TransformerDecoder.
        # emb = self.embeddings(tgt, step=step)

        padding_idx = self.embeddings.padding_idx
        if self.training:
            ends_mask = tgt_ends.gt(padding_idx)
            for i in range(tgt_batch):
                nozero_ends = tgt_ends[i][ends_mask[i]]
                tgt_words[i][nozero_ends[-1]] = padding_idx
        else:
            ends_mask = tgt_ends.gt(-1)

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim
        emb = emb.transpose(0, 1).contiguous()
        output = self.pos_emb(emb, step)

        src_memory_bank = memory_bank
        memory_dim = memory_bank.shape[2]

        tgt_pad_mask = tgt_words.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)
        tar_ref_mask = ~tar_ref_mask.unsqueeze(1).expand(tgt_batch, tgt_len, tar_ref_mask.size(1))

        src_pad_mask = tarref_word.data.eq(padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tarref_word.size(1))
        
        if self.training:
            sent_states = torch.zeros((tgt_batch,tgt_len,memory_dim),dtype=sent_outputs.dtype, device=sent_outputs.device)
            for index,(start, end, sent_output, end_mask) in enumerate(zip(tgt_starts,tgt_ends,sent_outputs,ends_mask)):
                start = start[end_mask]
                end = end[end_mask]
                for s,e,sent in zip(start,end,sent_output):
                    sent = sent.unsqueeze(0).expand(e+1-s,memory_dim)
                    sent_states[index][s:e+1] = sent
        else:
            sent_states = sent_outputs

        for i in range(self.num_layers):
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank, tar_ref_state, tar_ref_mask, 
                    src_pad_mask, tgt_pad_mask,wordsum,sent_states,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
        ###这里的输出output就是第六层transformer decoder的输出g(L2)
        output = self.layer_norm(output)

        # copy.shape = [4,211,3600]
        # src_memory_bank.shape = [4,50,72,256] ??? 这个是不是得调整一下

        #output.shape = [4,211,256] key.shape = [4,3600,256]  mask = [4,211,3600]
        
        copy = self.copy_attention(query=output,
                                          key=src_memory_bank,
                                          value=src_memory_bank,
                                          mask=src_pad_mask
                                          )
        copy = copy.transpose(0,1)
        '''
        copy_ent = self.copy_attention(query=output,
                                          key=ent_memory_bank,
                                          value=ent_memory_bank,
                                          mask=ent_pad_mask
                                          )
        copy_ent = copy_ent.transpose(0,1)
        '''
        #copy_or_generator.shape = [211,4,1]
        #output.shape = [4,211,256]
        copy_or_generate = self.copy_or_generate(output).transpose(0,1)
        # Process the result and update the attentions.
        outputs = output.transpose(0, 1).contiguous()

        ###假如我只是copy entity word呢，结果会如何？
        if self.training:
            return outputs[:-1], {'attn': copy[:-1], 'copy_or_generate': copy_or_generate[:-1], 'src':tarref_word, 'state':state}
        else:
            return outputs, {'attn': copy, 'copy_or_generate': copy_or_generate, 'src':tarref_word, 'state':state}
            
    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

    def get_normalized_probs(self, src_words, extra_zeros, outputs, copy_attn, copy_or_generate,log_probs=True):
        """Get normalized probabilities (or log probs) from a net's output."""
        if log_probs:
            generate = self.generator(outputs) * copy_or_generate
            if extra_zeros is not None:
                generate = torch.cat([generate, extra_zeros], 1)
            copy = copy_attn * (1 - copy_or_generate)
            final = generate.scatter_add(1, src_words, copy)
            final = torch.log(final+1e-15)
            return final
        else:
            generate = self.generator(outputs) * copy_or_generate
            copy = copy_attn * (1 - copy_or_generate)
            final = generate.scatter_add(1, src_words, copy)
            return final
            
            #return self.generator(outputs)

class TransformerDecoderState(DecoderState):
    """ Transformer Decoder state base class """

    def __init__(self, src):
        """
        Args:
            src (FloatTensor): a sequence of source words tensors
                    with optional feature tensors, of size (len x batch).
        """
        self.src = src
        self.previous_input = None
        self.previous_layer_inputs = None
        self.cache = None

    @property
    def _all(self):
        """
        Contains attributes that need to be updated in self.beam_update().
        """
        if (self.previous_input is not None
                and self.previous_layer_inputs is not None):
            return (self.previous_input,
                    self.previous_layer_inputs,
                    self.src)
        else:
            return (self.src,)

    def detach(self):
        if self.previous_input is not None:
            self.previous_input = self.previous_input.detach()
        if self.previous_layer_inputs is not None:
            self.previous_layer_inputs = self.previous_layer_inputs.detach()
        self.src = self.src.detach()

    def update_state(self, new_input, previous_layer_inputs):
        state = TransformerDecoderState(self.src)
        state.previous_input = new_input
        state.previous_layer_inputs = previous_layer_inputs
        return state

    def _init_cache(self, memory_bank, num_layers):
        self.cache = {}
        batch_size = memory_bank.size(1)
        depth = memory_bank.size(-1)

        for l in range(num_layers):
            layer_cache = {
                "memory_keys": None,
                "memory_values": None
            }
            layer_cache["self_keys"] = None
            layer_cache["self_values"] = None
            self.cache["layer_{}".format(l)] = layer_cache

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimension. """
        self.src = self.src.data.repeat(1, beam_size, 1)

    def map_batch_fn(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        self.src = fn(self.src, 0)
        if self.cache is not None:
            _recursive_map(self.cache)

class MultiPaperAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiPaperAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if(self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        key = shape(key)
        value = shape(value)

        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)


        drop_attn = self.dropout(attn)
        if(self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output,attn
        else:
            context = torch.matmul(drop_attn, value)
            return context,attn