"""
Implementation of "Attention is All You Need"
"""

import torch
import torch.nn as nn
import numpy as np
import math

from module.attention import MultiHeadedAttention
from module.neural import PositionwiseFeedForward
from module.neural import SenPositionalEncoding
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

    def forward(self, inputs, memory_bank, src_pad_mask, tgt_pad_mask, layer_cache=None, step=None):
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
        word_context = self.context_attn(memory_bank, memory_bank, query_norm,
                                      mask=src_pad_mask,
                                      layer_cache=layer_cache,
                                      type="context")

        output = self.feed_forward(self.drop(word_context) + query)

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


class Sen_TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings, generator, symbols):
        super(Sen_TransformerDecoder, self).__init__()

        self.decoder_type = 'transformer'
        self.num_layers = num_layers
        self.embeddings = embeddings
        self.pos_emb = SenPositionalEncoding(dropout,self.embeddings.embedding_dim)
        self.transformer_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_layers)])

        self.generator = generator
        self.symbols = symbols

        # TransformerDecoder has its own attention mechanism.
        # Set up a separated copy attention layer, if needed.
        self._copy = False
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

        self.copy_or_generate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        self.copy_attention = MultiheadOnlyAttention(1, d_model, dropout=0)

    def forward(self, tgt, tgt_sent, memory_bank, state, memory_lengths=None,
                step=None, cache=None,memory_masks=None):
        """
        See :obj:`onmt.modules.RNNDecoderBase.forward()`
        """
        src = state.src
        src_words = src.transpose(0, 1)
        src_batch, src_len = src_words.size()

        if self.training:
            BOD = torch.full([src_batch],self.symbols['BOD'],dtype=src.dtype,device=src.device)
            emb_BOD = self.embeddings(BOD) * math.sqrt(self.embeddings.embedding_dim)
            emb_BOD = emb_BOD.unsqueeze(1)
            output = torch.cat([emb_BOD, tgt_sent[:,:-1,:]], dim=1)

            #tgt_temp = torch.cat((BOD.unsqueeze(1),tgt),dim=-1)
            output = self.pos_emb(output, tgt, step)
            #output = self.pos_emb(output, step)
        else:
            output = tgt_sent
            output = self.pos_emb(output, tgt, step)

        tgt_batch, tgt_len = tgt.size()

        padding_idx = self.embeddings.padding_idx

        tgt_pad_mask = torch.eq(tgt, padding_idx).unsqueeze(1) \
            .expand(tgt_batch, tgt_len, tgt_len)

        # first_hidden_state = self.first_sent_emb.repeat(src_batch, 1, 1)
        # shifted_sum_state = torch.cat([first_hidden_state, tgt_state], dim=1)

        src_memory_bank = memory_bank
        
        #ent_memory_bank = ent_context.view(src_batch,-1,memory_dim)
        #tgt_len = 211
        if (not memory_masks is None):
            src_len = memory_masks.size(-1)
            src_pad_mask = memory_masks.expand(src_batch, tgt_len, src_len)

        else:
            src_pad_mask = src_words.data.eq(padding_idx).unsqueeze(1) \
                .expand(src_batch, tgt_len, src_len)
        
        for i in range(self.num_layers):
            output, all_input \
                = self.transformer_layers[i](
                    output, src_memory_bank,
                    src_pad_mask, tgt_pad_mask,
                    layer_cache=state.cache["layer_{}".format(i)]
                    if state.cache is not None else None,
                    step=step)
        
        ###这里的输出output就是第六层transformer decoder的输出g(L2)
        output = self.layer_norm(output)
        #outputs = output.transpose(0, 1).contiguous()
        return output, state

    def init_decoder_state(self, src, memory_bank,
                           with_cache=False):
        """ Init decoder state """
        if(src.dim()==3):
            src = src.view(src.size(0),-1).transpose(0,1)
        else:
            src = src.transpose(0,1)
        state = TransformerDecoderState(src)
        if with_cache:
            state._init_cache(memory_bank, self.num_layers)
        return state

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

        self.src = fn(self.src, 1)
        if self.cache is not None:
            _recursive_map(self.cache)