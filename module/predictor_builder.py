#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math

import torch
import pdb

from itertools import count

from tensorboardX import SummaryWriter
import torch.nn.functional as F

from module.beam import GNMTGlobalScorer
from module.cal_rouge import test_rouge, rouge_results_to_str
from module.neural import tile
from module.utlis_dataloader import load_to_cuda
from transformers import top_k_top_p_filtering
from module import data
import copy
import re

def build_predictor(args, wordvocab, symbols, model, device, logger=None):
    scorer = GNMTGlobalScorer(args.alpha,length_penalty='wu')
    translator = Translator(args, model, wordvocab, symbols, device, global_scorer=scorer, logger=logger)
    return translator

def _bottle(_v):
    return _v.view(-1, _v.size(2))

class Translator(object):

    def __init__(self,
                 args,
                 model,
                 vocab,
                 symbols,
                 device,
                 n_best=1,
                 global_scorer=None,
                 logger=None,
                 dump_beam=""):
        self.logger = logger
        self.cuda = args.visible_gpus != '-1'
        self.args = args

        self.model = model
        self.decoder = self.model.decoder
        self.vocab = vocab
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']
        self.unk_token = symbols['UNK']
        self.device  = device


        self.n_best = n_best
        self.max_length = args.max_length
        self.global_scorer = global_scorer
        self.beam_size = args.beam_size
        self.min_length = args.min_length
        self.dump_beam = dump_beam

        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        tensorboard_log_dir = self.args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    def _build_target_tokens(self, pred, article_oovs):
        # vocab = self.fields["tgt"].vocab
        tokens = []
        for tok in pred:
            tok = int(tok)
            tokens.append(tok)
            if tokens[-1] == self.end_token:
                tokens = tokens[:-1]
                break
        if self.args.use_bert:
            tokens = [t for t in tokens if t<self.vocab.size()]
            #tokens = self.vocab.decode(tokens).split(' ')
            tokens = [self.vocab.id2word(t) for t in tokens]

        else:
            tokens = data.outputids2words(tokens, self.vocab, article_oovs)
        return tokens

    def from_batch(self, translation_batch, article_oovs):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = len(batch['text'])
        preds, pred_score, gold_score, tgt_str, src = list(zip(*list(zip(translation_batch["predictions"],
                                                                         translation_batch["scores"],
                                                                         translation_batch["gold_score"],
                                                                         batch['raw_tgt_text'], batch['text']))))

        translations = []
        for b in range(batch_size):
            pred_sents = sum([self._build_target_tokens(preds[b][n], article_oovs[b])
                for n in range(self.n_best)],[])
            #pdb.set_trace()
            gold_sent = tgt_str[b].split()
                #raw_src = self.vocab.decode(list([int(w) for w in src[b]]))
            
            y = src[b].reshape(-1)
            raw_src = ' '.join([self.vocab.id2word(t) for t in list([int(w) for w in y])])
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations


    def translate(self,
                  data_iter,step):

        self.model.eval()
        gold_path = self.args.result_path + '.%d.gold'%step
        can_path = self.args.result_path + '.%d.candidate'%step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        # self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        # self.can_out_file = codecs.open(can_path, 'w', 'utf-8')
        # # self.raw_gold_out_file = codecs.open(raw_gold_path, 'w', 'utf-8')
        # self.raw_can_out_file = codecs.open(raw_can_path, 'w', 'utf-8')

        raw_src_path = self.args.result_path + '.%d.raw_src' % step
        #self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        ct = 0
        
        with torch.no_grad():
            for batch in data_iter:
                batch = load_to_cuda(batch, self.device)
                article_oovs = batch['article_oovs']
                with torch.no_grad():
                    batch_data = self._fast_translate_batch(
                        batch,
                        self.max_length,
                        min_length=self.min_length,
                        n_best=self.n_best)

                translations = self.from_batch(batch_data, article_oovs)

                for trans in translations:
                    pred, gold, src = trans
                    pred_str = ' '.join(pred).replace('<Q>', ' ').replace(r' +', ' ').replace('<unk>', 'UNK').strip()
                    pred_str = pred_str.replace('@ cite', '@cite').replace('@ math', '@math').replace('<BOD>','')
                    pred_str = re.sub(r'<type(\d+)>','',pred_str)
                    gold_str = ' '.join(gold).replace('<t>', '').replace('</t>', '').replace('<Q>', ' ').replace(r' +',
                                                                                                                 ' ').strip()
                    gold_str = gold_str.lower()
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    #self.src_out_file.write(src.strip() + '\n')
                    
                # self.raw_can_out_file.flush()
                # self.raw_gold_out_file.flush()
                self.can_out_file.flush()
                self.gold_out_file.flush()
                #self.src_out_file.flush()

        #pdb.set_trace()
        # self.raw_can_out_file.close()
        # self.raw_gold_out_file.close()
        self.can_out_file.close()
        self.gold_out_file.close()
        #self.src_out_file.close()

        if(step!=-1 and self.args.report_rouge):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger.info('Rouges at step %d \n%s'%(step,rouge_results_to_str(rouges)))
            if self.tensorboard_writer is not None:
                self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
                self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)


    def _report_rouge(self, gold_path, can_path):
        self.logger.info("Calculating Rouge")
        candidates = codecs.open(can_path, encoding="utf-8")
        references = codecs.open(gold_path, encoding="utf-8")
        results_dict = test_rouge(candidates, references, 1)
        return results_dict

    def translate_batch(self, batch,  fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                batch,
                self.max_length,
                min_length=self.min_length,
                n_best=self.n_best)

    def _fast_translate_batch(self,
                              batch,
                              max_length,
                              min_length=0,
                              n_best=1):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam

        beam_size = self.beam_size
        tar_ref_context, tarref_word, tar_ref_state, tar_ref_mask, wordsum0 = self.model.encoder(batch)

        dec_states = self.model.decoder.init_decoder_state(tarref_word, tar_ref_context, with_cache=True)

        device = tar_ref_context.device
        batch_size = tar_ref_context.size(0)

        #use beam search
        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        tar_ref_context = tile(tar_ref_context, beam_size, dim=0)
        tarref_word = tile(tarref_word, beam_size, dim=0)
        tar_ref_state = tile(tar_ref_state, beam_size, dim=0)
        tar_ref_mask = tile(tar_ref_mask, beam_size, dim=0)
        wordsum = sum([8*[x] for x in wordsum0],[])
        
        extra_zeros = batch['extra_zeros']
        
        if extra_zeros!=None:
            extra_zeros = tile(extra_zeros, beam_size, dim=0)
        
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        
        ###第一步填入bos_token
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)


        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                        device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch
        #use beam search 
        for step in range(max_length):
            decoder_input = alive_seq[:, -1].view(1, -1)

            #pdb.set_trace()
            # if (self.args.hier):
            #     dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
            #                                              memory_masks=mask_hier,
            #                                              step=step)
            # else:
            #     dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
            #                                              step=step)
            new_tensor = torch.zeros([decoder_input.shape[0],decoder_input.shape[1]],device=decoder_input.device,dtype=torch.long)
            new_tensor.fill_(self.vocab.word2id(self.unk_token))
            decoder_input = torch.where(decoder_input>=self.vocab.size(),new_tensor,decoder_input)
            dec_out, cache_dict = self.model.decoder(decoder_input,tarref_word, tar_ref_context, \
                          tar_ref_state, tar_ref_mask, wordsum, dec_states,step = step)
            # Generator forward.

            dec_states = cache_dict['state']

            copy_attn = cache_dict['attn']
            copy_or_generate = cache_dict['copy_or_generate']
            src_words = cache_dict['src']
            
            bottled_output = _bottle(dec_out)
            bottled_copyattn = _bottle(copy_attn.contiguous())
            bottled_cog = _bottle(copy_or_generate.contiguous())
            batch_size, src_len = src_words.size()
            split_size = dec_out.shape[0]
            src_words = src_words.unsqueeze(0).expand(split_size, batch_size ,src_len).contiguous()
            bottled_src = _bottle(src_words)

            if extra_zeros is not None:
                _, extra_len = extra_zeros.size()
                extra_zeros2 = extra_zeros.unsqueeze(0).expand(split_size, batch_size ,extra_len).contiguous()
                bottled_extra_zeros = _bottle(extra_zeros2)
            else:
                bottled_extra_zeros = None
            log_probs = self.decoder.get_normalized_probs(bottled_src, bottled_extra_zeros, bottled_output, bottled_copyattn, bottled_cog)

            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20
                
            log_probs[:, 0] = -1e20
            ### ngram blocking
            alive_size = alive_seq.shape[0]
            if self.args.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(alive_size)]
                for bbsz_idx in range(alive_size):
                    gen_tokens = alive_seq[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.args.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                                gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]
            
            if self.args.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(alive_seq[bbsz_idx, step + 2 - self.args.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.args.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(alive_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(alive_size)]

                for bbsz_idx in range(alive_size):
                    log_probs[bbsz_idx, banned_tokens[bbsz_idx]] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty
            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.floor_divide(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1).long()
            #pdb.set_trace()
            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        for n, (score, pred) in enumerate(best_hyp):
                            if n >= n_best:
                                break
                            results["scores"][b].append(score)
                            results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))

            wordsum_temp = copy.deepcopy(wordsum)
            wordsum = []
            select_indices2 = select_indices.tolist()
            for indice in select_indices2:
                wordsum.append(wordsum_temp[indice])
            # Reorder states.
            select_indices = batch_index.view(-1).long()
            tar_ref_context = tar_ref_context.index_select(0, select_indices)
            tarref_word = tarref_word.index_select(0, select_indices)
            tar_ref_state = tar_ref_state.index_select(0, select_indices)
            tar_ref_mask = tar_ref_mask.index_select(0, select_indices)
            if extra_zeros!=None:
                extra_zeros = extra_zeros.index_select(0, select_indices)
            # mask_hier = mask_hier.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

