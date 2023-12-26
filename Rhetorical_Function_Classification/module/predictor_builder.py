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
from tqdm import tqdm

from module.beam import GNMTGlobalScorer
from module.cal_rouge import test_rouge, rouge_results_to_str
from module.neural import tile
from module.utlis_dataloader import load_to_cuda,pad_sent_entity
from transformers import top_k_top_p_filtering
from module import data
import copy
from sklearn.metrics import confusion_matrix
import json

oracle_labels = ["General descriptions of the topic","Reference to current state of knowledge","General reference to previous research or scholarship: research objective","General reference to previous research or scholarship: approaches taken","General reference to previous research or scholarship: about results","Reference to single investigations in the past:  about objective","Reference to single investigations in the past: about method","Reference to single investigations in the past: about result","Summarize the above references","Other reference purpose","Describing the objective","Describing the motivation","Describing used methods","Describing the results","Explaining the method relationship between own work and references","Explaining the objective relationship between own work and references","Explaining the result relationship between own work and references","Explaining the inadequacies of previous studies","Explain the significance of references","Other comments","Signalling Transition","Other functional sentences","Not sure"]

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
        self.device = device

        tensorboard_log_dir = self.args.model_path

        self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

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
        ct = 0
        gold_path = self.args.result_path + 'train_output.json'
        f = codecs.open(gold_path, 'w', 'utf-8')
        preds = []
        trues = []
        raw_sens = []
        total_json = []
        with torch.no_grad():
            for batch in tqdm(data_iter):
                batch = load_to_cuda(batch, self.device)
                label_logits = self.model(batch)
                pred = label_logits.max(1)[1]
                labels = batch['label']
                label_mask = labels.ne(104)
                labels = labels[label_mask]
                raw_sens.extend(batch['raw_sen'])

                preds.extend(pred.tolist())
                trues.extend(labels.tolist())

                i = 0
                for sens in batch['raw_sen']:
                    d = {}
                    d['sentences'] = sens
                    labs = pred[i:i+len(sens)]
                    d['label'] = [oracle_labels[p] for p in labs]
                    i = i+len(sens)
                    json.dump(d,f)
                    f.write('\n')

                f.flush()

                # f.flush()
        f.close()


        x = confusion_matrix(trues,preds)
        trues = torch.LongTensor(trues)
        preds = torch.LongTensor(preds)
        num_correct = preds.eq(trues).sum().item()
        acc = num_correct/trues.shape[0]

        print(acc)

        # pdb.set_trace()
        # raw_sens = sum(raw_sens,[])
        # preds = preds.tolist()
        # trues = trues.tolist()
        # for p,t,sen in zip(preds,trues,raw_sens):
        #     f.write(sen+'\n')
        #     f.write('###True: ' + oracle_labels[t] + '\n')
        #     f.write('###Prediction: ' + oracle_labels[p] + '\n\n')
        # f.close()



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

        sen_dec_state = self.model.sent_decoder.init_decoder_state(tarref_word, tar_ref_context)
        dec_states = self.model.decoder.init_decoder_state(tarref_word, tar_ref_context, with_cache=True)

        device = tar_ref_context.device
        batch_size = tar_ref_context.size(0)

        #use beam search
        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        sen_dec_state.map_batch_fn(
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

        start_sent = torch.full(
            [batch_size * beam_size, 1],
            self.start_doc,
            dtype=torch.long,
            device=device)

        tgt_sent_pos = torch.full(
            [batch_size * beam_size, 1],
            0,
            dtype=torch.long,
            device=device)

        tgt_ends = torch.full(
            [batch_size * beam_size, 1],
            0,
            dtype=torch.long,
            device=device)

        current_sent_seq = [[] for _ in range(batch_size*beam_size)]

        tgt_sent = self.model.sent_decoder.embeddings(start_sent) * math.sqrt(self.decoder.embeddings.embedding_dim)
        current_sent_state, _ = self.model.sent_decoder(tgt_ends, tgt_sent, tar_ref_context, sen_dec_state)

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
                          tar_ref_state, tar_ref_mask, wordsum, current_sent_state, tgt_ends, tgt_ends, dec_states,step = step)
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

            topks = topk_ids.view(-1).tolist()
            l_select_indices = select_indices.tolist()
            current_sent_seq_temp = copy.deepcopy(current_sent_seq)

            for index,(_id, indice) in enumerate(zip(topks, l_select_indices)):
                current_sent_seq[index] = current_sent_seq_temp[indice] + [_id]

            tgt_ends = tgt_ends.index_select(0, select_indices)
            current_sent_state = current_sent_state.index_select(0, select_indices)
            tgt_sent = tgt_sent.index_select(0, select_indices)

            sent_finished = topk_ids.eq(self.end_doc).reshape(-1)
            if sent_finished.any():
                x,y,z = current_sent_state.size()
                sen_index = sent_finished.int().eq(1)
                sen_index = sen_index.unsqueeze(1).unsqueeze(2).expand(x,y,z)
                #now_sent = self.tgt_encoder(tgt,tgt_starts,tgt_ends)
                current_sent_list = [torch.LongTensor(x[:-1]) for x in current_sent_seq]
                current_sent_tensor = pad_sent_entity(current_sent_list, self.pad_id,self.start_token,self.end_token, flatten = False)
                current_sent_tensor = current_sent_tensor.transpose(0,1).to(device)
                current_sent_repre = self.model.tgt_encoder(current_sent_tensor,tgt_ends,tgt_ends)
                
                temp = (~sent_finished).int().unsqueeze(1)
                
                tgt_ends = torch.cat((tgt_ends, temp),dim=-1)

                tgt_sent = torch.cat((tgt_sent,current_sent_repre),dim=1)

                next_sent_state, _ = self.model.sent_decoder(tgt_ends, tgt_sent, tar_ref_context, sen_dec_state)
                next_sent_state = next_sent_state[:,-1:,:]
                current_sent_state = torch.where(sen_index==True,next_sent_state,current_sent_state)

                nonzero_index = torch.nonzero(sent_finished==1).squeeze(1).tolist()
                for ind in nonzero_index:
                    current_sent_seq[ind] = []

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)

            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                tgt_ends = tgt_ends.view(-1,beam_size,tgt_ends.size(-1))
                tgt_sent = tgt_sent.view(-1,beam_size,tgt_sent.size(-2),tgt_sent.size(-1))
                current_sent_state = current_sent_state.view(-1,beam_size,current_sent_state.size(-2),current_sent_state.size(-1))
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
                tgt_ends = tgt_ends.index_select(0, non_finished) \
                    .view(-1, tgt_ends.size(-1))
                tgt_sent = tgt_sent.index_select(0, non_finished) \
                    .view(-1, tgt_sent.size(-2),tgt_sent.size(-1))
                current_sent_state = current_sent_state.index_select(0, non_finished) \
                    .view(-1, current_sent_state.size(-2),current_sent_state.size(-1))
                select_indices3 = non_finished.tolist()
                current_sent_seq_temp = copy.deepcopy(current_sent_seq)
                current_sent_seq = []
                for indice in select_indices3:
                    for rep in range(beam_size):
                        current_sent_seq.append(current_sent_seq_temp[indice*beam_size+rep])

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
            sen_dec_state.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results

