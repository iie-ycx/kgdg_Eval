import sys
import glob
import json
import os
import time
import codecs
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams
from collections import Counter

def rounder(num):
    return round(num, 2)

def get_ngram(sent, n_size, label=None):

    def _ngram(sent, n_size):
        ngram_list = []
        for left in range(len(sent) - n_size):
            ngram_list.append(sent[left:left + n_size + 1])
        return ngram_list

    ngram_list = _ngram(sent, n_size)
    if label is not None:
        ngram_list = [ngram + '_' + label for ngram in ngram_list]
    return ngram_list

def seq_rep_n(pred_responses, knowledges, n):
    results = 0.
    zero = 0.
    for pred, target in zip(pred_responses, knowledges):
        pred_tokens = pred.split(' ')
        target_token = target.split(' ')

        ngs_pred = [ng for ng in ngrams(pred_tokens, n)]
        ngs_know = [ng for ng in ngrams(target_token, n)]

        overlap_num = 0
        x = Counter(ngrams(pred_tokens, n))
        for ngram in x:
            if ngram in ngs_know:
                overlap_num += x[ngram]
        if len(ngs_pred) == 0:
            zero += 1
            continue
        result = overlap_num / len(ngs_pred)
        results += result
    results = results / (len(pred_responses)- zero)
    return results

def eval_seq_rep_n_file(response_file, knowledge_file, tokenizer, n):
    seq_rep_list = []
    response_list = []
    with codecs.open(response_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            run_result = ' '.join(tokenized)
            response_list.append(run_result)
            

    knowledge_list = []
    with codecs.open(knowledge_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            ref_result = ' '.join(tokenized)
            knowledge_list.append(ref_result)

    for i in range(n):
        seq_rep_list.append(seq_rep_n(response_list, knowledge_list, i+1))

    
    return {'seq_rep-1': rounder(seq_rep_list[0]*100), 'seq_rep-2': rounder(seq_rep_list[1]*100), 'seq_rep-3': rounder(seq_rep_list[2]*100), 'seq_rep-4': rounder(seq_rep_list[3]*100)}


if __name__ == '__main__':

    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data1/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    
    run_file = '../data/wow/raw_split/reference/seen_reference.txt'
    knowledge_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    #run_file = '../output/wow/test_seen_0.txt'

    seq_rep = eval_seq_rep_n_file(run_file, knowledge_file, tokenizer, 4)

    print("seq_rep:", seq_rep)