import sys

import json
import os
import time

import codecs
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk import ngrams
from collections import Counter

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import scipy.stats
import math

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


def seq_rep_n_per(pred_responses, knowledges, n, percent):
    results = 0.
    zero = 0.
    count = 0.
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
        if result > percent:
            count += 1

    results = count / (len(pred_responses)- zero)
    return results

def seq_rep_n_num(pred_responses, knowledges, n):
    results = []
    zero = 0.
    count = 0.
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

        results.append(result)
    return results

def seq_rep_n_histogram(pred_responses, knowledges, n):
    results = 0.
    zero = 0.
    percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
        for j in range(10):
            if result < percents[j]:
                count[j] += 1
                break
    count_np = np.array(count)
    results = list(count_np / (len(pred_responses)- zero))        
    
    return results   


def hist_dist_difference_MSE(ts_seen_pred_responses, ts_seen_knowledge, ref_res, n=4):

    MSE_seen_list = []


    for i in range(n):
        pred_seen_res = seq_rep_n_histogram(ts_seen_pred_responses, ts_seen_knowledge, i+1)
        
        MSE_seen = mean_squared_error(pred_seen_res,ref_res)

        MSE_seen_list.append(MSE_seen)


    return MSE_seen_list


def hist_dist_difference_MAE(ts_seen_pred_responses, ts_seen_knowledge, ref_res, n=4):

    MAE_seen_list = []

    for i in range(n):
        pred_seen_res = seq_rep_n_histogram(ts_seen_pred_responses, ts_seen_knowledge, i+1)
        
        MAE_seen = mean_absolute_error(pred_seen_res, ref_res[i])

        MAE_seen_list.append(MAE_seen)
    
    MAE_mean = sum(MAE_seen_list) / 4
    MAE_seen_list.append(MAE_mean)

    return MAE_seen_list



def hist_dist_difference_KL(ts_seen_pred_responses, ts_seen_knowledge, ref_res, n=4):

    KL_seen_list = []

    for i in range(n):
        pred_seen_res = seq_rep_n_histogram(ts_seen_pred_responses, ts_seen_knowledge, i+1)
            
        KL_seen = scipy.stats.entropy(pred_seen_res,ref_res)

        KL_seen_list.append(KL_seen)

    return KL_seen_list


def seq_percent_n_gram(ts_seen_pred_responses, ts_seen_knowledge):
    
    n_grams = [8, 16, 24, 32]
    copying_metircs_list = []
    for n_gram in n_grams:
        pred_seen = seq_rep_n_per(ts_seen_pred_responses, ts_seen_knowledge, n_gram, 0)
        copying_metircs_list.append(pred_seen)
        #print('%d_gram co-occurence proportion pred_seen: %.2f pred_unseen: %.2f ref: %.2f ' \
        #    %(n_gram, pred_seen, pred_unseen, ref_all))
    copying_mean = sum(copying_metircs_list) / 4
    copying_metircs_list.append(copying_mean)

    return copying_metircs_list


def seq_percent_n_gram_ref(target_responses, knowledges):

    n_grams = [8, 16, 24, 32]

    for n_gram in n_grams:
        ref_all = seq_rep_n_per(target_responses, knowledges, n_gram, 0)
        print('%d_gram co-occurence proportion ref: %.2f ' \
            %(n_gram, ref_all*100))


def custom_eval_metrics_file(response_file, knowledge_file, targets, knowledges, tokenizer):

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
            know_result = ' '.join(tokenized)
            knowledge_list.append(know_result)
    
    ref_res = []
    for i in range(4):
        ref_res.append(seq_rep_n_histogram(targets, knowledges, i+1))


    MAE_list = hist_dist_difference_MAE(response_list, knowledge_list, ref_res, n=4)
    copying = seq_percent_n_gram(response_list, knowledge_list)

    return {'copying-avg':rounder(copying[4]*100), 'MAE-avg': rounder(MAE_list[4]*100), 'copying-1':rounder(copying[0]*100), 'copying-2':rounder(copying[1]*100),'copying-3':rounder(copying[2]*100),'copying-4':rounder(copying[3]*100),\
         'MAE-1': rounder(MAE_list[0]*100), 'MAE-2': rounder(MAE_list[1]*100), 'MAE-3': rounder(MAE_list[2]*100), 'MAE-4': rounder(MAE_list[3]*100)
     }

if __name__ == '__main__':

    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    
    run_file = '../data/wow/raw_split/reference/seen_reference.txt'
    knowledge_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    #run_file = '../output/wow/test_seen_0.txt'


    target_file_1 = '../data/wow/raw_split/reference/train_reference.txt'
    target_file_2 = '../data/wow/raw_split/reference/dev_reference.txt'
    target_file_3 = '../data/wow/raw_split/reference/seen_reference.txt'
    target_file_4 = '../data/wow/raw_split/reference/unseen_reference.txt'
    target_files = [target_file_1, target_file_2, target_file_3, target_file_4]


    knowledge_file_1 = '../data/wow/raw_split/knowledge/train_knowledge.txt'
    knowledge_file_2 = '../data/wow/raw_split/knowledge/dev_knowledge.txt'
    knowledge_file_3 = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    knowledge_file_4 = '../data/wow/raw_split/knowledge/unseen_knowledge.txt'
    knwoledge_files = [knowledge_file_1, knowledge_file_2, knowledge_file_3, knowledge_file_4]

    targets = []
    knowledges = []

    for file in target_files:
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                run_result = ' '.join(tokenized)
                targets.append(run_result)

    for file in knwoledge_files:
        with codecs.open(file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                run_result = ' '.join(tokenized)
                knowledges.append(run_result)

    result = custom_eval_metrics_file(run_file, knowledge_file, targets, knowledges, tokenizer)
    print(result)
    #seq_percent_n_gram_ref(targets, knowledges)

