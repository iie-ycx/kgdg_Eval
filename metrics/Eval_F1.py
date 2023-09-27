import codecs

import re
from collections import Counter

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.

    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values

    :return: tuple (p, r, f1) for precision, recall, f1
    """
    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def _f1_score(guess, answers):
    """Return the max F1 score between the guess and *any* answer."""
    if guess is None or answers is None:
        return 0
    g_tokens = normalize_answer(guess).split()
    scores = [
        _prec_recall_f1_score(g_tokens, normalize_answer(a).split()) for a in answers
    ]
    return max(f1 for p, r, f1 in scores)

def rounder(num):
    return round(num, 2)


def eval_f1_file(run_file, ref_file, tokenizer=None):
    run_list = []
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            run_result = ' '.join(tokenized)
            #print(run_result)
            #exit()
            run_list.append(run_result)
            

    ref_list = []
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            ref_result = ' '.join(tokenized)
            ref_list.append(ref_result)


    f1 = 0.
    for id in range(len(ref_list)):
        f1 += _f1_score(run_list[id], [ref_list[id]])
        
        
    return {'F1': rounder(f1*100/len(run_list))}

if __name__ == '__main__':
    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    ref_file = '../data/wow/raw_split/reference/seen_reference.txt'
    run_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    prefix = '/data/yangchenxu/degeneration/unlike-all-avg'
    #prefix = '..'
    run_file = prefix + '/output/wow/test_seen_11.txt'

    f1 = eval_f1_file(run_file, ref_file, tokenizer)
    print("f1", f1)