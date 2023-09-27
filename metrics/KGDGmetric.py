from metric import NLGEval
from transformers import BertTokenizer
import nltk
import codecs
from Eval_MAE_dist import custom_eval_metrics_file
from Eval_BOW import eval_BOW_file
from degen_rate import DegenerationRate

def rounder_1(num):
    return round(num, 1)

def rounder_3(num):
    return round(num, 3)

def eval_degen_file(run_file, knowledge_file, tokenizer):

    preds = []
    knowledges = []

    with codecs.open(run_file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                preds.append(tokenized)

    with codecs.open(knowledge_file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                knowledges.append(tokenized)

    res, _ = DegenerationRate(preds, knowledges, percent=0.75)
    res_dic = {}
    res_dic['degen_rate'] = res

    return res_dic

if __name__ == '__main__':

    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()


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

    ref_file = '../data/wow/raw_split/reference/seen_reference.txt'
    run_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    knowledge_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'

    res_general = eval_BOW_file(run_file, ref_file, tokenizer)
    res_degen = eval_degen_file(run_file, knowledge_file, tokenizer)
    custom_eval_metrics = custom_eval_metrics_file(run_file, knowledge_file, targets, knowledges, tokenizer)
    
    
    res = {}

    res.update(res_general)
    res.update(res_degen)
    res.update(custom_eval_metrics)

    print(res)
    
