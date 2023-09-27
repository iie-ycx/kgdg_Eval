from metric import NLGEval
from transformers import BertTokenizer
import nltk
import codecs

def findint(s1,s2):
    if len(s2)>len(s1):
        s1,s2= s2,s1
    len1 = len(s1)
    len2 = len(s2)
    xmax = 0
    xindex = 0
    matrix = [[0]*len1 for i in range(len2)]
    for i,x in enumerate(s2):
        for j ,y in enumerate(s1):
            if x==y:
#                 matrix2[i][j]=1
                if i==0 or j ==0:  
                    matrix[i][j]=1  
                else:
                    matrix[i][j] = matrix[i-1][j-1]+1  
                if matrix[i][j]> xmax:  
                    xmax = matrix[i][j]  
                    xindex = j  

    return s1[xindex-xmax+1:xindex+1]

def DegenerationRate(pred_responses, knowledges, percent=0.7):

    sum_proportion = 0.
    count = 0.
    for pred, target in zip(pred_responses, knowledges):

        LCS = findint(pred,target)

        knowledge_proportion = len(LCS) / len(pred)
        sum_proportion = sum_proportion + knowledge_proportion

        if knowledge_proportion >= percent:
            count += 1

    results = count / len(pred_responses)
    mean_proportion = sum_proportion / len(pred_responses)

    return (results, mean_proportion)


def rounder_1(num):
    return round(num, 1)

def rounder_3(num):
    return round(num, 3)


def eval_BOW_file(run_file, ref_file, tokenizer=None):
    run_list = []
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            run_result = ' '.join(tokenized)
            run_list.append(run_result)
            
    ref_list = []
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            tokenized = (tokenizer(line))
            ref_result = ' '.join(tokenized)
            ref_list.append(ref_result)

    from metric import NLGEval

    metric = NLGEval(no_glove=False)

    metric_res, metric_res_list = metric.compute_metrics([ref_list], run_list)

    metric_res['Bleu_1'] = rounder_1(metric_res['Bleu_1'] * 100)
    metric_res['Bleu_2'] = rounder_1(metric_res['Bleu_2'] * 100)
    metric_res['Bleu_3'] = rounder_1(metric_res['Bleu_3'] * 100)
    metric_res['Bleu_4'] = rounder_1(metric_res['Bleu_4'] * 100)
    metric_res['METEOR'] = rounder_1(metric_res['METEOR'] * 100)
    metric_res['ROUGE_L'] = rounder_1(metric_res['ROUGE_L'] * 100)

    metric_res['Average'] = rounder_3(metric_res['EmbeddingAverageCosineSimilarity'])
    metric_res['Extrema'] = rounder_3(metric_res['VectorExtremaCosineSimilarity'])
    metric_res['Greedy'] = rounder_3(metric_res['GreedyMatchingScore'])

    del(metric_res['EmbeddingAverageCosineSimilarity'])
    del(metric_res['VectorExtremaCosineSimilarity'])
    del(metric_res['GreedyMatchingScore'])
    del(metric_res['CIDEr'])


    return metric_res



if __name__ == '__main__':

    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    ref_file = '../data/wow/raw_split/reference/seen_reference.txt'
    run_file = '../data/wow/raw_split/knowledge/seen_knowledge.txt'
    #run_file = '../output/wow/test_seen_0.txt'

    res = eval_BOW_file(run_file, ref_file, tokenizer)

    print(res)


