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

if __name__ == '__main__':
    
    from transformers import BertTokenizer
    def bert_tokenizer():
        t = BertTokenizer.from_pretrained(
            "/data/ycx/pretrain-models/bert-base-uncased", do_lower_case=True)  # do_lower_case Whether to lower case the input.
        return t.tokenize, t.vocab, t.ids_to_tokens

    tokenizer, vocab2id, id2vocab = bert_tokenizer()

    run_file = '/data/yangchenxu/degeneration/NT-test/output/wow/test_seen_14.txt'
    #run_file = '/data/yangchenxu/degeneration/ND-test/output/wow/test_seen_14.txt'
    
    knowledge_file = '/data/yangchenxu/degeneration/bart/alpha4/data/wow/raw_split/knowledge/seen_knowledge.txt'
    target_file = '../data/wow/raw_split/reference/seen_reference.txt'

    preds = []
    knowledges = []
    targets = []
    with codecs.open(run_file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                preds.append(tokenized)

    with codecs.open(knowledge_file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                knowledges.append(tokenized)

    with codecs.open(target_file, encoding='utf-8') as f:
            for line in f:
                tokenized = (tokenizer(line))
                targets.append(tokenized)

    res = DegenerationRate(preds, knowledges, percent=0.75)

    print(res)

