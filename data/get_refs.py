import sys
import os
import nltk
import codecs
from sys import *
import random
from transformers import BartTokenizer, BlenderbotTokenizer
from tqdm import tqdm
import json
import re
import pdb

def load_answer(file, tokenizer):
    print("load_answer")
    answer = []
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 3)

            assert len(temp) == 4,"all_previous_query_id;all_previous_query_id;all_previous_query_id	current_query_id	background_id;background_id 	response_content"
            if len(temp[0]) < 1:
                temp[0] = []
            else:
                temp[0] = temp[0].split(';')
            temp[2] = temp[2].split(';')
            
            
            answer.append(temp)
    return answer


def load_passage(file, pool, tokenizer):  # background_id	background_content
    print("load_passage")
    poolset = set()
    for k in pool:
        poolset.update(pool[k])

    passage = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t', 1)
            assert len(temp) == 2, "load_passage"
            denoise = re.sub(r'^.*__knowledge__ ', '', temp[1])
            if temp[0] in poolset:
                passage[temp[0]] = denoise
                
    print("passage:{}, poolset:{}".format(len(passage), len(poolset)))
    return passage  # {background_id1:background_content, background_id2:background_content}


def load_pool(file):  # current_query_id Q0 background_id rank relevance_score model_name
    print("load_pool")
    pool = {}
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 6, "load_pool"
            if temp[0] not in pool:
                pool[temp[0]] = [temp[2]]  # {“current_query_id”:[background_id1]}
            else:
                pool[temp[0]].append(temp[2])  # {“current_query_id”:[background_id1,background_id2,background_id3...]}
    return pool


def load_qrel(file):
    print("load_qrel")
    qrel = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            assert len(temp) == 4, "load_qrel"
            if int(temp[3]) > 0:
                qrel[temp[0]] = temp[2]  # {current_query_id:background_id1, current_query_id2:background_id2........}
    return qrel

#对话内容  utterance 一轮10句
def load_query(file, tokenizer):  # query_id	query_content
    print("load_query")
    query = dict()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t',1)
            assert len(temp) == 2, "load_query"
            query[temp[0]] = temp[1]  # {1_1:[query_tokens],}
    return query


def load_split(dataset, file):
    train = set()
    dev = set()
    test_seen = set()
    test_unseen = set()
    with codecs.open(file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t')
            assert len(temp) == 2, "query_id train/dev/test_seen/test_unseen"
            if temp[1] == 'train':
                train.add(temp[0])
            elif temp[1] == 'dev':
                dev.add(temp[0])
            elif temp[1] == 'test_seen':
                test_seen.add(temp[0])
            elif temp[1] == 'test_unseen':
                test_unseen.add(temp[0])
    return train, dev, test_seen, test_unseen


def split_data(dataset, split_file, episodes):
    print("split_data:", dataset)
    train_episodes = list()
    dev_episodes = list()

    train, dev, test_seen, test_unseen = load_split(dataset, split_file)
    test_seen_episodes = list()
    test_unseen_episodes = list()
    for episode in episodes:
        if episode[0]['query_id'] in train:
            train_episodes.append(episode)
        elif episode[0]['query_id'] in dev:
            dev_episodes.append(episode)
        elif episode[0]['query_id'] in test_seen:
            test_seen_episodes.append(episode)
        elif episode[0]['query_id'] in test_unseen:
            test_unseen_episodes.append(episode)
    return train_episodes, dev_episodes, test_seen_episodes, test_unseen_episodes


def load_default(answer_file, passage_file, pool_file, qrel_file, query_file, tokenizer, randoms=1):
    answer = load_answer(answer_file, tokenizer)  # [[all_previous_query_ids],current_query_id,[background_ids],[response_tokens]]
    
    pool = load_pool(pool_file)  # {“current_query_id1”:[background_id1,background_id2,background_id3...]，“current_query_id2”:[background_id1,background_id2,background_id3...]}
    query = load_query(query_file, tokenizer)  # {current_query_id_1:[query_tokens],current_query_id_2:[query_tokens]}
    passage = load_passage(passage_file, pool, tokenizer)  # {background_id1:[background_tokens], [background_id2:[background_tokens]}
    average_pool = 0

    episodes = []
    ini_episode_index = "?"
    examples = []
    episode_index = []


    for i in tqdm(range(len(answer))):
        for j in range(randoms):
            c_id, q_id, knowledge_id, response = answer[i]  # c_id is a lis，q_id is string，p_id is a list，ans is a list

            knowledge_pool = pool[q_id]

            average_pool += len(knowledge_pool)

            for p in knowledge_id:  # label knowledge sentence id
                if p not in knowledge_pool:
                    raise Exception("label knowledge is not in knowledge pool")

            # we want the correct knowledge to always be in index 0
            k = knowledge_pool.index(knowledge_id[0])
            if k == 0:
                pass
            else:
                knowledge_pool[0], knowledge_pool[k] = knowledge_pool[k], knowledge_pool[0]

            example = dict()
            example['context_id'] = c_id  # list ：[previous utterance]
            example['query_id'] = q_id  # string ：current query
            example['response'] = response  # list
            

            example['knowledge_pool'] = knowledge_pool  # list
            example['knowledge_label'] = knowledge_id

            example['answer_file'] = answer_file
            example['passage_file'] = passage_file
            example['pool_file'] = pool_file
            example['query_file'] = query_file

            current_episode_index = "_".join(q_id.split("_")[:-1])

            if current_episode_index != ini_episode_index:
                if len(examples) == 0:
                    pass
                else:
                    #print("episode_index:", ini_episode_index)
                    episode_index.append(ini_episode_index)
                    episodes.append(examples)  # [[{example1},{example2},{example3}],[{example1},{example2},{example3}],...]
                    examples = []
                ini_episode_index = current_episode_index

            examples.append(example)  # [{example1},{example2},{example3}...]

            if i == (len(answer)-1):
                #print("episode_index:", current_episode_index)
                episode_index.append(current_episode_index)
                episodes.append(examples)
                examples = []

    total_number_examples = sum([len(episode) for episode in episodes])
    print('total episodes:', len(episodes))
    print('total examples:', total_number_examples)
    print('the lowest length of episodes:', min([len(episode) for episode in episodes]))
    print('the maximum length of episodes:', max([len(episode) for episode in episodes]))
    print('average length of episodes:', total_number_examples/len(episodes))
    print("average knowledge pool:", average_pool / total_number_examples)

    return episodes, query, passage

def get_ref_response(episodes, query, passage):

    responses = []

    for ids in range(len(episodes)):
        episode = episodes[ids]
        
        for id_in_episode, example in enumerate(episode):
            
            # process for response
            response = example['response']
            
            responses.append(response)

    return responses





def get_context(episodes, query, passage):
    contexts = []

    for ids in range(len(episodes)):
        episode = episodes[ids]
        
        for id_in_episode, example in enumerate(episode):
            context = ''
            cur_query = query[example['query_id']]
            
            
            hist_list = example['context_id']
            if len(hist_list) == 0:
                pass
            else:
                for hist in hist_list:
                    history = query[hist]
                    context = context + '\t' + history
            context = context + '\t' + cur_query
            contexts.append(context)
        #pdb.set_trace()
    return contexts





def get_knowledge(episodes, query, passage):

    knowledges = []

    for ids in range(len(episodes)):
        episode = episodes[ids]
        
        for id_in_episode, example in enumerate(episode):
            
            # process for response
            pid = example['knowledge_pool'][0]

            p = passage[pid]
            
            knowledges.append(p)

    return knowledges



if __name__ == '__main__':
    import torch
    
    data_path = './wow/raw_data/'
    

    query = torch.load(data_path + 'query_wow.pkl')
    passage = torch.load(data_path + 'passage_wow.pkl')
    train_episodes = torch.load(data_path + 'train_wow.pkl')
    dev_episodes = torch.load(data_path + 'dev_wow.pkl')
    test_seen_episodes = torch.load(data_path + 'test_seen_wow.pkl')
    test_unseen_episodes = torch.load(data_path + 'test_unseen_wow.pkl')


    context_seen_output_path = './wow/raw_split/context/seen_context.txt'
    context_unseen_output_path = './wow/raw_split/context/unseen_context.txt'


    context_seen_responses = get_context(test_seen_episodes, query, passage)
    context_unseen_responses = get_context(test_unseen_episodes, query, passage)

    
    with open(context_seen_output_path, 'w', encoding='utf-8') as fout:
        for response in context_seen_responses:
            
            fout.write(response + '\n')

    with open(context_unseen_output_path, 'w', encoding='utf-8') as fout:
        for response in context_unseen_responses:
            
            fout.write(response + '\n')


