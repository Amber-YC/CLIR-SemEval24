import pandas as pd
import os
import codecs
import csv

def load_train_data(path):
    """load text and scores from file"""
    with codecs.open(path, encoding='utf-8-sig') as f:
        text_pair = []
        score = []
        for row in csv.DictReader(f, skipinitialspace=True):
            sent_pair = row['Text'].split('\n')
            text_pair.append(tuple(sent_pair))
            score.append(float(row['Score']))
    return text_pair, score

def load_eval_data(path):
    """load text from trach C file"""
    with codecs.open(path, encoding='utf-8-sig') as f:
        text_pair = []
        for row in csv.DictReader(f, skipinitialspace=True):
            sent_pair = row['Text'].split('\t')
            text_pair.append(tuple(sent_pair))
    return text_pair

if __name__ == '__main__':
    train_file = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
    test_file = '../Semantic_Relatedness_SemEval2024-main/Track C/amh/amh_dev.csv'
    train = load_train_data(train_file)
    test = load_eval_data(test_file)
    print(train[0]) # a list of tuples consisting of text pairs from training set
    print(train[1]) # a list of scores of each text_pair from training set
    print(test) # # a list of tuples consisting of text pairs




