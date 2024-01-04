import pandas as pd
import os
import codecs
import csv
import re
import sklearn
from datasets import load_dataset

def load_data(path):
    df = pd.read_csv(path)
    df['Text1'] = df['Text'].apply(lambda x: re.split(r'\n|\t|\\n', x)[0])
    df['Text2'] = df['Text'].apply(lambda x: re.split(r'\n|\t|\\n', x)[1])
    df.drop("Text", axis="columns")
    return df

# def load_train_data(path):
#     """load text and scores from file"""
#     # with codecs.open(path, encoding='utf-8-sig') as f:
#     #     sent_pairs = []
#     #     score = []
#     #     for row in csv.DictReader(f, skipinitialspace=True):
#     #         sent_pair = re.split(r'\n|\t|\\n', row["Text"])
#     #         sent_pairs.append(sent_pair)
#     #         score.append(float(row['Score']))
#     # return sent_pairs, score
#     df = pd.read_csv(path)
#     df['Text'] = df['Text'].apply(lambda x: re.split(r'\n|\t|\\n', x))
#     return df
#
# def load_aim_data(path):
#     """load text from trach C file"""
#     with codecs.open(path, encoding='utf-8-sig') as f:
#         sent_pairs = []
#         for row in csv.DictReader(f, skipinitialspace=True):
#             sent_pair = re.split(r'\n|\t|\\n', row["Text"])
#             # sent_pair = row["Text"].split('\n')
#             # print(sent_pair)
#             sent_pairs.append(sent_pair)
#
#     return sent_pairs
#
#
# def get_batches(batch_size, text_pairs, labels=None, shuffle=True):
#     total_data_size = len(text_pairs)
#     index_ls = [i for i in range(total_data_size)]
#
#     if shuffle:
#         dataset = sklearn.utils.shuffle(index_ls)
#
#     if labels:
#         for start_i in range(0, total_data_size, batch_size):
#             # get batch_texts, batch_labels
#             end_i = min(total_data_size, start_i + batch_size)
#             batch_text_pairs = text_pairs[start_i:end_i]
#             batch_labels = labels[start_i:end_i]
#             yield batch_text_pairs, batch_labels
#
#     else:
#         for start_i in range(0, total_data_size, batch_size):
#             # get batch_texts
#             end_i = min(total_data_size, start_i + batch_size)
#             batch_text_pairs = text_pairs[start_i:end_i]
#             yield batch_text_pairs




if __name__ == '__main__':
    train_file = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
    test_file = '../Semantic_Relatedness_SemEval2024-main/Track C/amh/amh_dev.csv'

    train_data = load_data(train_file)
    test_data = load_data(test_file)
    print(train_data['Text1'][:5])
    print(test_data)




