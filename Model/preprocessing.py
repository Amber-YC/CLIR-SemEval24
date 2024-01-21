import pandas as pd
import os
import codecs
import csv
import re
import sklearn
from datasets import load_dataset
from torch.utils.data import DataLoader

def load_data(path):
    df = pd.read_csv(path)
    df["pairs"] = df['Text'].apply(lambda x: re.split(r'\n|\t|\\n', x)) # easier to split training and val set
    # always use .apply to deal with df columns/rows
    df.drop("Text", axis="columns")
    return df



def get_batches(batch_size, data, shuffle=True):
    # total_data_size = len(data)
    # index_ls = [i for i in range(total_data_size)]
    #
    # if shuffle:
    #     data = sklearn.utils.shuffle(data)
    #
    # for start_i in range(0, total_data_size, batch_size):
    #     # get batch_texts
    #     end_i = min(total_data_size, start_i + batch_size)
    #     batch_text_pairs = data[start_i:end_i]
    #     yield batch_text_pairs
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    #return data_loader
    # RAM issue
    for batch in data_loader:
            yield batch



if __name__ == '__main__':
    train_file = '../data/Track A/eng/eng_train.csv'
    test_file = '../data/Track C/amh/amh_dev.csv'

    train_data = load_data(train_file)
    test_data = load_data(test_file)
    print(train_data["pairs"][:5])
    print(test_data)




