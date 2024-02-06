#!/usr/bin/env python
import sys
import os.path
import pandas as pd
import numpy as np
from scipy import stats
import pandas as pd


import pandas as pd
import numpy as np
from scipy import stats

def evaluate_spearman(pred_file, gold_file):
    try:
        # read files
        pred_df = pd.read_csv(pred_file).rename(columns={'pairid': 'PairID'})
        gold_df = pd.read_csv(gold_file, usecols=['PairID', 'Score']).rename(columns={'Score': 'gold_score'})

        # merge
        merged_df = pd.merge(pred_df, gold_df, how='inner', on='PairID')

        # 计算Spearman相关性
        spearman_corr = stats.spearmanr(merged_df['pred_score'], merged_df['gold_score'])[0]

        return spearman_corr

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def spearman_corr(gold_file_path, pred_files):
    for pred_file in pred_files:
        pred_file_path = "../result" + pred_file + ".csv"
        spearman_corr = evaluate_spearman(pred_file_path, gold_file_path)
        if spearman_corr is not None:
            print(f"Spearman Correlation between {pred_file} and gold_file: {spearman_corr}")


if __name__ == "__main__":

    # spearman correlation of eng
    gold_file_path = "../data/Track A/eng/eng_dev_with_labels.csv"
    pred_files = ["/eng/eng_biencoder_baseline", "/eng/eng_crossencoder_baseline", "/eng/eng_biencoder", "/eng/eng_crossencoder"]
    spearman_corr(gold_file_path, pred_files)

    # spearman correlation of amh
    gold_file_path = "../data/Track A/amh/amh_dev_with_labels.csv"
    pred_files = ["/amh/amh_biencoder_baseline", "/amh/amh_crossencoder_baseline", "/amh/amh_biencoder", "/amh/amh_crossencoder"]
    spearman_corr(gold_file_path, pred_files)
