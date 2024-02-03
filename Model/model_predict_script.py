#!/usr/bin/env python
import argparse
from preprocessing import load_data, get_batches
from adapters_model import get_biencoder_encoding, get_crossencoder_encoding, arb_adapter, amh_adapter, ind_adapter, eng_adapter
from datasets import Dataset
import torch
import adapters.composition as ac
from adapters_model import bertmodel
from model_build import BiEncoderNN, CrossEncoderNN
from tqdm import tqdm
import warnings
import logging

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")


def main(lang):
    # load dataset
    if lang == 'arb':
        dataset_path = '../data/Track C/arb/arb_dev.csv'
        lang_adapter = arb_adapter
    elif lang == 'amh':
        dataset_path = '../data/Track C/amh/amh_dev.csv'
        lang_adapter = amh_adapter
    elif lang == 'ind':
        dataset_path = '../data/Track C/ind/ind_dev.csv'
        lang_adapter = ind_adapter
    elif lang == 'eng':
        dataset_path = '../data/Track C/eng/eng_dev.csv'
        lang_adapter = eng_adapter

    data = load_data(dataset_path)
    dataset = Dataset.from_pandas(data[['PairID', 'pairs']])
    print(f"{dataset_path} data is loaded.")


    """BiEncoder_Baseline"""
    # encoding
    biencoder_dataset = get_biencoder_encoding(dataset)
    # creat an CrossEncoderNN instance
    biencoder_baseline_model = BiEncoderNN(bertmodel)
    # load fine-tuned model
    loaded_model_state_dict = torch.load('../Model/biencoder_baseline_model.pt')
    biencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False)  # "strict=False" necessary, otherwise runtime error
    biencoder_baseline_model.model.set_active_adapters(None)

    # prediction and saving the result
    baseline_scores, baseline_sample_ids = biencoder_baseline_model.predict(biencoder_dataset, output_path=f'../result/{lang}/{lang}_biencoder_baseline.csv')
    print(f"Run Fine-Tuned BiEncoder Baseline Model without Adapters on {lang}: ")
    print(baseline_scores, baseline_sample_ids)


    """CrossEncoder_Baseline"""
    # encoding
    crossencoder_dataset = get_crossencoder_encoding(dataset)
    # creat an CrossEncoderNN instance
    crossencoder_baseline_model = CrossEncoderNN(bertmodel)
    # load fine-tuned model
    loaded_model_state_dict = torch.load('../Model/crossencoder_baseline_model.pt')
    crossencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False)  # "strict=False" necessary, otherwise runtime error
    crossencoder_baseline_model.model.set_active_adapters(None)

    # prediction and saving the result
    baseline_scores, baseline_sample_ids = crossencoder_baseline_model.predict(crossencoder_dataset, output_path=f'../result/{lang}/{lang}_crossencoder_baseline.csv')
    print(f"Run Fine-Tuned CrossEncoder Baseline Model without Adapters on {lang}: ")
    print(baseline_scores, baseline_sample_ids)


    """BiEncoder"""
    # creat an BiEncoderNN instance
    biencoder_model = BiEncoderNN(bertmodel)
    # load fine-tuned model
    loaded_model_state_dict = torch.load('../Model/biencoder_model.pt')
    biencoder_model.load_state_dict(loaded_model_state_dict)
    biencoder_model.model.set_active_adapters((ac.Stack(lang_adapter, "STR")))  # set active adapters

    # prediction and saving the result
    biencoder_scores, biencoder_sample_ids = biencoder_model.predict(biencoder_dataset, output_path=f'../result/{lang}/{lang}_crossencoder.csv')
    print(f"Run Fine-Tuned BiEncoder Model with Adapters on {lang}: ")
    print(biencoder_scores, biencoder_sample_ids)


    """CrossEncoder"""
    # creat an CrossEncoderNN instance
    crossencoder_model = CrossEncoderNN(bertmodel)
    # load fine-tuned model
    loaded_model_state_dict = torch.load('../Model/crossencoder_model.pt')
    crossencoder_model.load_state_dict(loaded_model_state_dict)
    crossencoder_model.model.set_active_adapters((ac.Stack(lang_adapter, "STR")))

    # prediction and saving the result
    crossencoder_scores, crossencoder_sample_ids = crossencoder_model.predict(crossencoder_dataset, output_path=f'../result/{lang}/{lang}_biencoder.csv')
    print(f"Run Fine-Tuned CrossEncoder Model with Adapters on {lang}: ")
    print(crossencoder_scores, crossencoder_sample_ids)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on different datasets.")
    parser.add_argument("--language", "-l", type=str, required=True, choices=["arb", "amh", "ind", "eng"], help="Test Language")

    args = parser.parse_args()
    main(args.language)
    # main('arb')

