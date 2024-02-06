#!/usr/bin/env python
import argparse
from preprocessing import load_data, get_batches
from model_adapters import bertmodel, get_biencoder_encoding, get_crossencoder_encoding, arb_adapter, amh_adapter, ind_adapter, eng_adapter
from datasets import Dataset
import torch
import adapters.composition as ac
from model_build import BiEncoderNN, CrossEncoderNN
from tqdm import tqdm
import warnings
import logging

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

# license

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
    # create a BiEncoderNN instance
    biencoder_baseline_model = BiEncoderNN(bertmodel)
    # load the fully fine-tuned bert model without adapters
    loaded_model_state_dict = torch.load('../Model/biencoder_baseline_model.pt')
    biencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False)  # "strict=False" necessary, otherwise runtime error
    # Baseline model without setting active any adapters
    biencoder_baseline_model.model.set_active_adapters(None)

    # predict and save the result
    baseline_scores, baseline_sample_ids = biencoder_baseline_model.predict(biencoder_dataset, output_path=f'../result/{lang}/{lang}_biencoder_baseline.csv')
    print(f"Run Fully Fine-Tuned BiEncoder Baseline Model without Adapters on {lang}: ")
    print(baseline_scores, baseline_sample_ids)

    """CrossEncoder_Baseline"""
    # encoding
    crossencoder_dataset = get_crossencoder_encoding(dataset)
    # create a CrossEncoderNN instance
    crossencoder_baseline_model = CrossEncoderNN(bertmodel)
    # load the fully fine-tuned bert model without adapters
    loaded_model_state_dict = torch.load('../Model/crossencoder_baseline_model.pt')
    crossencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False)  # "strict=False" necessary, otherwise runtime error
    # Baseline model without setting active any adapters
    crossencoder_baseline_model.model.set_active_adapters(None)

    # predict and save the result
    baseline_scores, baseline_sample_ids = crossencoder_baseline_model.predict(crossencoder_dataset, output_path=f'../result/{lang}/{lang}_crossencoder_baseline.csv')
    print(f"Run Fully Fine-Tuned CrossEncoder Baseline Model without Adapters on {lang}: ")
    print(baseline_scores, baseline_sample_ids)

    """BiEncoder"""
    # create a BiEncoderNN instance
    biencoder_model = BiEncoderNN(bertmodel)
    # load the bert model with fine-tuned LA(eng) and TA(STR)
    loaded_model_state_dict = torch.load('../Model/biencoder_model.pt')
    biencoder_model.load_state_dict(loaded_model_state_dict)
    # set active a specific LA and the TA(STR)
    biencoder_model.model.set_active_adapters((ac.Stack(lang_adapter, "STR")))

    # predict and save the result
    biencoder_scores, biencoder_sample_ids = biencoder_model.predict(biencoder_dataset, output_path=f'../result/{lang}/{lang}_biencoder.csv')
    print(f"Run BiEncoder Model with Fine-Tuned TA and LA on {lang}: ")
    print(biencoder_scores, biencoder_sample_ids)

    """CrossEncoder"""
    # create a CrossEncoderNN instance
    crossencoder_model = CrossEncoderNN(bertmodel)
    # load the bert model with fine-tuned LA(eng) and TA(STR)
    loaded_model_state_dict = torch.load('../Model/crossencoder_model.pt')
    crossencoder_model.load_state_dict(loaded_model_state_dict)
    # set active a specific LA and the TA(STR)
    crossencoder_model.model.set_active_adapters((ac.Stack(lang_adapter, "STR")))

    # predict and save the result
    crossencoder_scores, crossencoder_sample_ids = crossencoder_model.predict(crossencoder_dataset, output_path=f'../result/{lang}/{lang}_crossencoder.csv')
    print(f"Run CrossEncoder Model with Fine-Tuned TA and LA on {lang}: ")
    print(crossencoder_scores, crossencoder_sample_ids)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run model on different datasets.")
    # parser.add_argument("--language", "-l", type=str, required=True, choices=["arb", "amh", "ind", "eng"], help="Test Language")
    #
    # args = parser.parse_args()
    # main(args.language)
    main('arb')
    main('amh')
    main('ind')
    main('eng')

    """To execute this script in the terminal, use the following command as an example:"""
    # Example: python Model/model_predict.py -l arb
