from transformers import AutoTokenizer, BertTokenizer
from preprocessing import load_data
from adapters import AutoAdapterModel, AdapterConfig, AdapterSetup, BnConfig, BertAdapterModel
import numpy as np
import torch.nn as nn
import torch
import adapters.composition as ac
from datasets import Dataset
# from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
# from sentence_transformers import BiEncoder

# license

"""Load tokenizer"""
# we must keep the pretrained-transformer of the tokenizer and model the same
berttokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


"""Load pre-trained BERT model from Hugging Face Hub"""
# The `BertAdapterModel` class is specifically designed for working with adapters
bertmodel = BertAdapterModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)


"""Setup Adapters"""


def set_lang_adapter(model, lang: str, non_linearity="relu", reduction_factor=2):
    """set language adapter for each language (eng, amh, arb, ind)
    parameters:
    - lang: pretrained-adapter name for each language,
            eng: en/wiki@ukp
            arb: ar/wiki@ukp
            amh: am/wiki@ukp
            ind: id/wiki@ukp
    - non-linearity: relu for eng, arb, amh; gelu for ind
    - reduction_factor: 2 for eng, arb, ind; 16 for amh

    hyper_param:
    - pretrained-model: "bert-base-multilingual-cased"
    - Architecture: pfeiffer
    - head: False for eng, arb, ind; True for amh

    """
    config = AdapterConfig.load("pfeiffer", non_linearity=non_linearity, reduction_factor=reduction_factor)
    lang_adapter = model.load_adapter(lang, config=config, with_head=False)  # language adapters are loaded but not activated
    print(f"{lang} language adapter loaded.")
    return lang_adapter


# add task adapter
adapter_config = BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=16, non_linearity="relu")
task_adapter = bertmodel.add_adapter("STR", config=adapter_config)
print("task adapter added.")


# add language adapter
eng_adapter = set_lang_adapter(bertmodel, "en/wiki@ukp")
arb_adapter = set_lang_adapter(bertmodel, "ar/wiki@ukp")
amh_adapter = set_lang_adapter(bertmodel, "am/wiki@ukp", reduction_factor=16)
ind_adapter = set_lang_adapter(bertmodel, "id/wiki@ukp", non_linearity='gelu')


"""Encoding Functions"""

def encode_biencoder_batch(tidx, batch, tokenizer=berttokenizer):
    """Encodes a batch of input data using the model tokenizer. tidx= 0 if text1, tidx=1 if text2"""
    text = [pair[tidx] for pair in batch['pairs']]
    encode_text = tokenizer(text, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
    return encode_text


def get_biencoder_encoding(dataset):
    t1_dataset = dataset.map(lambda x: encode_biencoder_batch(0, x), batched=True)
    t2_dataset = dataset.map(lambda x: encode_biencoder_batch(1, x), batched=True)
    dataset = dataset.add_column('t1_input_ids', t1_dataset["input_ids"])
    dataset = dataset.add_column('t1_attention_mask', t1_dataset["attention_mask"])
    dataset = dataset.add_column('t2_input_ids', t2_dataset["input_ids"])
    dataset = dataset.add_column('t2_attention_mask', t2_dataset["attention_mask"])
    if "Score" in dataset.column_names:
        dataset = dataset.rename_column("Score", "labels")
        dataset.set_format(type="torch", columns=['PairID', 't1_input_ids', 't1_attention_mask', 't2_input_ids', 't2_attention_mask', "labels"])
    else:
        dataset.set_format(type="torch", columns=['PairID', 't1_input_ids', 't1_attention_mask', 't2_input_ids', 't2_attention_mask'])

    return dataset


def encode_crossencoder_batch(batch, tokenizer=berttokenizer):
    """Encodes a batch of input pairs using the model tokenizer for the cross encoder model."""
    pairs = batch["pairs"]
    encode_text = tokenizer(pairs, max_length=300, truncation=True, padding="max_length", return_tensors="pt")
    return encode_text


def get_crossencoder_encoding(dataset):
    encode_dataset = dataset.map(lambda x: encode_crossencoder_batch(x), batched=True)
    dataset = dataset.add_column('input_ids', encode_dataset["input_ids"])
    dataset = dataset.add_column('attention_mask', encode_dataset["attention_mask"])

    if "Score" in dataset.column_names:
        dataset = dataset.rename_column("Score", "labels")
        dataset.set_format(type="torch", columns=['PairID', 'input_ids', 'attention_mask', "labels"])
    else:
        dataset.set_format(type="torch", columns=['PairID', 'input_ids', 'attention_mask'])

    return dataset



if __name__ == "__main__":
    train_file_eng = '../data/Track A/eng/eng_train.csv'
    eng_test_data = load_data(train_file_eng)[:5]
    eng_test_dataset = Dataset.from_pandas(eng_test_data)
    eng_test_dataset = get_biencoder_encoding(eng_test_dataset)
    eng_input_ids1, eng_attention_mask1, eng_input_ids2, eng_attention_mask2, eng_labels = eng_test_dataset['t1_input_ids'], \
                                                                       eng_test_dataset['t1_attention_mask'], \
                                                                       eng_test_dataset['t2_input_ids'], \
                                                                       eng_test_dataset['t2_attention_mask'], \
                                                                       eng_test_dataset['labels']
    # print("Inputs shape: ",inputs.shape)
    eng_adapter = set_lang_adapter(bertmodel, "en/wiki@ukp")
    bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
    batch_output = bertmodel(eng_input_ids2, eng_attention_mask2, output_hidden_states=True)
    last_hidden_state_shape = batch_output.hidden_states[-1].size()
    print(batch_output)
    print("eng_last_hidden_size", last_hidden_state_shape)

    aim_file_ind = '../data/Track C/ind/ind_dev.csv'
    ind_data = load_data(aim_file_ind)[:5]
    ind_test_dataset = Dataset.from_pandas(ind_data)
    ind_test_dataset = get_biencoder_encoding(ind_test_dataset)
    ind_input_ids1, ind_attention_mask1, ind_input_ids2, ind_attention_mask2 = ind_test_dataset['t1_input_ids'], \
                                                                       ind_test_dataset['t1_attention_mask'], \
                                                                       ind_test_dataset['t2_input_ids'], \
                                                                       ind_test_dataset['t2_attention_mask']
    # print("Inputs shape: ",inputs.shape)
    id_adapter = set_lang_adapter(bertmodel, "id/wiki@ukp")
    bertmodel.set_active_adapters(ac.Stack(id_adapter, "STR"))
    batch_output = bertmodel(ind_input_ids2, ind_attention_mask2, output_hidden_states=True)
    last_hidden_state_shape = batch_output.hidden_states[-1].size()
    print(batch_output)
    print("ind_last_hidden_shape",  last_hidden_state_shape)

