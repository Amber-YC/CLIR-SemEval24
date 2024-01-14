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

# we must keep the pretrained-transformer of the tokenizer and model the same
# Load tokenizer
berttokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")


# Load pre-trained BERT model from Hugging Face Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
bertmodel = BertAdapterModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)

def set_lang_adapter(lang: str, non_linearity="relu", reduction_factor=2):
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
    lang_adapter = bertmodel.load_adapter(lang, config=config, with_head=False) # language adapters are loaded but not activated
    print(f"{lang} language adapter loaded.")
    return lang_adapter


"""add task adapter"""
adapter_config = BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=16, non_linearity="relu")
task_adapter = bertmodel.add_adapter("STR", config=adapter_config)
# bertmodel.add_classification_head("STR", num_labels=3)
print("task adapter added.")




def encode_batch(tidx, batch, tokenizer=berttokenizer):
    """Encodes a batch of input data using the model tokenizer. tidx= 0 if text1, tidx=1 if text2"""
    text = [pair[tidx] for pair in batch['pairs']]
    encode_text = tokenizer(text, max_length=100, truncation=True, padding="max_length", return_tensors="pt")
    return encode_text


def get_pair_encoding(dataset):

    t1_dataset = dataset.map(lambda x: encode_batch(0, x), batched=True)
    t2_dataset = dataset.map(lambda x: encode_batch(1, x), batched=True)
    dataset = dataset.add_column('t1_input_ids', t1_dataset["input_ids"])
    dataset = dataset.add_column('t1_attention_mask', t1_dataset["attention_mask"])
    # dataset = dataset.add_column('t1_token_type_ids', t1_dataset["token_type_ids"])
    dataset = dataset.add_column('t2_input_ids', t2_dataset["input_ids"])
    dataset = dataset.add_column('t2_attention_mask', t2_dataset["attention_mask"])
    # dataset = dataset.add_column('t2_token_type_ids', t2_dataset["token_type_ids"])
    if "Score" in dataset.column_names:
        dataset = dataset.rename_column("Score", "labels")
        dataset.set_format(type="torch", columns=['t1_input_ids', 't1_attention_mask', 't2_input_ids', 't2_attention_mask', "labels"])
    else:
        dataset.set_format(type="torch", columns=['t1_input_ids', 't1_attention_mask', 't2_input_ids', 't2_attention_mask'])

    return dataset



if __name__ == "__main__":
    train_file_eng = '../data/Track A/eng/eng_train.csv'
    eng_test_data = load_data(train_file_eng)[:5]
    eng_test_dataset = Dataset.from_pandas(eng_test_data)
    eng_test_dataset = get_pair_encoding(eng_test_dataset)
    eng_input_ids1, eng_attention_mask1, eng_input_ids2, eng_attention_mask2, eng_labels = eng_test_dataset['t1_input_ids'], \
                                                                       eng_test_dataset['t1_attention_mask'], \
                                                                       eng_test_dataset['t2_input_ids'], \
                                                                       eng_test_dataset['t2_attention_mask'], \
                                                                       eng_test_dataset['labels']
    # print("Inputs shape: ",inputs.shape)
    eng_adapter = set_lang_adapter("en/wiki@ukp")
    bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
    batch_output = bertmodel(eng_input_ids2, eng_attention_mask2, output_hidden_states=True)
    last_hidden_state_shape = batch_output.hidden_states[-1].size()
    print(batch_output)
    print("eng_last_hidden_size", last_hidden_state_shape)


    aim_file_ind = '../data/Track C/ind/ind_dev.csv'
    ind_data = load_data(aim_file_ind)[:5]
    ind_test_dataset = Dataset.from_pandas(ind_data)
    ind_test_dataset = get_pair_encoding(ind_test_dataset)
    ind_input_ids1, ind_attention_mask1, ind_input_ids2, ind_attention_mask2 = ind_test_dataset['t1_input_ids'], \
                                                                       ind_test_dataset['t1_attention_mask'], \
                                                                       ind_test_dataset['t2_input_ids'], \
                                                                       ind_test_dataset['t2_attention_mask']
    # print("Inputs shape: ",inputs.shape)
    id_adapter = set_lang_adapter("id/wiki@ukp")
    bertmodel.set_active_adapters(ac.Stack(id_adapter, "STR"))
    batch_output = bertmodel(ind_input_ids2, ind_attention_mask2, output_hidden_states=True)
    last_hidden_state_shape = batch_output.hidden_states[-1].size()
    print(batch_output)
    print("ind_last_hidden_shape",  last_hidden_state_shape)

