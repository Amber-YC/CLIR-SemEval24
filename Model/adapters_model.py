from transformers import AutoTokenizer, BertTokenizer
from preprocessing import load_data
from adapters import AutoAdapterModel, AdapterConfig, AdapterSetup, BnConfig
import numpy as np
import torch.nn as nn
import torch
import adapters.composition as ac
# from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
# from sentence_transformers import BiEncoder

# we must keep the pretrained-transformer of the tokenizer and model the same
# Load tokenizer
berttokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def encode_batch(batch, tokenizer=berttokenizer):
    """Encodes a batch of input data using the model tokenizer."""
    text1_ls = batch['Text1'].tolist()
    text2_ls = batch['Text2'].tolist()
    tokenized_batch = tokenizer(text1_ls, text1_ls, padding=True, return_tensors="pt")
    return tokenized_batch



# Load pre-trained BERT model from Hugging Face Hub
# The `BertAdapterModel` class is specifically designed for working with adapters
model = AutoAdapterModel.from_pretrained("bert-base-multilingual-cased")


# add language layer
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
    lang_adapter = model.load_adapter(lang, config=config, with_head=False) # language adapters are loaded but not activated
    print(f"{lang} language adapter loaded.")
    return lang_adapter


# add task adapter
"""use single bottleneck adapter configuration"""
adapter_config = BnConfig(mh_adapter=False, output_adapter=True, reduction_factor=16, non_linearity="relu")
task_adapter = model.add_adapter("STR", config=adapter_config)
model.add_masked_lm_head("STR", activation_function="relu")
model.train_adapter("STR")
print("task adapter added.")

def get_word_embeddings(inputs, lang_adapter):
    # stack and activate the certain language adapter and task adapter
    with AdapterSetup(ac.Stack(lang_adapter, "STR")):
        outputs = model(**inputs)
    return outputs
# 冻结la？如何train？
# 是否可以使用 biencoder？


if __name__ == "__main__":
    train_file_eng = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
    eng_data = load_data(train_file_eng)
    eng_tokens = encode_batch(eng_data[:5])
    print(eng_tokens['input_ids'])
    print(eng_tokens['attention_mask'])
    eng_adapter = set_lang_adapter("en/wiki@ukp")
    # model.set_active_adapters(eng_adapter)
    eng_output = get_word_embeddings(eng_tokens, eng_adapter)
    print("eng_output:", eng_output)


    aim_file_ind = '../Semantic_Relatedness_SemEval2024-main/Track C/ind/ind_dev.csv'
    ind_data = load_data(aim_file_ind)
    ind_tokens = encode_batch(ind_data[:5])
    ind_adapter = set_lang_adapter("id/wiki@ukp")
    # model.set_active_adapters(ind_adapter) # 只激活这一个
    ind_output = get_word_embeddings(ind_tokens, ind_adapter)
    print("ind_output:", ind_output)

