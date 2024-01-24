from preprocessing import load_data, get_batches
from adapters_model import get_biencoder_encoding, get_crossencoder_encoding, arb_adapter, amh_adapter, ind_adapter
from datasets import Dataset
import torch
import adapters.composition as ac
from adapters_model import bertmodel
from model_Baseline import BaselineNN
from model_BiEncoder import BiEncoderNN
from model_CrossEncoder import CrossEncoderNN
from tqdm import tqdm


"""run model on arb"""

"""load dataset"""
trackc_arb_dev = '../data/Track C/arb/arb_dev.csv'
arb_data = load_data(trackc_arb_dev)
arb_dataset = Dataset.from_pandas(arb_data[['PairID', "pairs"]])
print('Run fine-tuned BertAdapterModel on unlabeled language(arb) dataset')


"""BiEncoder"""
# encoding
arb_biencoder_dataset = get_biencoder_encoding(arb_dataset)

# creat an CrossEncoderNN instance
biencoder_model = BiEncoderNN(bertmodel)
# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/biencoder_model.pt')
biencoder_model.load_state_dict(loaded_model_state_dict)

# prediction and saving the result
biencoder_scores, biencoder_sample_ids = biencoder_model.predict(arb_biencoder_dataset)
print(biencoder_scores, biencoder_sample_ids)


"""CrossEncoder"""
# encoding
arb_crossencoder_dataset = get_crossencoder_encoding(arb_dataset)

# creat an CrossEncoderNN instance
crossencoder_model = CrossEncoderNN(bertmodel)
# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/crossencoder_model.pt')
crossencoder_model.load_state_dict(loaded_model_state_dict)

# prediction and saving the result
crossencoder_scores, crossencoder_sample_ids = crossencoder_model.predict(arb_crossencoder_dataset)
print(crossencoder_scores, crossencoder_sample_ids)


"""Baseline"""
# encoding
arb_baseline_dataset = get_crossencoder_encoding(arb_dataset)

# creat an CrossEncoderNN instance
baseline_model = CrossEncoderNN(bertmodel)
# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/baseline_model.pt')
baseline_model.load_state_dict(loaded_model_state_dict)

# prediction and saving the result
baseline_scores, baseline_sample_ids = baseline_model.predict(arb_crossencoder_dataset)
print(baseline_scores, baseline_sample_ids)