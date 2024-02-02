from preprocessing import load_data, get_batches
from adapters_model import get_biencoder_encoding, get_crossencoder_encoding, arb_adapter, amh_adapter, ind_adapter
from datasets import Dataset
import torch
import adapters.composition as ac
from adapters_model import bertmodel
from model_CrossEncoder_Baseline import Baseline_CrossEncoderNN
from model_BiEncoder_Baseline import Baseline_BiEncodoerNN
from model_BiEncoder import BiEncoderNN
from model_CrossEncoder import CrossEncoderNN
from tqdm import tqdm
import warnings
import logging

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

"""run model on arb"""

"""load dataset"""
trackc_arb_dev = '../data/Track C/arb/arb_dev.csv'
arb_data = load_data(trackc_arb_dev)
arb_dataset = Dataset.from_pandas(arb_data[['PairID', "pairs"]])
print('Run fine-tuned BertAdapterModel on unlabeled language(arb) dataset')

"""BiEncoder_Baseline"""
# encoding
arb_biencoder_dataset = get_biencoder_encoding(arb_dataset)

# creat an CrossEncoderNN instance
biencoder_baseline_model = Baseline_BiEncodoerNN(bertmodel)

# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/biencoder_baseline_model.pt')
biencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False) # "strict=False" necessary, otherwise runtime error
biencoder_baseline_model.model.set_active_adapters(None)

# prediction and saving the result
baseline_scores, baseline_sample_ids = biencoder_baseline_model.predict(arb_biencoder_dataset, output_path='../result/arb/arb_biencoder_baseline.csv')
print("Run Fine-Tuned BiEncoder Baseline Model without Adapters on Arb: ")
print(baseline_scores, baseline_sample_ids)



"""CrossEncoder_Baseline"""
# encoding
arb_crossencoder_dataset = get_crossencoder_encoding(arb_dataset)

# creat an CrossEncoderNN instance
crossencoder_baseline_model = Baseline_CrossEncoderNN(bertmodel)
# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/crossencoder_baseline_model.pt')
crossencoder_baseline_model.load_state_dict(loaded_model_state_dict, strict=False) # "strict=False" necessary, otherwise runtime error
crossencoder_baseline_model.model.set_active_adapters(None)

# prediction and saving the result
baseline_scores, baseline_sample_ids = crossencoder_baseline_model.predict(arb_crossencoder_dataset, output_path='../result/arb/arb_crossencoder_baseline.csv')
print("Run Fine-Tuned CrossEncoder Baseline Model without Adapters on Arb: ")
print(baseline_scores, baseline_sample_ids)


"""BiEncoder"""
# creat an BiEncoderNN instance
biencoder_model = BiEncoderNN(bertmodel)

# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/biencoder_model.pt')
biencoder_model.load_state_dict(loaded_model_state_dict)
biencoder_model.model.set_active_adapters((ac.Stack(arb_adapter, "STR"))) #set active adapters

# prediction and saving the result
biencoder_scores, biencoder_sample_ids = biencoder_model.predict(arb_biencoder_dataset, output_path='../result/arb/arb_crossencoder.csv')
print("Run Fine-Tuned BiEncoder Model with Adapters on Arb: ")
print(biencoder_scores, biencoder_sample_ids)


"""CrossEncoder"""
# creat an CrossEncoderNN instance
crossencoder_model = CrossEncoderNN(bertmodel)
# load fine-tuned model
loaded_model_state_dict = torch.load('../Model/crossencoder_model.pt')
crossencoder_model.load_state_dict(loaded_model_state_dict)
crossencoder_model.model.set_active_adapters((ac.Stack(arb_adapter, "STR")))

# prediction and saving the result
crossencoder_scores, crossencoder_sample_ids = crossencoder_model.predict(arb_crossencoder_dataset, output_path='../result/arb/arb_biencoder.csv')
print("Run Fine-Tuned CrossEncoder Model with Adapters on Arb: ")
print(crossencoder_scores, crossencoder_sample_ids)


