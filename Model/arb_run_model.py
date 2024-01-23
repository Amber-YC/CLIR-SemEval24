from preprocessing import load_data, get_batches
from adapters_model import get_biencoder_encoding, get_crossencoder_encoding, arb_adapter, amh_adapter, ind_adapter
from datasets import Dataset
import torch
import adapters.composition as ac
from adapters_model import bertmodel
from model_Baseline import BaselineNN
from model_BiEncoder import BiEncoderNN
from model_CrossEncoder import CrossEncoderNN


"""run model on arb"""
trackc_arb_dev = '../data/Track C/arb/arb_dev.csv'
arb_data = load_data(trackc_arb_dev)
arb_dataset = Dataset.from_pandas(arb_data[['PairID', "pairs"]])



"""Biencoder Model"""
arb_biencoder_dataset = get_biencoder_encoding(arb_dataset)
biencoder_bert = bertmodel.load_state_dict(torch.load("biencoder_model"))
biencoder_bert.set_active_adapters(ac.Stack(arb_adapter, "STR")) # 儲存的是哪個model？load是哪個model的數據？
biencoder_model = BiEncoderNN(biencoder_bert)
biencoder_scores, biencoder_sample_ids = biencoder_model.predict(arb_biencoder_dataset)
print(biencoder_scores, biencoder_sample_ids)

"""CrossEncoder"""
arb_crossencoder_dataset = get_crossencoder_encoding(arb_dataset)
crossencoder_bert = bertmodel.load_state_dict(torch.load("crossencoder_model"))
crossencoder_bert.set_active_adapters(ac.Stack(arb_adapter, "STR")) # 儲存的是哪個model？load是哪個model的數據？
crossencoder_model = CrossEncoderNN(crossencoder_bert)
crossencoder_scores, crossencoder_sample_ids = crossencoder_model.predict(arb_crossencoder_dataset)
print(biencoder_scores, biencoder_sample_ids)


"""Baseline"""
arb_baseline_dataset = get_crossencoder_encoding(arb_dataset)
baseline_bert = bertmodel.load_state_dict(torch.load("baseline_model.pt"))
baseline_model = BaselineNN(baseline_bert)
baseline_scores, baseline_sample_ids = baseline_model.predict(arb_baseline_dataset)

print(baseline_scores, baseline_sample_ids)

