from sklearn.metrics import r2_score
from transformers import AutoTokenizer, AutoModel, TrainingArguments, EvalPrediction
from preprocessing import load_train_data, load_eval_data
from sklearn.model_selection import train_test_split
import sklearn
from adapters import AutoAdapterModel, AdapterConfig, AdapterType, AdapterTrainer
import numpy as np
from datasets import Dataset


# Pre-trained transformer
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  encoded_text = tokenizer(batch, max_length=80, truncation=True, padding="max_length", return_tensors="pt")
  return encoded_text

train_file_eng = '../Semantic_Relatedness_SemEval2024-main/Track A/eng/eng_train.csv'
train_pairs_eng, train_scores_eng = load_train_data(train_file_eng)
texts_train, texts_val, labels_train, labels_val = train_test_split(train_pairs_eng, train_scores_eng,
                                                                        test_size=0.2, random_state=42)
dataset = {"train": {}, "validation": {}}
for idx, pair in enumerate(texts_train):
    dataset["train"][idx] = {"text0": encode_batch(pair[0]), "text1": encode_batch(pair[1]), "label": labels_train[idx]}
for idx, pair in enumerate(texts_val):
    dataset["validation"][idx] = {"text0": encode_batch(pair[0]), "text1": encode_batch(pair[1]), "label": labels_val[idx]}

dataset_train = Dataset.from_dict(dataset["train"])
dataset_val = Dataset.from_dict(dataset["validation"])

print(dataset_train)