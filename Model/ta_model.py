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

# prepare dataset
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


# Create a Hugging Face Dataset object from the dictionary
dataset = Dataset.from_dict(dataset)

model = AutoAdapterModel.from_pretrained("bert-base-multilingual-cased")

# Add la layer
config = AdapterConfig.load("pfeiffer", non_linearity="relu", reduction_factor=2)
la_model = model.load_adapter("en/wiki@ukp", config=config)
model.set_active_adapters(la_model)

# Add ta adapter
task_name = "semantic_textual_relatedness"
model.add_adapter(task_name, config="seq_bn")
# Set up the regression head for the task adapter
model.add_regression_head(task_name)
# Activate the adapter
model.train_adapter(task_name)




training_args = TrainingArguments(
    learning_rate=1e-4,
    num_train_epochs=6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    logging_steps=200,
    output_dir="./training_output",
    overwrite_output_dir=True,
    # The next line is important to ensure the dataset labels are properly passed to the model
    remove_unused_columns=False,
)


def compute_r_squared(p: EvalPrediction):
    preds = p.predictions.flatten()
    labels = p.label_ids

    r_squared = r2_score(labels, preds)
    return {"r_squared": r_squared}


trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    compute_metrics=compute_r_squared(),
)