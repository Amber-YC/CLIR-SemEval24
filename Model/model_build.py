from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
import adapters_model
from adapters_model import berttokenizer, bertmodel, encode_biencoder_batch, set_lang_adapter, get_crossencoder_encoding, get_biencoder_encoding, eng_adapter
import numpy as np
import torch.nn as nn
import torch
import adapters.composition as ac
from adapters import AdapterTrainer, AdapterSetup
from transformers import TrainingArguments, EvalPrediction
from datasets import Dataset
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import math
import wandb
import pandas as pd
from scipy.stats import spearmanr
import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

"""load english training and eval data"""
eng_train_path = '../data/Track A/eng/eng_train.csv'
eng_test_path = '../data/Track A/eng/eng_dev.csv'

eng_training_data = load_data(eng_train_path)
eng_test_data = load_data(eng_test_path)

"""encoding for biencoder model"""
eng_biencoder_training_dataset = get_biencoder_encoding(Dataset.from_pandas(eng_training_data[["PairID", "pairs", "Score"]]))
eng_biencoder_test_dataset = get_biencoder_encoding(Dataset.from_pandas(eng_test_data[['PairID', "pairs"]]))

eng_biencoder_split = eng_biencoder_training_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)

"""encoding for crossencoder model"""
eng_crossencoder_training_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_training_data[["PairID", "pairs", "Score"]]))
eng_crossencoder_test_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_test_data[['PairID', "pairs"]]))

eng_crossencoder_split = eng_crossencoder_training_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)


"""BiencoderNN and Baseline_BiencoderNN"""
class BiEncoderNN(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids1, attention_mask1,input_ids2, attention_mask2):
        out1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1).hidden_states[-1][:, 0, :]
        out2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2).hidden_states[-1][:, 0, :]

        cos_similarity = F.cosine_similarity(out1, out2)
        sim = (cos_similarity + 1) / 2
        return sim

    def evaluate(self, eval_data, batch_size=20, loss_fn=nn.MSELoss()):
        self.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        test_input = get_batches(batch_size=batch_size, data=eval_data)
        batch_num = len(test_input)

        with torch.no_grad():
            for batch in tqdm(test_input, total=batch_num, desc="Evaluation", ncols=80):
                input_ids1 = batch['t1_input_ids']
                attention_mask1 = batch['t1_attention_mask']
                input_ids2 = batch['t2_input_ids']
                attention_mask2 = batch['t2_attention_mask']
                labels = batch['labels']
                outputs = self.forward(input_ids1, attention_mask1, input_ids2, attention_mask2)
                total_loss += loss_fn(outputs, labels)

                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / batch_num
        perplexity = math.exp(avg_loss)
        spearman_corr, _ = spearmanr(all_predictions, all_labels)
        return perplexity, avg_loss, spearman_corr

    def predict(self, test_data, output_path, batch_size=20):
        self.eval()
        test_input = get_batches(batch_size=batch_size, data=test_data)
        batch_num = len(test_input)
        scores = []
        sample_ids = []

        with torch.no_grad():
            for batch in tqdm(test_input, total=batch_num, desc="Prediction", ncols=80):
                input_ids1 = batch['t1_input_ids']
                attention_mask1 = batch['t1_attention_mask']
                input_ids2 = batch['t2_input_ids']
                attention_mask2 = batch['t2_attention_mask']

                outputs = self.forward(input_ids1, attention_mask1, input_ids2, attention_mask2)

                scores.extend(outputs.cpu().numpy())
                sample_ids.extend(batch['PairID'])

        # Create a DataFrame with PairID and Pred_Score columns
        result_df = pd.DataFrame({'PairID': sample_ids, 'Pred_Score': scores})
        # Save the DataFrame to a CSV file
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        result_df.to_csv(output_path, index=False)

        return scores, sample_ids

"""CrossEncoderNN and Baseline_CrossencoderNN"""
class CrossEncoderNN(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        self.classifier = nn.Linear(in_features=768, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1][:, 0, :]
        out = self.sigmoid(self.classifier(out)).squeeze()
        return out

    def evaluate(self, test_data, batch_size=20, loss_fn=nn.MSELoss()):
        self.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        test_input = get_batches(batch_size=batch_size, data=test_data)
        batch_num = len(test_input)

        with torch.no_grad():
            for batch in tqdm(test_input, total=batch_num, desc="Evaluation", ncols=80):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                outputs = self.forward(input_ids, attention_mask)
                total_loss += loss_fn(outputs, labels)

                all_predictions.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / batch_num
        perplexity = math.exp(avg_loss)

        spearman_corr, _ = spearmanr(all_predictions, all_labels)

        return perplexity, avg_loss, spearman_corr

    def predict(self, test_data, output_path, batch_size=20):
        self.eval()
        test_input = get_batches(batch_size=batch_size, data=test_data)
        batch_num = len(test_input)
        scores = []
        sample_ids = []

        with torch.no_grad():
            for batch in tqdm(test_input, total=batch_num, desc="Prediction", ncols=80):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']

                outputs = self.forward(input_ids, attention_mask)

                scores.extend(outputs.cpu().numpy())
                sample_ids.extend(batch['PairID'])

        # Create a DataFrame with PairID and Pred_Score columns
        result_df = pd.DataFrame({'PairID': sample_ids, 'Pred_Score': scores})

        # Save the DataFrame to a CSV file
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        result_df.to_csv(output_path, index=False)

        return scores, sample_ids

"""train model"""
def train_model(model, model_type, model_save_name, train_data, val_data, loss_fn, batch_size=16, epochs=10, opt=None):

    if model_type == "biencoder":
        best_model = BiEncoderNN(transformer_model=bertmodel)
    elif model_type == "crossencoder":
        best_model = CrossEncoderNN(transformer_model=bertmodel)

    best_epoch = 0
    best_validation_perplexity = 100000.

    for epoch in range(epochs):
        total_loss = 0.0
        train_input = get_batches(batch_size=batch_size, data=train_data)
        batch_num = len(train_input)
        for batch in tqdm(train_input, total=batch_num, desc="Training", ncols=80):
            opt.zero_grad()
            if model_type == "biencoder":
                input_ids1 = batch['t1_input_ids']
                attention_mask1 = batch['t1_attention_mask']
                input_ids2 = batch['t2_input_ids']
                attention_mask2 = batch['t2_attention_mask']
                labels = batch['labels']
                outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            elif model_type == "crossencoder":
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                outputs = model(input_ids, attention_mask)

            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item()
        average_loss = total_loss/batch_num

        # Evaluate and print accuracy at end of each epoch
        validation_perplexity, validation_loss, validation_spearman_corr = model.evaluate(val_data)

        # Log metrics to wandb
        wandb.log({"epoch": epoch+1,
                   "training average loss": average_loss,
                   "Validation loss": validation_loss,
                   "validation_spearman_corr": validation_spearman_corr})

        # remember best model:
        if validation_perplexity < best_validation_perplexity:
            print(f"new best model found!")
            best_epoch = epoch + 1
            best_validation_perplexity = validation_perplexity

            # always save best model
            torch.save(model.state_dict(), model_save_name)
        # print losses
        print(f"training loss: {average_loss}")
        print(f"validation loss: {validation_loss}")
        print(f"validation perplexity: {validation_perplexity}")
        print(f'validation_spearman_corr:{validation_spearman_corr}')

    wandb.finish()

    # load best model and do final test
    loaded_model_state_dict = torch.load(model_save_name)
    best_model.load_state_dict(loaded_model_state_dict)
    val_perplexity, val_loss, val_spear = best_model.evaluate(val_data)

    # print final score
    print("\n -- Training Done --")
    print(f" - using model from epoch {best_epoch} for final evaluation on validation dataset")
    print(f" - final score: perplexity={val_perplexity}, loss={val_loss}, spearman corr={val_spear}")

    return best_model

def build_and_train(model_name, mini=False, lr=0.01, batch_size=16, epochs=10):

    if mini == True:
        eng_biencoder_train = eng_biencoder_split['train'].select([i for i in range(100)])
        eng_biencoder_val = eng_biencoder_split['test'].select([i for i in range(10)])
        eng_biencoder_test = eng_biencoder_test_dataset.select([i for i in range(5)])

        eng_crossencoder_train = eng_crossencoder_split['train'].select([i for i in range(100)])
        eng_crossencoder_val = eng_crossencoder_split['test'].select([i for i in range(10)])
        eng_crossencoder_test = eng_crossencoder_test_dataset.select([i for i in range(5)])

    else:
        eng_biencoder_train = eng_biencoder_split['train']
        eng_biencoder_val = eng_biencoder_split['test']
        eng_biencoder_test = eng_biencoder_test_dataset

        eng_crossencoder_train = eng_biencoder_split['train']
        eng_crossencoder_val = eng_biencoder_split['test']
        eng_crossencoder_test = eng_biencoder_test_dataset

    """hyper params for training"""
    loss_fn = nn.MSELoss()

    if model_name == "baseline_biencoder":
        bertmodel.set_active_adapters(None)
        model = BiEncoderNN(transformer_model=bertmodel)
        model_type = "biencoder"
        model_save_name = 'biencoder_baseline_model.pt'
        project_name = "SemEval_BiEncoder_Baseline_eng"
        output_path = '../result/eng/eng_biencoder_baseline.csv'

        eng_train = eng_biencoder_train
        eng_val = eng_biencoder_val
        eng_test = eng_biencoder_test

    elif model_name == "baseline_crossencoder":
        bertmodel.set_active_adapters(None)
        model = CrossEncoderNN(transformer_model=bertmodel)
        model_type = "crossencoder"
        model_save_name = 'crossencoder_baseline_model.pt'
        project_name = "SemEval_CrossEncoder_Baseline_eng"
        output_path = '../result/eng/eng_crossencoder_baseline.csv'

        eng_train = eng_crossencoder_train
        eng_val = eng_crossencoder_val
        eng_test = eng_crossencoder_test

    elif model_name == "biencoder":
        # Set the adapters(la+ta) to be used in every forward pass
        bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
        # Freeze all model weights except of those of task adapter
        bertmodel.train_adapter(['STR'])
        model = BiEncoderNN(transformer_model=bertmodel)
        model_type = "biencoder"
        model_save_name = 'biencoder_model.pt'
        project_name = "SemEval_BiEncoder_eng"
        output_path = '../result/eng/eng_biencoder.csv'

        eng_train = eng_biencoder_train
        eng_val = eng_biencoder_val
        eng_test = eng_biencoder_test


    elif model_name == "crossencoder":
        # Set the adapters(la+ta) to be used in every forward pass
        bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
        # Freeze all model weights except of those of task adapter
        bertmodel.train_adapter(['STR'])

        model = CrossEncoderNN(transformer_model=bertmodel)
        model_type = "crossencoder"
        model_save_name = 'crossencoder_model.pt'
        project_name = "SemEval_CrossEncoder_eng"
        output_path = '../result/eng/eng_crossencoder.csv'

        eng_train = eng_crossencoder_train
        eng_val = eng_crossencoder_val
        eng_test = eng_crossencoder_test


    """train the model"""
    # Initialize wandb
    wandb.init(project=project_name)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # train model
    best_model = train_model(model, model_type, model_save_name, eng_train, eng_val, loss_fn, batch_size=batch_size, epochs=epochs, opt=opt)
    scores, sample_ids = best_model.predict(eng_test, output_path)
    print(f"MODEL NAME: {model_name}")
    print(f'scores:{scores}')
    print(f'sample_ids:{sample_ids}')



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Initiate and Train Models")
    # parser.add_argument("--model_name", type=str, required=True, choices=["baseline_biencoder", "baseline_crossencoder", "biencoder", "crossencoder"], help="the Name of the Model")
    # parser.add_argument("--mini", type=bool, required=False, choices=[True, False], help="Train on Mini Dataset or Large")
    # parser.add_argument("--mini", type=bool, required=False, choices=[True, False], help="Train on Mini Dataset or Large")
    # args = parser.parse_args()
    # main(args.dataset_path, args.mini)

    build_and_train("baseline_biencoder", mini=True, epochs=1)
    build_and_train("baseline_crossencoder", mini=True, epochs=1)
    build_and_train("biencoder", mini=True, epochs=1)
    build_and_train("crossencoder", mini=True, epochs=1)
