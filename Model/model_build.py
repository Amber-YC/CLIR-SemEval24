import argparse

from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
import model_adapters
from model_adapters import berttokenizer, bertmodel, encode_biencoder_batch, set_lang_adapter, get_crossencoder_encoding, get_biencoder_encoding, eng_adapter
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

# license

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

"""Data Preprocessing"""

# load english training and validation data
eng_train_path = '../data/Track A/eng/eng_train.csv'
eng_test_path = '../data/Track A/eng/eng_dev.csv'
eng_training_data = load_data(eng_train_path)
eng_test_data = load_data(eng_test_path)

# encoding for biencoder model
eng_biencoder_training_dataset = get_biencoder_encoding(Dataset.from_pandas(eng_training_data[["PairID", "pairs", "Score"]]))
eng_biencoder_test_dataset = get_biencoder_encoding(Dataset.from_pandas(eng_test_data[['PairID', "pairs"]]))
# divide training dataset into training and validation dataset
eng_biencoder_split = eng_biencoder_training_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)

# encoding for crossencoder model
eng_crossencoder_training_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_training_data[["PairID", "pairs", "Score"]]))
eng_crossencoder_test_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_test_data[['PairID', "pairs"]]))
# divide training dataset into training and validation dataset
eng_crossencoder_split = eng_crossencoder_training_dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)


class BiEncoderNN(nn.Module):
    """
    Neural Network class for Bi-Encoder model.

    Args:
        transformer_model (nn.Module): Transformer model used in the bi-encoder.

    Methods:
        forward(input_ids1, attention_mask1, input_ids2, attention_mask2):
            Forward pass of the bi-encoder model, computing similarity between two sets of inputs.

        evaluate(eval_data, batch_size=32, loss_fn=nn.MSELoss()):
            Evaluate the model on the provided evaluation data, computing perplexity, average loss, and Spearman correlation.

        predict(test_data, output_path, batch_size=32):
            Make predictions on the given test data and save results to a CSV file.
    """
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model

    def forward(self, input_ids1, attention_mask1,input_ids2, attention_mask2):
        out1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1).hidden_states[-1][:, 0, :]
        out2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2).hidden_states[-1][:, 0, :]

        cos_similarity = F.cosine_similarity(out1, out2)
        sim = (cos_similarity + 1) / 2  # transformation maps the cosine similarity range from [-1, 1] to [0, 1]
        return sim

    def evaluate(self, eval_data, batch_size=32, loss_fn=nn.MSELoss()):
        self.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        test_input = get_batches(batch_size=batch_size, data=eval_data)
        batch_num = len(test_input)

        with torch.no_grad():  # to avoid unnecessary memory consumption and computational
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

    def predict(self, test_data, output_path, batch_size=32):
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

        # create a DataFrame with PairID and Pred_Score columns
        result_df = pd.DataFrame({'pairid': sample_ids, 'pred_score': scores})
        # save the DataFrame to a CSV file
        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        result_df.to_csv(output_path, index=False)

        return scores, sample_ids


class CrossEncoderNN(nn.Module):
    """
    Neural Network class for Cross-Encoder model.

    Args:
        transformer_model (nn.Module): Transformer model used in the cross-encoder.

    Methods:
        forward(input_ids, attention_mask):
            Forward pass of the cross-encoder model, applying a linear classifier with sigmoid activation.

        evaluate(test_data, batch_size=32, loss_fn=nn.MSELoss()):
            Evaluate the model on the provided test data, computing perplexity, average loss, and Spearman correlation.

        predict(test_data, output_path, batch_size=32):
            Make predictions on the given test data and save results to a CSV file.
    """
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        self.classifier = nn.Linear(in_features=768, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1][:, 0, :]
        out = self.sigmoid(self.classifier(out)).squeeze()
        return out

    def evaluate(self, test_data, batch_size=32, loss_fn=nn.MSELoss()):
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

    def predict(self, test_data, output_path, batch_size=32):
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

        result_df = pd.DataFrame({'pairid': sample_ids, 'pred_score': scores})

        directory = os.path.dirname(output_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        result_df.to_csv(output_path, index=False)

        return scores, sample_ids


"""Training"""


def train_model(model, model_type, model_save_name, train_data, val_data, loss_fn, batch_size=32, epochs=10, opt=None):

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

        # evaluate and print accuracy at end of each epoch
        validation_perplexity, validation_loss, validation_spearman_corr = model.evaluate(val_data)

        # log metrics to wandb
        wandb.log({"epoch": epoch+1,
                   "training average loss": average_loss,
                   "validation loss": validation_loss,
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


def build_and_train(model_name, mini=False, lr=0.001, batch_size=32, epochs=10):
    # train on mini or large dataset
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

        eng_crossencoder_train = eng_crossencoder_split['train']
        eng_crossencoder_val = eng_crossencoder_split['test']
        eng_crossencoder_test = eng_crossencoder_test_dataset

    # train with which model
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
        # set the adapters(la+ta) to be used in every forward pass
        bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
        # freeze all model weights except of those of task adapter
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
        # set the adapters(la+ta) to be used in every forward pass
        bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
        # freeze all model weights except of those of task adapter
        bertmodel.train_adapter(['STR'])

        model = CrossEncoderNN(transformer_model=bertmodel)
        model_type = "crossencoder"
        model_save_name = 'crossencoder_model.pt'
        project_name = "SemEval_CrossEncoder_eng"
        output_path = '../result/eng/eng_crossencoder.csv'

        eng_train = eng_crossencoder_train
        eng_val = eng_crossencoder_val
        eng_test = eng_crossencoder_test

    # initialize wandb
    wandb.init(project=project_name)

    # hyperparameters
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # train and get best model
    best_model = train_model(model, model_type, model_save_name, eng_train, eng_val, loss_fn, batch_size=batch_size, epochs=epochs, opt=opt)
    # predict on the best model
    scores, sample_ids = best_model.predict(eng_test, output_path)
    print(f"MODEL NAME: {model_name}")
    print(f'scores:{scores}')
    print(f'sample_ids:{sample_ids}')


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Initiate and Train Models")
    # parser.add_argument("--model_name", "-mn", type=str, required=True,
    #                     choices=["baseline_biencoder", "baseline_crossencoder", "biencoder", "crossencoder"],
    #                     help="the Name of the Model")
    # parser.add_argument("--mini", "-mi", type=bool, required=True,
    #                     choices=[True, False],
    #                     help="Train on Mini Dataset or Large")
    # parser.add_argument("--epochs", "-e", type=int, required=True,
    #                     help="Number of training epochs")
    # parser.add_argument("--batch_size", "-b", type=int, required=True,
    #                     help="Batch size for training")
    # args = parser.parse_args()
    #
    # build_and_train(args.model_name, args.mini, args.epochs, args.batch_size)

    """To execute this script in the terminal, use the following command as an example:"""
    # Example: python Model/model_build.py -mn baseline_biencoder -mi True -e 3 -b 32
    # not done yet, need to change path before run it in terminal

    # test on mini dataset
    # build_and_train("baseline_biencoder", mini=True, epochs=10)
    # build_and_train("baseline_crossencoder", mini=True, epochs=10)
    # build_and_train("biencoder", mini=True, epochs=10)
    # build_and_train("crossencoder", mini=True, epochs=10)

    # train on large dataset
    build_and_train("baseline_biencoder")
    build_and_train("baseline_crossencoder")
    build_and_train("biencoder")
    build_and_train("crossencoder")

    """Our training results for each model on a large dataset can be found at:"""
    # "baseline_biencoder"
    # https://wandb.ai/lvertiefungcl2324_xinaohan/SemEval_BiEncoder_Baseline_eng/runs/105yhxfa?workspace=user-hanxinao

    # "baseline_crossencoder"
    # https://wandb.ai/lvertiefungcl2324_xinaohan/SemEval_CrossEncoder_Baseline_eng/runs/kq0f7l6r?workspace=user-hanxinao

    # "biencoder"
    # https://wandb.ai/lvertiefungcl2324_xinaohan/SemEval_BiEncoder_eng/runs/nwt89r1l?workspace=user-hanxinao

    # "crossencoder"
    # https://wandb.ai/lvertiefungcl2324_xinaohan/SemEval_CrossEncoder_eng/runs/t4golst5?workspace=user-hanxinao