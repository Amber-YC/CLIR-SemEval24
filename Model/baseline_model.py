import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
import adapters_model
from adapters_model import berttokenizer, bertmodel, encode_crossencoder_batch, set_lang_adapter, get_crossencoder_encoding, eng_adapter
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
from scipy.stats import spearmanr


"""load english dataset"""
eng_train_path = '../data/Track A/eng/eng_train.csv'
eng_val_path = '../data/Track A/eng/eng_dev_with_labels.csv'
eng_test_path = '../data/Track A/eng/eng_test.csv'

eng_training_data = load_data(eng_train_path)
eng_validation_data = load_data(eng_val_path)
eng_test_data = load_data(eng_test_path)

eng_training_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_training_data[["PairID", "pairs", "Score"]]))
eng_validation_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_validation_data[["PairID", "pairs", "Score"]]))
eng_test_dataset = get_crossencoder_encoding(Dataset.from_pandas(eng_test_data[['PairID', "pairs"]]))

class CrossEncoderNN(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        #self.model.train_adapter("STR")  # Freeze all model weights except of those of this adapter
        self.classifier = nn.Linear(in_features=768, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # sentence embeddings
        out = self.model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1][:,0,:]# or [cls] embedding, or cnn + pool
        lin = self.classifier(out)

        similarity = self.sigmoid(lin).squeeze()

        return similarity

    def evaluate(self, test_data):
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

    def predict(self, test_data, output_path='../result/eng/eng_crossencoder_res.csv'):
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
        result_df.to_csv(output_path, index=False)

        return scores, sample_ids


def train_model(model, train_data, val_data, epochs=10, opt=None):
    model_save_name = 'crossencoder_model.pt'

    best_model = CrossEncoderNN(transformer_model=bertmodel)
    best_epoch = 0
    best_validation_perplexity = 100000.

    for epoch in range(epochs):
        total_loss = 0.0
        train_input = get_batches(batch_size=batch_size, data=train_data)
        batch_num = len(train_input)
        for batch in tqdm(train_input, total=batch_num, desc="Training", ncols=80):
            opt.zero_grad()
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

    # load best model and do final test
    loaded_model_state_dict = torch.load(model_save_name)
    best_model.load_state_dict(loaded_model_state_dict)
    val_perplexity, val_loss, val_spear = best_model.evaluate(val_data)

    # print final score
    print("\n -- Training Done --")
    print(f" - using model from epoch {best_epoch} for final evaluation on validation dataset")
    print(f" - final score: perplexity={val_perplexity}, loss={val_loss}, spearman corr={val_spear}")

    return best_model


if __name__ == "__main__":
    """build the model"""
    baseline_model = CrossEncoderNN(transformer_model=bertmodel)

    """hyper params for training"""
    lr = 0.001
    batch_size = 16
    epochs = 2
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(baseline_model.parameters(), lr=lr)

    """train the model"""
    # Initialize wandb
    wandb.init(project="SemEval_Baseline_eng")

    # create mini dataset
    # tqdm only support DataSet, use '.select' to form mini dataset instead of slices
    eng_training_dataset_mini = eng_training_dataset.select([i for i in range(50)])
    eng_validation_dataset_mini = eng_validation_dataset.select([i for i in range(5)])
    eng_test_dataset_mini = eng_test_dataset.select([i for i in range(5)])

    """train STR task adapter on labeled english dataset"""
    best_model = train_model(baseline_model, eng_training_dataset_mini, eng_validation_dataset_mini, epochs=epochs,
                             opt=opt)
    """predict on eng_test_dataset"""
    scores, sample_ids = best_model.predict(eng_test_dataset_mini)
    print(f'scores:{scores}')
    print(f'sample_ids:{sample_ids}')
