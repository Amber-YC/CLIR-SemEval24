from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
import adapters_model
from adapters_model import berttokenizer, bertmodel, encode_biencoder_batch, set_lang_adapter, get_biencoder_encoding
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

"""load english training and eval data"""
tracka_eng = '../data/Track A/eng/eng_train.csv'
eng_data = load_data(tracka_eng)
eng_dataset = Dataset.from_pandas(eng_data[["pairs", "Score"]])

eng_dataset = get_biencoder_encoding(eng_dataset)
# print(eng_dataset)
eng_split = eng_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
# print(eng_split)

class BiEncoderNN(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        self.model.train_adapter("STR") # Freeze all model weights except of those of this adapter

        #for name, param in self.model.named_parameters():
        #    if "STR" in name:
        #    #if "STR" in name or "en/wiki@ukp" in name:
        #        param.requires_grad = True
        #    else:
        #        param.requires_grad = False
        #self.fc1 = nn.Linear(in_features=768, out_features=128)
        # relu
        #self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.sigmoid = nn.Sigmoid()


    def forward(self, input_ids1, attention_mask1,input_ids2, attention_mask2):
        # sentence embeddings
        out1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1).hidden_states[-1][:, 0, :] # or [cls] embedding, or cnn + pool
        out2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2).hidden_states[-1][:, 0, :]
        #lin1 = self.fc2(self.fc1(out1))
        #lin2 = self.fc2(self.fc1(out2))
        #cos = F.cosine_similarity(lin1, lin2)
        cos = F.cosine_similarity(out1, out2)
        similarity = (cos+1) / 2
        #print(f'cos: {cos}')
        #print(f'sig: {similarity}')
        return similarity

    def evaluate(self, test_data):
        self.eval()
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

        avg_loss = total_loss / batch_num
        perplexity = math.exp(avg_loss)

        return perplexity, avg_loss


eng_adapter = set_lang_adapter("en/wiki@ukp")
bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR")) # Set the adapters(la+ta) to be used in every forward pass
model = BiEncoderNN(transformer_model=bertmodel)

"""hyper params for training"""
lr = 0.01
batch_size = 16
epochs = 3
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)
print(eng_split['train'])

"""train the model"""
# Initialize wandb
wandb.init(project="SemEval_BiEncoder_eng")
def train_model(train_data, test_data, epochs=epochs, opt=opt):
    model_save_name = 'eng-biencoder-model.pt'

    #best_model = None
    best_model = BiEncoderNN(transformer_model=bertmodel)
    best_epoch = 0
    best_validation_perplexity = 100000.

    for epoch in range(epochs):
        total_loss = 0.0
        train_input = get_batches(batch_size=batch_size, data=train_data)
        batch_num = len(train_input)
        for batch in tqdm(train_input, total=batch_num, desc="Training", ncols=80):
            opt.zero_grad()
            input_ids1 = batch['t1_input_ids']
            attention_mask1 = batch['t1_attention_mask']
            input_ids2 = batch['t2_input_ids']
            attention_mask2 = batch['t2_attention_mask']
            labels = batch['labels']
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            #print(f'output_sig: {outputs}, label:{labels}')
            loss = loss_fn(outputs, labels)
            #print(f'loss: {loss.item()}')
            loss.backward()
            opt.step()

            total_loss += loss.item()
        average_loss = total_loss/batch_num


        # Evaluate and print accuracy at end of each epoch
        validation_perplexity, validation_loss = model.evaluate(test_data)

        # Log metrics to wandb
        wandb.log({"epoch": epoch + 1, "training average loss": average_loss, "Validation loss": validation_loss})

        # remember best model:
        if validation_perplexity < best_validation_perplexity:
            print(f"new best model found!")
            best_epoch = epoch + 1
            best_validation_perplexity = validation_perplexity

            # always save best model
            torch.save(model.state_dict(), model_save_name)
            #torch.save(model, model_save_name)
        # print losses
        print(f"training loss: {average_loss}")
        print(f"validation loss: {validation_loss}")
        print(f"validation perplexity: {validation_perplexity}")

    # load best model and do final test
    loaded_model_state_dict = torch.load(model_save_name)
    best_model.load_state_dict(loaded_model_state_dict)

    test_perplexity, test_loss = best_model.evaluate(test_data)

    #best_model = torch.load(model_save_name)
    #test_perplexity, test_loss = best_model.evaluate(test_data)

    # print final score
    print("\n -- Training Done --")
    print(f" - using model from epoch {best_epoch} for final evaluation")
    #print(f" - final score: {test_accuracy}")
    print(f" - final score: perplexity={test_perplexity}, validation loss={test_loss}")


# """load arb language data"""
# trackc_arb_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/arb/arb_dev.csv'
# arb_data = load_data(trackc_arb_dev)
#
# """load amh language data"""
# trackc_amh_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/amh/amh_dev.csv'
# amh_data = load_data(trackc_arb_dev)
#
# """load ind language data"""
# trackc_ind_dev = '../Semantic_Relatedness_SemEval2024-main/Track C/ind/ind_dev.csv'
# ind_data = load_data(trackc_ind_dev)


if __name__=="__main__":
    # remember the best model

    eng_train = eng_split['train'].select([i for i in range(100)])
    eng_val = eng_split['test'].select([i for i in range(10)])

    #eng_train = eng_split['train']
    #eng_val = eng_split['test']
    #print(eng_train[:5])
    train_model(eng_train, eng_val, epochs=epochs)

