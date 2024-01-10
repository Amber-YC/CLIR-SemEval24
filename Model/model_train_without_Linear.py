from sklearn.model_selection import train_test_split
from preprocessing import load_data, get_batches
import adapters_model
from adapters_model import berttokenizer, bertmodel, encode_batch, set_lang_adapter, get_pair_encoding
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
import sklearn
import wandb
from sklearn.metrics import mean_squared_error

"""load english training and eval data"""
tracka_eng = '../data/Track A/eng/eng_train.csv'
eng_data = load_data(tracka_eng)
eng_dataset = Dataset.from_pandas(eng_data[["pairs", "Score"]])

eng_dataset = get_pair_encoding(eng_dataset)
# print(eng_dataset)
eng_split = eng_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
# print(eng_split)

class BertNN(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.model = transformer_model
        self.model.train_adapter("STR")

        for name, param in self.model.named_parameters():
            if "STR" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.fc1 = nn.Linear(in_features=768, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.sigmoid = nn.Sigmoid()

        # # 3 convolutional layers with 2, 3, 5 kernals
        # self.conv1 = torch.nn.Conv1d(in_channels=768, out_channels=hidden_dim, kernel_size=2)
        # self.conv2 = torch.nn.Conv1d(in_channels=768, out_channels=hidden_dim, kernel_size=3)
        # self.conv3 = torch.nn.Conv1d(in_channels=768, out_channels=hidden_dim, kernel_size=5)
        # # maxpool layer
        # self.maxpool = torch.nn.AdaptiveMaxPool1d(1)


    def forward(self, input_ids1, attention_mask1,input_ids2, attention_mask2):
        # 获取两个句子的嵌入
        out1 = self.model(input_ids=input_ids1, attention_mask=attention_mask1).hidden_states[-1][:,0,:] # or [cls] embedding, or cnn + pool
        out2 = self.model(input_ids=input_ids2, attention_mask=attention_mask2).hidden_states[-1][:,0,:]
        # hidden_states[-1]表示最后一层的hidden，维度特征为[batch_size, sequence_length, hidden_size]
        # sequence_length[0]:semantic, sequence_length[-1]:generation
        # hidden_states[-1][:,0,:]==>tensor(2, 768)
        # print(out1.size())


        # # adjust the dimensions to go forward to cnn
        # out1 = out1.transpose(1, 2)  # [batch_size, reduced_size, sequence_length]
        # out2 = out2.transpose(1, 2)  # [batch_size, reduced_size, sequence_length]

        # def get_pooled_output(output):
        #     # output = output.permute(0, 2, 1)
        #
        #     conv1_out = F.relu(self.conv1(output))
        #     conv2_out = F.relu(self.conv2(output))
        #     conv3_out = F.relu(self.conv3(output))
        #
        #     pool1_out = self.maxpool(conv1_out).squeeze(2)
        #     pool2_out = self.maxpool(conv2_out).squeeze(2)
        #     pool3_out = self.maxpool(conv3_out).squeeze(2)
        #
        #     concat_out = torch.cat((pool1_out, pool2_out, pool3_out), dim=1)
        #     dropout = self.drop(concat_out)
        #     return dropout
        # out1 = get_pooled_output(out1)
        # out2 = get_pooled_output(out2)

        # f11 = self.fc1(F.relu(out1))
        # f12 = self.fc1(F.relu(out2))
        # f21 = self.fc2(F.relu(f11))
        # f22 = self.fc2(F.relu(f12))
        # print(f21.size(), f22.size())
        cos_sim = F.cosine_similarity(out1, out2)
        #print(cos_sim)
        similarity = (cos_sim + 1) / 2

        return similarity

eng_adapter = set_lang_adapter("en/wiki@ukp")
bertmodel.set_active_adapters(ac.Stack(eng_adapter, "STR"))
model = BertNN(transformer_model=bertmodel)

"""hyper params for training"""
lr = 0.001
batch_size = 2
epochs = 100
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=lr)
print(eng_split['train'])

"""train the model"""
def train_model(input_data, epochs=epochs, opt=opt):
    best_model = None
    best_dev_error = 0.0
    wandb.init(project = "CLIR - Semantic Relatedness Score")
    dev_input = input_data['test']
    dev_input_ids1 = dev_input['t1_input_ids']
    dev_attention_mask1 = dev_input['t1_attention_mask']
    dev_input_ids2 = dev_input['t2_input_ids']
    dev_attention_mask2 = dev_input['t2_attention_mask']
    dev_labels = dev_input['labels']

    for epoch in range(epochs):
        total_loss = 0.0
        train_input = get_batches(batch_size=batch_size, data=input_data['train'])
        batch_num = len(train_input)
        # print(train_input)
        for batch in tqdm(train_input, total=batch_num, desc="Training", ncols=80):
            # print(batch)
            opt.zero_grad()
            input_ids1 = batch['t1_input_ids']
            attention_mask1 = batch['t1_attention_mask']
            input_ids2 = batch['t2_input_ids']
            attention_mask2 = batch['t2_attention_mask']
            labels = batch['labels']
            outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            # print(f'output: {outputs}, label:{labels}')
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            total_loss += loss.item()

            dev_predictions = predict(model, dev_input_ids1, dev_attention_mask1, dev_input_ids2, dev_attention_mask2)
            dev_error = mean_squared_error(dev_labels, dev_predictions)

            if dev_error < best_dev_error:
                best_dev_error = dev_error
                best_model = model.state_dict()  # store the model with smallest squared mse on val data
            wandb.log({"loss": average_loss, "dev_error": dev_error, "best_dev_error": best_dev_error})

        wandb.finish()

        average_loss = total_loss / batch_num
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss}')

def predict(model, input_ids1, attention_mask1, input_ids2, attention_mask2):
    predictions = []
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        prediction = outputs
        predictions.extend(prediction.tolist())
    model.train()
    return predictions



# """load arb language data"""
# trackc_arb_dev = '../data/Track C/arb/arb_dev.csv'
# arb_data = load_data(trackc_arb_dev)
#
# """load amh language data"""
# trackc_amh_dev = '../data/Track C/amh/amh_dev.csv'
# amh_data = load_data(trackc_arb_dev)
#
# """load ind language data"""
# trackc_ind_dev = '../data/Track C/ind/ind_dev.csv'
# ind_data = load_data(trackc_ind_dev)


if __name__=="__main__":
    eng_tiny_dataset = eng_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
    print(eng_tiny_dataset)
    train_model(eng_tiny_dataset, epochs=epochs)