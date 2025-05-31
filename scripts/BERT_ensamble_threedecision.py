import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import torch
from tqdm import tqdm, trange
from torch import nn
import warnings
from transformers import DistilBertTokenizer, DistilBertModel, XLNetTokenizer, XLNetModel, RobertaTokenizer, RobertaModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import re
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
def parse_time_to_seconds(time_str):
    try:
        # 正規表現を使用して日数と時間部分を抽出
        match = re.match(r'(\d+)\s*days?\s*(\d{1,2}):?(\d{2})?:?(\d{2})?', time_str)
        if match:
            days, hours, minutes, seconds = match.groups()
            days = int(days)
            hours = int(hours) if hours else 0
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            
            total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
            return float(total_seconds)
        else:
            return float(-1)
    except:
      return float(-1)

class MultiModel(nn.Module):
    def __init__(self, model, hidden_size, num_features, num_labels):
        super(MultiModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.3)

        # feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, num_features*4),
            nn.LayerNorm(num_features*4),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(num_features*4, num_features),
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size+num_features, hidden_size+num_features),
            nn.LayerNorm(hidden_size+num_features),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size+num_features, num_labels),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, input_features):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        dropout_output = self.dropout(cls_output)

        # feature encoder
        features_output = self.feature_encoder(input_features)
        
        final_output = self.classifier(torch.cat([dropout_output, features_output], dim=1))
        return final_output

def preprocess_data(df, tokenizer, text_columns):
    inputs = tokenizer(
        df[text_columns[0]].tolist(), 
        df[text_columns[1]].tolist(), 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    return inputs

def get_dataloader(input_ids, attention_mask, features, labels, batch_size, shuffle=True):
    train_data = TensorDataset(input_ids, attention_mask, features, labels)
    if (shuffle):
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_model(model, train_dataloader, validation_dataloader, loss_weights, epoch=10):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(loss_weights)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
        print("Training on GPU")
    else:
        device = torch.device("cpu")
        print("Training on CPU")
    
    all_predictions = []

    for _ in trange(epoch, desc="Epoch"):

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, input_features, labels = batch
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, input_masks, input_features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            # tracking variablesを更新
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            # Clear cache
            torch.cuda.empty_cache()

        print("Train loss: {}".format(tr_loss/nb_tr_steps))

        # Validation

        epoch_predictions = []

        # モデルをevaluationモードにする
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, input_features, labels = batch

            # gradientsを計算または保存しないようにモデルに指示し，メモリを節約して高速化する
            with torch.no_grad():
                output = model(input_ids, input_masks, input_features)

            output = output.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(output, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            # Store predictions
            epoch_predictions.append(output)

        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

        # Collect all predictions for the epoch
        all_predictions.append(np.concatenate(epoch_predictions, axis=0))
    
    # Return overall Spearman scores and predictions
    return all_predictions

# CSV読み込み
train_df = pd.read_csv('train.csv')
train_df['timeToReply'] = train_df['timeToReply'].apply(parse_time_to_seconds)
train_df['reviewCreatedVersion'].fillna(-1, inplace=True)

# スコアを５つから３つに変更
train_df['score'] = train_df['score'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})

loss_weights = torch.tensor([1.2, 1.5, 0.8]).to(device).float()
print(train_df.score.value_counts())

# Suppress specific warnings
warnings.filterwarnings('ignore')

# Load the tokenizers and models from Hugging Face
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
distilbert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

xlnet_tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
xlnet_model = XLNetModel.from_pretrained('xlnet-base-cased')

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')


# Initialize models
distilbert_model = MultiModel(DistilBertModel.from_pretrained('distilbert-base-uncased'), hidden_size=768, num_features=3, num_labels=3)
xlnet_model = MultiModel(XLNetModel.from_pretrained('xlnet-base-cased'), hidden_size=768, num_features=3, num_labels=3)
roberta_model = MultiModel(RobertaModel.from_pretrained('roberta-base'), hidden_size=768, num_features=3, num_labels=3)

# Preprocess data for each model
train_df, validation_df, train_labels, validation_labels = train_test_split(train_df, train_df.score.values, random_state=123, test_size=0.1)
train_features = torch.tensor(train_df[['thumbsUpCount', 'timeToReply', 'reviewCreatedVersion']].values).float()
validation_features = torch.tensor(validation_df[['thumbsUpCount', 'timeToReply', 'reviewCreatedVersion']].values).float()

batch_size = 128

# Train the models
print("Training DistilBERT model...")
train_inputs = preprocess_data(train_df, distilbert_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, distilbert_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"],train_features, torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], validation_features, torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
distilbert_predictions = train_model(distilbert_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(distilbert_model.state_dict(), 'distilbert_ensamble_features_3d.pth')
print("Model saved successfully.")

batch_size = 72

print("Training RoBERTa model...")
train_inputs = preprocess_data(train_df, roberta_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, roberta_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], train_features, torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], validation_features, torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
roberta_predictions = train_model(roberta_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(roberta_model.state_dict(), 'roberta_ensamble_feature_3d.pth')
print("Model saved successfully.")

batch_size = 52

print("Training XLNet model...")
train_inputs = preprocess_data(train_df, xlnet_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, xlnet_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], train_features, torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], validation_features, torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
xlnet_predictions = train_model(xlnet_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(xlnet_model.state_dict(), 'xlnet_ensamble_features_3d.pth')
print("Model saved successfully.")

# Combine predictions by averaging
distilbert_predictions = torch.tensor(distilbert_predictions).to(device)
xlnet_predictions = torch.tensor(xlnet_predictions).to(device)
roberta_predictions = torch.tensor(roberta_predictions).to(device)
ensemble_predictions = (distilbert_predictions + xlnet_predictions + roberta_predictions) / 3

labels = torch.tensor(validation_labels).to('cpu').numpy()
for epoch, prediction in enumerate(ensemble_predictions):
    print(f"Epoch {epoch + 1} Validation Accuracy: {flat_accuracy(prediction.detach().cpu().numpy(), labels):.4f}")