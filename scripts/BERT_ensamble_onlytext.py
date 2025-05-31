import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import torch
from tqdm import tqdm, trange
from torch import nn
import warnings
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import re
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
import math

from sklearn.metrics import cohen_kappa_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# 必要なnltkデータをダウンロード
nltk.download('stopwords')
nltk.download('punkt_tab')

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

# ストップワードを削除する関数
def remove_stopwords(sentence):
    words = word_tokenize(sentence)
    filtered_sentence = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)

class MultiModel(nn.Module):
    def __init__(self, model, hidden_size, num_labels):
        super(MultiModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]  # CLS token
        dropout_output = self.dropout(cls_output)
        linear_output = self.linear(dropout_output)
        final_output = self.softmax(linear_output)
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

def get_dataloader(input_ids, attention_mask, labels, batch_size, shuffle=True):
    train_data = TensorDataset(input_ids, attention_mask, labels)
    if (shuffle):
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def compute_metrics(preds, labels):
    qwk = cohen_kappa_score(labels, preds.argmax(-1), weights='quadratic')
    return qwk
    

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
            input_ids, input_masks, labels = batch
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, input_masks)
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
        eval_loss, eval_score = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_masks, labels = batch
            

            # gradientsを計算または保存しないようにモデルに指示し，メモリを節約して高速化する
            with torch.no_grad():
                output = model(input_ids, input_masks)

            output = output.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            tmp_eval_score = compute_metrics(output, label_ids)

            eval_score += tmp_eval_score
            nb_eval_steps += 1

            # Store predictions
            epoch_predictions.append(output)
            

        print("Validation qwk score: {}".format(eval_score/nb_eval_steps))

        # Collect all predictions for the epoch
        all_predictions.append(np.concatenate(epoch_predictions, axis=0))
    # Return overall Spearman scores and predictions
    return all_predictions

# CSV読み込み
train_df = pd.read_csv('train.csv')
train_df['timeToReply'] = train_df['timeToReply'].apply(parse_time_to_seconds)
train_df['reviewCreatedVersion'].fillna(-1, inplace=True)

# 英語のストップワードリストを取得
stop_words = set(stopwords.words('english'))
stop_words.update(['hello', 'thanks', 'friend', 'bank', 'bankapp', 'dear', 'thank', ','])

# sentencesの各文からストップワードを削除
train_df['review'] = [remove_stopwords(review) for review in train_df.review.values]
train_df['replyContent'] = [remove_stopwords(replie) for replie in train_df.replyContent.values]

loss_weights = []
score_counts = train_df.score.value_counts()
print(score_counts)
for i in range(len(score_counts)):
    loss_weights.append(max(math.log(len(train_df) / score_counts[i]), 1.0))

loss_weights = torch.tensor([1.0, 2.0, 2.0, 1.5, 0.8]).to(device).float()
print(loss_weights)

# Suppress specific warnings
warnings.filterwarnings('ignore')

# Load the tokenizers and models from Hugging Face
mdeberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
mdeberta_model = AutoModel.from_pretrained('microsoft/mdeberta-v3-base')

deberta_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
deberta_model = AutoModel.from_pretrained('microsoft/deberta-v3-base')

debertaL_tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
debertaL_model = AutoModel.from_pretrained('microsoft/deberta-v3-large')

# Initialize models
mdeberta_model = MultiModel(mdeberta_model, hidden_size=768, num_labels=5)
deberta_model = MultiModel(deberta_model, hidden_size=768, num_labels=5)
debertaL_model = MultiModel(debertaL_model, hidden_size=1024, num_labels=5)

# Preprocess data for each model
train_df, validation_df, train_labels, validation_labels = train_test_split(train_df, train_df.score.values, random_state=123, test_size=0.1)
train_df = train_df.astype(str)
validation_df = validation_df.astype(str)

batch_size = 40

# Train the models
print("Training mDeberta model...")
train_inputs = preprocess_data(train_df, mdeberta_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, mdeberta_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
mdeberta_predictions = train_model(mdeberta_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(mdeberta_model.state_dict(), 'mdeberta_ensamble_onlytext.pth')
print("Model saved successfully.")
batch_size = 10

print("Training Deberta-Large model...")
train_inputs = preprocess_data(train_df, debertaL_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, debertaL_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
debertaL_predictions = train_model(debertaL_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(debertaL_model.state_dict(), 'deberta_large_ensamble_onlytext.pth')
print("Model saved successfully.")

batch_size = 45

# Train the models
print("Training Deberta model...")
train_inputs = preprocess_data(train_df, deberta_tokenizer, ['review', 'replyContent'])
validation_inputs = preprocess_data(validation_df, deberta_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], torch.tensor(train_labels), batch_size=batch_size)
validation_dataloader = get_dataloader(validation_inputs["input_ids"], validation_inputs["attention_mask"], torch.tensor(validation_labels), batch_size=batch_size, shuffle=False)
deberta_predictions = train_model(deberta_model, train_dataloader, validation_dataloader, loss_weights)
torch.save(deberta_model.state_dict(), 'deberta_ensamble_onlytext.pth')
print("Model saved successfully.")

# Combine predictions by averaging
mdeberta_predictions = torch.tensor(mdeberta_predictions).to(device)
deberta_predictions = torch.tensor(deberta_predictions).to(device)
debertaL_predictions = torch.tensor(debertaL_predictions).to(device)
ensemble_predictions = (mdeberta_predictions + deberta_predictions + debertaL_predictions) / 3

labels = torch.tensor(validation_labels).to('cpu').numpy()
for epoch, prediction in enumerate(ensemble_predictions):
    print(f"Epoch {epoch + 1} Validation qwk score: {compute_metrics(prediction.detach().cpu().numpy(), labels):.4f}")