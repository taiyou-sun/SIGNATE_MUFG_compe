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
import re
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import torch.nn.functional as F

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

def get_dataloader(input_ids, attention_mask, batch_size, shuffle=True):
    train_data = TensorDataset(input_ids, attention_mask)
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
    

def outputcheck_model(model, validation_dataloader):
    model.eval()

    # Validation

    epoch_predictions = []

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_masks = batch

        # gradientsを計算または保存しないようにモデルに指示し，メモリを節約して高速化する
        with torch.no_grad():
            output = model(input_ids, input_masks)

        output = output.detach().cpu().numpy()

        # Store predictions
        epoch_predictions.append(output)

    # Collect all predictions for the batch
    predictions = np.concatenate(epoch_predictions, axis=0)
    
    # Return predictions
    return predictions

# CSV読み込み
train_df = pd.read_csv('train.csv')
train_df['timeToReply'] = train_df['timeToReply'].apply(parse_time_to_seconds)
train_df['reviewCreatedVersion'].fillna(-1, inplace=True)

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
distilbert_model = MultiModel(DistilBertModel.from_pretrained('distilbert-base-uncased'), hidden_size=768, num_labels=5)
xlnet_model = MultiModel(XLNetModel.from_pretrained('xlnet-base-cased'), hidden_size=768, num_labels=5)
roberta_model = MultiModel(RobertaModel.from_pretrained('roberta-base'), hidden_size=768, num_labels=5)

# Load the models
distilbert_model.load_state_dict(torch.load('distilbert_ensamble_2.pth'))
distilbert_model.to(device)
xlnet_model.load_state_dict(torch.load('xlnet_ensamble_2.pth'))
xlnet_model.to(device)
roberta_model.load_state_dict(torch.load('roberta_ensamble_2.pth'))
roberta_model.to(device)

batch_size = 64

train_features = torch.tensor(train_df[['thumbsUpCount', 'timeToReply', 'reviewCreatedVersion']].values).float()

#Train the models
print("Loading DistilBERT model...")
train_inputs = preprocess_data(train_df, distilbert_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], batch_size=batch_size)
distilbert_predictions = outputcheck_model(distilbert_model, train_dataloader)
print("Prediction completed.")

print("Loading XLNet model...")
train_inputs = preprocess_data(train_df, xlnet_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], batch_size=batch_size)
xlnet_predictions = outputcheck_model(xlnet_model, train_dataloader)
print("Prediction completed.")

print("Loading RoBERTa model...")
train_inputs = preprocess_data(train_df, roberta_tokenizer, ['review', 'replyContent'])
train_dataloader = get_dataloader(train_inputs["input_ids"], train_inputs["attention_mask"], batch_size=batch_size)
roberta_predictions = outputcheck_model(roberta_model, train_dataloader)
print("Prediction completed.")

# Combine predictions by averaging
distilbert_predictions = torch.tensor(distilbert_predictions)
xlnet_predictions = torch.tensor(xlnet_predictions)
roberta_predictions = torch.tensor(roberta_predictions)
ensemble_predictions = (distilbert_predictions + xlnet_predictions + roberta_predictions) / 3

prediction_labels = torch.argmax(ensemble_predictions, axis=1).detach().cpu().numpy()
save_df = pd.DataFrame({
    'id': train_df['Unnamed: 0'],
    'lable': train_df['score'],
    'prediction': prediction_labels,
    'match': train_df['score'] == prediction_labels
})
save_df.to_csv('outputcheck.csv', index=False)