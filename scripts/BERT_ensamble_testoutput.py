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

def get_dataloader(input_ids,attention_mask, features, batch_size):
    train_data = TensorDataset(input_ids, attention_mask, features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def split_into_words(text):
    words = re.split(r'\s+|[,\.:!?"&]', text)
    return words

def get_sentence(df):
    reviews = df.review.values
    replies = df.replyContent.values
    sentences = [reviews[i] + " " + replies[i] for i in range(len(reviews))]
    sentences = np.array(sentences)

    return sentences

def model(model, train_dataloader):
    model.eval()
    # Validation

    epoch_predictions = []

    for batch in train_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_masks, input_features = batch

        # gradientsを計算または保存しないようにモデルに指示し，メモリを節約して高速化する
        with torch.no_grad():
            output = model(input_ids, input_masks, input_features)

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

test_df = pd.read_csv('test.csv')
test_df['timeToReply'] = test_df['timeToReply'].apply(parse_time_to_seconds)
test_df['reviewCreatedVersion'].fillna(-1, inplace=True)

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
distilbert_model = MultiModel(DistilBertModel.from_pretrained('distilbert-base-uncased'), hidden_size=768, num_features=3, num_labels=5)
xlnet_model = MultiModel(XLNetModel.from_pretrained('xlnet-base-cased'), hidden_size=768, num_features=3, num_labels=5)
roberta_model = MultiModel(RobertaModel.from_pretrained('roberta-base'), hidden_size=768, num_features=3, num_labels=5)

# Load the models
distilbert_model.load_state_dict(torch.load('distilbert_ensamble_features_weak.pth'))
distilbert_model.to(device)
xlnet_model.load_state_dict(torch.load('xlnet_ensamble_features_weak.pth'))
xlnet_model.to(device)
roberta_model.load_state_dict(torch.load('roberta_ensamble_feature_weak.pth'))
roberta_model.to(device)

test_features = torch.tensor(test_df[['thumbsUpCount', 'timeToReply', 'reviewCreatedVersion']].values).float()

print("Loading TF-IDF+SVC model...")
sentences = get_sentence(train_df)
df_train = pd.DataFrame({
    'text': sentences,
    'label': train_df.score.values
})
df_train['text'] = df_train['text'].apply(lambda x: split_into_words(x))

sentences = get_sentence(test_df)
df_test = pd.DataFrame({
    'text': sentences
})
df_test['text'] = df_test['text'].apply(lambda x: split_into_words(x))

X_train = [' '.join(row) for row in df_train['text'].values]
X_test = [' '.join(row) for row in df_test['text'].values]
# tf-idf
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("tfidf check")
# svc model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, df_train['label'])
print("svc check")
# output
y_pred = svm_model.predict(X_test_tfidf)
SVC_predictions = torch.tensor(y_pred).to(device)
num_classes = 5  # クラス数（ここでは0から4の5クラス）
SVC_predictions = F.one_hot(SVC_predictions, num_classes=num_classes).to(torch.float32)
print("Prediction completed.")


batch_size = 128

# Train the models
print("Loading DistilBERT model...")
test_inputs = preprocess_data(test_df, distilbert_tokenizer, ['review', 'replyContent'])
test_dataloader = get_dataloader(test_inputs["input_ids"], test_inputs["attention_mask"],test_features, batch_size=batch_size)
distilbert_predictions = model(distilbert_model, test_dataloader)
print("Prediction completed.")


batch_size = 52

print("Loading XLNet model...")
test_inputs = preprocess_data(test_df, xlnet_tokenizer, ['review', 'replyContent'])
test_dataloader = get_dataloader(test_inputs["input_ids"], test_inputs["attention_mask"],test_features, batch_size=batch_size)
xlnet_predictions = model(xlnet_model, test_dataloader)
print("Prediction completed.")


batch_size = 72

print("Loading RoBERTa model...")
test_inputs = preprocess_data(test_df, roberta_tokenizer, ['review', 'replyContent'])
test_dataloader = get_dataloader(test_inputs["input_ids"], test_inputs["attention_mask"],test_features, batch_size=batch_size)
roberta_predictions = model(roberta_model, test_dataloader)
print("Prediction completed.")

# Combine predictions by averaging
distilbert_predictions = torch.tensor(distilbert_predictions).to(device)
xlnet_predictions = torch.tensor(xlnet_predictions).to(device)
roberta_predictions = torch.tensor(roberta_predictions).to(device)
SVC_predictions = SVC_predictions.to(device)
ensemble_predictions = (distilbert_predictions + xlnet_predictions + roberta_predictions) / 3

prediction_labels = torch.argmax(ensemble_predictions, axis=1).detach().cpu().numpy()
save_df = pd.DataFrame({
    'id': test_df['Unnamed: 0'],
    'prediction': prediction_labels,
})
save_df.to_csv('output.csv', index=False, header=False)