import os

# 現在のLD_LIBRARY_PATHを取得し、CONDA_PREFIX/lib/を追加
current_ld_library_path = os.getenv('LD_LIBRARY_PATH', '')
conda_prefix_lib = os.getenv('CONDA_PREFIX', '') + '/lib/'

# 環境変数LD_LIBRARY_PATHを設定
new_ld_library_path = current_ld_library_path + ':' + conda_prefix_lib if current_ld_library_path else conda_prefix_lib
os.environ['LD_LIBRARY_PATH'] = new_ld_library_path

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.utils import pad_sequences
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from sklearn.metrics import cohen_kappa_score
import math

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
    
class AutoModelMultimodal(nn.Module):
    def __init__(self, num_labels, num_features, dropout_rate=0.1):
        super().__init__()
        self.automodel = AutoModel.from_pretrained('microsoft/mdeberta-v3-base')
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base')
        
        # Improved feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.automodel.config.hidden_size),
            nn.LayerNorm(self.automodel.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.automodel.config.hidden_size, self.automodel.config.hidden_size),
            nn.LayerNorm(self.automodel.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.automodel.config.hidden_size, 1)
        )
        self.num_labels = num_labels

    def forward(self, input_ids, additional_features, attention_mask, token_type_ids, labels=None):
        automodel_output = self.automodel(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids).last_hidden_state[:, 0]
        
        feature_encoded = self.feature_encoder(additional_features)

        # 追加特徴量を融合された特徴量に加算
        fused_features = automodel_output + feature_encoded
        
        # 融合された特徴量を用いて分類
        logits = self.classifier(fused_features)
        
        return logits.squeeze(1)
        
    def predict(self, input_ids, additional_features, attention_mask, token_type_ids):
        self.eval()  # 推論モードに設定
        with torch.no_grad():  # 勾配計算をオフに
            predictions = self.forward(input_ids, additional_features, attention_mask, token_type_ids)
        predictions = predictions.detach().cpu().numpy()
        return predictions


# 使用例
num_labels = 5
num_features = 3
model = AutoModelMultimodal(num_labels=num_labels, num_features=num_features)
model.cuda()
save_dir = 'pths/mDeberta_qwk_w_regression'
kf_splits = 5
# testの実行

df_test = pd.read_csv('test.csv')
df_test['timeToReply'] = df_test['timeToReply'].apply(parse_time_to_seconds)
df_test['reviewCreatedVersion'].fillna(-1, inplace=True)

reviews = df_test.review.values
replies = df_test.replyContent.values
sentences = ["[CLS] " + reviews[i] + " [SEP] " + replies[i] for i in range(len(reviews))]
features = df_test[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply']].values

tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# AutoModelトークナイザーを使用して、トークンをAutoModelボキャブラリのインデックス番号に変換
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 最大シーケンス長
MAX_LEN = 350
# 最大長に満たない場合は 0 で埋める
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# attention masksの作成
attention_masks = []

for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

test_ids = torch.tensor(input_ids).to(device)
test_features = torch.tensor(features).float().to(device)
test_masks = torch.tensor(attention_masks).to(device)

sum_predictions = np.zeros(len(test_ids))
batch_size = 128
for fold in range(kf_splits):
    model.load_state_dict(torch.load(f'{save_dir}/fold{fold + 1}.pth'))
    model.cuda()

    predictions = []
    i = 0
    while True:
        gap = 0
        if i >= len(test_ids):
            break

        prediction = model.predict(test_ids[i:i+batch_size], test_features[i:i+batch_size], test_masks[i:i+batch_size], None)
        if i == 0:
            predictions = prediction
        else:
            predictions = np.concatenate([predictions, prediction])

        i += batch_size

        predictions = np.array(predictions)
    
    sum_predictions += predictions

ensambled_predictions = np.clip(np.round(sum_predictions / kf_splits, decimals=0).astype(int),0,4)

save_df = pd.DataFrame({
    'id': df_test['Unnamed: 0'],
    'prediction': ensambled_predictions,
})
save_df.to_csv(f'{save_dir}_ensamble.csv', index=False, header=False) 
