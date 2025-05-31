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

from transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
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

print("XLNetのvalidationなし")
# CSV読み込み
df_all = pd.read_csv('train.csv')
df_all.drop(['Unnamed: 0'], axis=1, inplace=True)
df_all['timeToReply'] = df_all['timeToReply'].apply(parse_time_to_seconds)
df_all['reviewCreatedVersion'].fillna(-1, inplace=True)

df_all['score'] = df_all['score'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})

df = df_all

reviews = df.review.values
reviews = ["[CLS] " + review + " [SEP] " for review in reviews]
replies = df.replyContent.values
replies = [reply + " [SEP]" for reply in replies]
sentences = [reviews[i] + replies[i] for i in range(len(reviews))]
features = df[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply']].values
labels = df.score.values

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

max_len = []
# 1文づつ処理
for sent in sentences:
    # Tokenizeで分割
    token_words = tokenizer.tokenize(sent)
    # 文章数を取得してリストへ格納
    max_len.append(len(token_words))
# 最大の値を確認
print('最大単語数: ', max(max_len))
print('上記の最大単語数にSpecial token（[CLS], [SEP]）の+2をした値が最大単語数')

# 最大シーケンス長
MAX_LEN = 350

# XLNetトークナイザーを使用して、トークンをXLNetボキャブラリのインデックス番号に変換
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 最大長に満たない場合は 0 で埋める
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# attention masksの作成
attention_masks = []

for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

train_ids, validation_ids, train_features, validation_features,train_masks,validation_masks, train_labels, validation_labels = train_test_split(input_ids, features,attention_masks, labels, random_state=123, test_size=0.0000000000000000000000001)

train_ids = torch.tensor(train_ids)
validation_ids = torch.tensor(validation_ids)
train_features = torch.tensor(train_features).float()
validation_features = torch.tensor(validation_features).float()
train_labels = torch.tensor(train_labels)
validation_labels = torch.tensor(validation_labels)
train_masks = torch.tensor(train_masks)
validation_masks = torch.tensor(validation_masks)

batch_size  = 32

print("XLNet_3dの学習しない層を消したモデル")
train_data = TensorDataset(train_ids,train_features, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_ids,validation_features, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
#

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.ln(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out += residual
        return self.relu(out)

class XLNetMultimodal(nn.Module):
    def __init__(self, num_labels, num_features, dropout_rate=0.1):
        super().__init__()
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        
        # Improved feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.xlnet.config.hidden_size),
            nn.LayerNorm(self.xlnet.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.fusion = nn.MultiheadAttention(self.xlnet.config.hidden_size, 8, dropout=dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.xlnet.config.hidden_size, self.xlnet.config.hidden_size),
            nn.LayerNorm(self.xlnet.config.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.xlnet.config.hidden_size, num_labels)
        )
        self.num_labels = num_labels

    def forward(self, input_ids, additional_features, attention_mask, token_type_ids, labels=None):
        xlnet_output = self.xlnet(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids).last_hidden_state
        
        feature_encoded = self.feature_encoder(additional_features)
        
        # XLNetの出力と追加特徴量の融合
        fused_xnet, _ = self.fusion(xlnet_output, xlnet_output, xlnet_output)
        # 追加特徴量を融合された特徴量に加算
        fused_features = fused_xnet[:, 0] + feature_encoded
        
        # 融合された特徴量を用いて分類
        logits = self.classifier(fused_features)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
        
    def predict(self, input_ids, additional_features, attention_mask, token_type_ids):
        self.eval()  # 推論モードに設定
        with torch.no_grad():  # 勾配計算をオフに
            logits = self.forward(input_ids, additional_features, attention_mask, token_type_ids)
            predictions = torch.argmax(logits, dim=-1)
        return predictions.detach().cpu().numpy()

# 使用例
model = XLNetMultimodal(num_labels=3, num_features=3)
model.cuda()
# 入力は前のアプローチと同様

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

train_loss_set = []

epoch = 10

for _ in trange(epoch, desc="Epoch"):

  #Training

  #モデルをトレーニングモードにする
  model.train()

  # Tracking variables
  tr_loss = 0
  nb_tr_examples, nb_tr_steps = 0, 0

  for step, batch in enumerate(train_dataloader):
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_features, b_input_mask, b_labels = batch
    optimizer.zero_grad()
    outputs = model(b_input_ids, b_input_features, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
    loss = outputs[0]
    logits = outputs[1]
    train_loss_set.append(loss.item())
    loss.backward()
    optimizer.step()

    # tracking variablesを更新
    tr_loss += loss.item()
    nb_tr_examples += b_input_ids.size(0)
    nb_tr_steps += 1

  print("Train loss: {}".format(tr_loss/nb_tr_steps))

  # Validation

  # モデルをevaluationモードにする
  model.eval()

  # Tracking variables
  eval_loss, eval_accuracy = 0, 0
  nb_eval_steps, nb_eval_examples = 0, 0

  for batch in validation_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_features, b_input_mask, b_labels = batch

    # gradientsを計算または保存しないようにモデルに指示し，メモリを節約して高速化する
    with torch.no_grad():
      output = model(b_input_ids, b_input_features, token_type_ids=None, attention_mask=b_input_mask)
      logits = output

    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_accuracy += tmp_eval_accuracy
    nb_eval_steps += 1

  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

torch.save(model.state_dict(), 'XLNet_3d.pth')
print("Model saved successfully.")

# testの実行

model.load_state_dict(torch.load('XLNet_3d.pth'))
model.to(device)

df_test = pd.read_csv('test.csv')
df_test['timeToReply'] = df_test['timeToReply'].apply(parse_time_to_seconds)
df_test['reviewCreatedVersion'].fillna(-1, inplace=True)

reviews = df_test.review.values
reviews = ["[CLS] " + review + " [SEP] " for review in reviews]
replies = df_test.replyContent.values
replies = [reply + " [SEP]" for reply in replies]
sentences = [reviews[i] + replies[i] for i in range(len(reviews))]
features = df_test[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply']].values

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

# XLNetトークナイザーを使用して、トークンをXLNetボキャブラリのインデックス番号に変換
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]


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

predictions = []
batch_size = 8
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

save_df = pd.DataFrame({
    'id': df_test['Unnamed: 0'],
    'prediction': predictions,
})
save_df.to_csv('output_3d.csv', index=False) 