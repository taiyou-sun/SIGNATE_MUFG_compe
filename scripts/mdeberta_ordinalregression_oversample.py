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
from sklearn.model_selection import KFold

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
      # 英語のストップワードリストを取得
    stop_words = set(stopwords.words('english'))
    stop_words.update(['hello', 'thanks', 'friend', 'bank', 'bankapp', 'dear', 'thank', ','])
    words = word_tokenize(sentence)
    filtered_sentence = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_sentence)


def limit_scorecounts(df, label, limit):
    zero_score_rows = df[df['score'] == label]

    # scoreが0の行数が400以上の場合、最初の400行を残す
    if len(zero_score_rows) > limit:
        zero_score_rows = zero_score_rows.sample(n=limit, random_state=42)

    # scoreが0ではない行を抽出
    non_zero_score_rows = df[df['score'] != label]

    # 両方のデータフレームを結合
    limited_df = pd.concat([zero_score_rows, non_zero_score_rows], ignore_index=True)

    return limited_df

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
            nn.Linear(256, 3),
            nn.LayerNorm(3),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.automodel.config.hidden_size+num_features, self.automodel.config.hidden_size+num_features),
            nn.LayerNorm(self.automodel.config.hidden_size+num_features),
            nn.GELU(),
            nn.Linear(self.automodel.config.hidden_size+num_features, num_labels-1),
            nn.Sigmoid(),
        )
        self.num_labels = num_labels

    def forward(self, input_ids, additional_features, attention_mask, token_type_ids, labels=None):
        automodel_output = self.automodel(input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  token_type_ids=token_type_ids).last_hidden_state[:, 0]
        
        feature_encoded = self.feature_encoder(additional_features)

        # 追加特徴量を融合された特徴量に加算
        fused_features = torch.concat([automodel_output, feature_encoded], dim=1)
        
        # 融合された特徴量を用いて分類
        logits = self.classifier(fused_features)

        return logits
        
    def predict(self, input_ids, additional_features, attention_mask, token_type_ids):
        self.eval()  # 推論モードに設定
        with torch.no_grad():  # 勾配計算をオフに
            logits = self.forward(input_ids, additional_features, attention_mask, token_type_ids)
        predictions = categorize_labels(logits)
        predictions = predictions.detach().cpu().numpy()
        return predictions

def compute_metrics(preds, labels):
    qwk = cohen_kappa_score(labels, preds, weights='quadratic')
    return qwk

def flat_accuracy(preds, labels):
    pred_flat = preds.flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def categorize_labels(logits, thresholds=0.5):

    labels = np.zeros(logits.shape[0])
    for i, logits in enumerate(logits):
        for l in logits:
            if l > thresholds:
                labels[i] += 1
            else:
                break
    return labels

def to_ordinal(y, num_classes=None, dtype="float32"):
    y = torch.tensor(y, dtype=torch.int)
    input_shape = y.shape
    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = torch.max(y) + 1
    n = y.shape[0]
    range_values = torch.arange(num_classes - 1)
    range_values = range_values.unsqueeze(0).repeat(n, 1)
    ordinal = torch.zeros((n, num_classes - 1), dtype=getattr(torch, dtype))
    ordinal[range_values < y.unsqueeze(-1)] = 1
    output_shape = input_shape + (num_classes - 1,)
    ordinal = ordinal.reshape(output_shape)
    return ordinal

# 使用例
num_labels = 5
num_features = 3
# # 入力は前のアプローチと同様


# CSV読み込み
df_original = pd.read_csv('train.csv')
# df_original = df_original[~df_original['score'].isin([0, 4])]
# df_original = limit_scorecounts(df_original, 3, 450)
print(df_original.score.value_counts())
original_size = len(df_original)
df_add = pd.read_csv('llama/train_augmented_by_llama2.csv')
df_all = pd.concat([df_original, df_add])
# df_all['score'] = df_all['score'].replace({1: 0, 2: 1, 3: 2})
# df_all['score'] = df_all['score'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 2})
# df_all.drop(['Unnamed: 0'], axis=1, inplace=True)
df_all['timeToReply'] = df_all['timeToReply'].apply(parse_time_to_seconds)
df_all['reviewCreatedVersion'].fillna(-1, inplace=True)
print(df_all.score.value_counts())


df = df_all

# sentencesの各文からストップワードを削除
# 必要なnltkデータをダウンロード
# nltk.download('stopwords')
# nltk.download('punkt_tab')
# reviews = [remove_stopwords(review) for review in df.review.values]
# replies = [remove_stopwords(replie) for replie in df.replyContent.values]
reviews = df.review.values
replies = df.replyContent.values
sentences = ["[CLS] " + reviews[i] + " [SEP] " + replies[i] for i in range(len(reviews))]

features = df[['thumbsUpCount', 'reviewCreatedVersion', 'timeToReply', 'Unnamed: 0']].values
labels = df.score.values

loss_weights = torch.tensor([0.5, 1.0, 1.0, 0.9, 0.3]).to(device).float()
print(loss_weights)

tokenizer = AutoTokenizer.from_pretrained('microsoft/mdeberta-v3-base', do_lower_case=True)
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

MAX_LEN = 350

# AutoModelトークナイザーを使用して、トークンをAutoModelボキャブラリのインデックス番号に変換
input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

# 最大長に満たない場合は 0 で埋める
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# attention masksの作成
attention_masks = []

for seq in input_ids:
  seq_mask = [float(i>0) for i in seq]
  attention_masks.append(seq_mask)

attention_masks = np.array(attention_masks)

# 水増ししたデータを一旦分離（validationに入れないため）
original_ids = input_ids[:original_size]
add_ids = torch.tensor(input_ids[original_size:])
original_features = features[:original_size]
# print(features[original_size:])
# for i, feature in enumerate(features[original_size:]):
#     print(i)
#     print(int(feature[0]))
#     print(float(feature[1]))
add_features = torch.tensor(features[original_size:])
original_masks = attention_masks[:original_size]
add_masks = torch.tensor(attention_masks[original_size:])
original_labels = labels[:original_size]
add_labels = torch.tensor(labels[original_size:])

batch_size = 20
#
epoch = 10
kf_splits = 5

total_score = 0
total_accuracy = 0

# K-Fold Cross Validation
kf = KFold(n_splits=kf_splits, shuffle=True, random_state=123)

for fold, (train_index, val_index) in enumerate(kf.split(original_ids)):

    model = AutoModelMultimodal(num_labels=num_labels, num_features=num_features)
    model.cuda()

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


    # Split data into train and validation sets for this fold
    train_ids, validation_ids = original_ids[train_index], original_ids[val_index]
    train_features, validation_features = original_features[train_index], original_features[val_index]
    train_masks, validation_masks = original_masks[train_index], original_masks[val_index]
    train_labels, validation_labels = original_labels[train_index], original_labels[val_index]
    
    # Convert to tensors

    train_ids = torch.concat([torch.tensor(train_ids), add_ids])
    validation_ids = torch.tensor(validation_ids)
    train_features = torch.concat([torch.tensor(train_features).float(), add_features.float()])
    validation_features = torch.tensor(validation_features).float()
    train_labels = to_ordinal(torch.concat([torch.tensor(train_labels).float(), add_labels.float()]))
    validation_labels = to_ordinal(torch.tensor(validation_labels))
    train_masks = torch.concat([torch.tensor(train_masks), add_masks])
    validation_masks = torch.tensor(validation_masks)

    # Create DataLoaders
    train_data = TensorDataset(train_ids, train_features, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_ids, validation_features, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Training loop
    max_score = 0
    max_accuracy = 0
    for _ in trange(epoch, desc="Epoch"):
        # Training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        criterion = nn.BCEWithLogitsLoss()

        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_features, b_input_mask, b_ordlabels = batch
            b_input_features = b_input_features[:, :num_features]
            optimizer.zero_grad()
            predictions = model(b_input_ids, b_input_features, token_type_ids=None, attention_mask=b_input_mask)
            loss = criterion(predictions, b_ordlabels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            # predictions = predictions.detach().cpu().numpy()
            # predictions = np.clip(np.round(predictions, decimals=0).astype(int),0,4)
            # lables = b_ordlabels.to('cpu').numpy()
            # prediction_data = np.concatenate((b_row_ids.reshape(-1, 1),lables.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
            # prediction_fail = prediction_data[prediction_data[:,1] != prediction_data[:,2]]
            # np.set_printoptions(suppress=True)
            # print(prediction_fail)

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation
        model.eval()
        eval_loss, eval_score, eval_accuracy = 0, 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_features, b_input_mask, b_ordlabels = batch
            b_row_ids = b_input_features[:, num_features].cpu().numpy()
            b_input_features = b_input_features[:, :num_features]

            with torch.no_grad():
                logits = model(b_input_ids, b_input_features, token_type_ids=None, attention_mask=b_input_mask)

            logits = logits.detach().cpu().numpy()
            b_ordlabels = b_ordlabels.to('cpu').numpy()

            predictions = categorize_labels(logits)
            lables = categorize_labels(b_ordlabels)
            # prediction_data = np.concatenate((b_row_ids.reshape(-1, 1),lables.reshape(-1, 1), predictions.reshape(-1, 1)), axis=1)
            # prediction_fail = prediction_data[prediction_data[:,1] != prediction_data[:,2]]
            # np.set_printoptions(suppress=True)
            # print(prediction_fail)

            tmp_eval_score = compute_metrics(predictions, lables)
            tmp_eval_accuracy = flat_accuracy(predictions, lables)

            if np.isnan(tmp_eval_score):
                # 予測が全て同じ場合は、qwkが計算できないため、1を代入
                tmp_eval_score = 1

            eval_score += tmp_eval_score
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        validation_score = eval_score / nb_eval_steps
        validation_accuracy = eval_accuracy / nb_eval_steps
        print("Validation qwk score: {}".format(validation_score))
        print("Validation Accuracy: {}".format(validation_accuracy))

        if validation_score > max_score:
            save_dir = f'pths/mDeberta_qwk_ordinalregression_oversampled'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(model.state_dict(), save_dir+f'/fold{fold + 1}.pth')
            print(f"{save_dir+f'/fold{fold + 1}.pth'} Model saved successfully.")
            max_score = validation_score
            max_accuracy = validation_accuracy

    total_score += max_score
    total_accuracy += max_accuracy
            
print("Average Validation qwk score: {}".format(total_score / kf_splits))
print("Average Validation Accuracy: {}".format(total_accuracy / kf_splits))



# テストデータの予測
model = AutoModelMultimodal(num_labels=num_labels, num_features=num_features)
model.cuda()
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
for fold in range(kf_splits):
    model.load_state_dict(torch.load(f'{save_dir}/fold{fold + 1}.pth'))
    model.cuda()

    predictions = []
    batch_size = 64
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
