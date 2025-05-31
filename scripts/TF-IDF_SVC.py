import numpy as np
import pandas as pd
import re

df = pd.read_csv("/content/drive/MyDrive/intern/MUFG Data Science Champion Ship 2024/train.csv")
reviews = df.review.values
replies = df.replyContent.values
sentences = [reviews[i] + " " + replies[i] for i in range(len(reviews))]
sentences = np.array(sentences)

df_train = pd.DataFrame({
    'text': sentences,
    'label': df.score.values
})

df = pd.read_csv("/content/drive/MyDrive/intern/MUFG Data Science Champion Ship 2024/test.csv")
reviews = df.review.values
replies = df.replyContent.values
sentences = [reviews[i] + " " + replies[i] for i in range(len(reviews))]
sentences = np.array(sentences)

df_test = pd.DataFrame({
    'id': df['Unnamed: 0'],
    'text': sentences
})

def split_into_words(text):
    words = re.split(r'\s+|[,\.:!?"&]', text)
    return words

df_train['new_text'] = df_train['text'].apply(lambda x: split_into_words(x))
df_test['new_text'] = df_test['text'].apply(lambda x: split_into_words(x))

from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

X_train = [' '.join(row) for row in df_train['new_text'].values]
X_test = [' '.join(row) for row in df_test['new_text'].values]

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

df_1 = pd.DataFrame({'Id': df_test['id'], 'Category': y_pred})
df_1.to_csv('output1.csv', index=False, header=False)