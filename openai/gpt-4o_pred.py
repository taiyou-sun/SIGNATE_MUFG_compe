import os
import pandas as pd
import openai
from tqdm import tqdm
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI APIキーの設定（環境変数から読み取る）
openai.api_key = os.getenv("OPENAI_API_KEY")

# データの読み込み
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('../test.csv')

def process_data(df, is_training=True):
    """
    データを処理し、フォーマットされたテキストを返す。
    
    :param df: データフレーム
    :param is_training: トレーニングデータの場合True
    :return: フォーマットされたデータ
    """
    processed_data = []
    for _, row in df.iterrows():
        review = row['review']
        reply = row['replyContent']
        features = f"thumbsUpCount: {row['thumbsUpCount']}, reviewCreatedVersion: {row['reviewCreatedVersion']}, timeToReply: {row['timeToReply']}"
        if is_training:
            score = row['score']
            processed_data.append(f"Review: {review}\n Reply: {reply}\n Features: {features}\nScore: {score}")
        else:
            processed_data.append(f"Review: {review}\n Reply: {reply}\n Features: {features}")
    return '\n\n'.join(processed_data)

def predict_scores(training_data, test_data):
    """
    OpenAI APIを使用してスコアを予測する。
    
    :param training_data: トレーニングデータのテキスト
    :param test_data: テストデータのテキスト
    :return: 予測スコアのリスト
    """
    prompt = f"""Given the following training data:

{training_data}

Now, predict the score for each of the following test data. Provide only the predicted score as a number for each test case, separated by '---'.

{test_data}

Format your response as:
Score 4
---
Score 2
---
...and so on."""

    try:
        print(prompt)
        response = openai.ChatCompletion.create(
            model="gpt-4o",  # Use the correct model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant that predicts scores based on reviews and features."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.5,
        )
        print(response['choices'][0]['message']['content'])
        return [score.strip() for score in response['choices'][0]['message']['content'].strip().split('---')]
    except openai.error.OpenAIError as e:
        print(f"Error with OpenAI API: {e}")
        return []

def main():
    # トレーニングデータの処理
    training_data = process_data(train_df)

    # バッチサイズの設定
    BATCH_SIZE = 1
    predicted_scores = []

    # バッチ処理による予測
    for i in tqdm(range(0, len(test_df), BATCH_SIZE)):
        batch = test_df.iloc[i:i+BATCH_SIZE]
        batch_test_data = process_data(batch, is_training=False)
        batch_predictions = predict_scores(training_data, batch_test_data)
        predicted_scores.extend(batch_predictions)
        print(batch_predictions)
        print(len(batch_predictions))
        time.sleep(60)

    # 予測結果をDataFrameに追加
    test_df['predicted_score'] = predicted_scores

    # 結果の出力
    test_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()
