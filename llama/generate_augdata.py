import transformers
import torch
import pandas as pd
import csv
import re
# import subprocess

# command = "huggingface-cli login --token"
# subprocess.run(command, shell=True)


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# 学習データを読み込む
df = pd.read_csv("../train.csv")  # your_data.csvを実際のファイル名に置き換えてください

# カスタマイズされた層化抽出関数
def custom_stratified_sample(df, stratify_col, sample_sizes):
    samples = []
    for value, size in sample_sizes.items():
        # 特定の score に対して抽出
        group = df[df[stratify_col] == value]
        samples.append(group.sample(min(len(group), size)))
    return pd.concat(samples)


def csv_like_split(line):
    return next(csv.reader([line], skipinitialspace=True))

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


# サンプルサイズの指定: {score値: 取得したい行数}
sample_sizes = {
    4: 8,  # score = 4 に対して多めにサンプルを取得
    3: 8,  # score = 3 に対して少し少なめにサンプルを取得
    2: 7,  # score = 2 に対して少し少なめにサンプルを取得
    1: 7,   # score = 1 に対して少し少なめにサンプルを取得
    0: 7,
}

# 初期のデータ数を取得
init_count = df.shape[0]
current_number = init_count
column_num = df.shape[1]

# 水増しするデータ数
# batch_sizeをある程度小さくしないとモデルの性能が低いのでデータにバリエーションを付けられない
augument_size = 5000
batch_size = 7

# 生成されたデータを格納する変数
augumented_data = ""

# ターミネータを設定
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

while current_number < (init_count + augument_size):
    # カスタムサンプリング実行
    sampled_df = custom_stratified_sample(df, 'score', sample_sizes)
    sampled_df = sampled_df.rename(columns={'Unnamed: 0': 'id'})
    sampled_df['id'] = range(current_number - len(sampled_df), current_number)
    sampled_df.index = sampled_df['id']

    # scoreが1と2のデータを水増しするためのプロンプト
    messages = [
        {
            "role": "system",
            "content": """You are a helpful AI assistant that can generate data for machine learning. You are given a sample of data from a review dataset and are tasked with generating new data based on your analysis of the existing data."""
        },
        {
        "role": "user",
        "content": f"""
    Here is a sample of data from a review dataset:
    ```csv
    {sampled_df.to_csv(index=False)}
    ```

    This is just a small sample, and in the full dataset.

    Please analyze the characteristics of reviews with scores 0, 1, 2, 3, and 4, and then generate **only** {batch_size} new rows of data with a score of either 1, 2, 3, 0 or 4(priority to 1, 2 and 3), based on your analysis.  

    Ensure the following:

    Please less frequently start 'replyContent' with the words "Hello" or "Dear".
    Please don't start 'review' with the words "The app", "App" or "I'm" words!!!!
    * **Double Quotes:** Enclose the content of the 'review' column in double quotes.
    * **Sequential IDs:** Start the 'id' column from {current_number} and increment sequentially.

    Output the data in CSV format with the same header as the example. **Do not include any explanatory text or commentary, just the raw CSV data without header.**
    """
    }
    ]

    # プロンプトを生成
    prompt = pipeline.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # LLaMAを使ってデータを生成
    outputs = pipeline(
        prompt,
        max_new_tokens=8192,  # 生成するデータ量に応じて調整
        do_sample=True,
        temperature=0.8,  # データの多様性を調整
        top_p=0.9,
        repetition_penalty=1.0,
        eos_token_id=terminators,
    )

    output_message = outputs[0]["generated_text"][len(prompt):]


    try:
        # ```で囲まれた部分のみ切り出す
        start = output_message.find("```\n") + 4
        end = output_message.rfind("```")
        augumented_batch = output_message[start:end]
        
        if augumented_batch == "":
            raise ValueError("No data generated")

        # current_numberからバッチサイズ分の行を見つける
        start_index = augumented_batch.find("\"" + str(current_number) + "\"")
        if start_index == -1:
            start_index = augumented_batch.find(str(current_number))
        end_index = augumented_batch.rfind("\"" + str(current_number + batch_size - 1) + "\"")
        if end_index == -1:
            end_index = augumented_batch.rfind(str(current_number + batch_size - 1))

        # 最後の行は誤っている確率が高いので削除
        augumented_batch = augumented_batch[start_index:end_index - 1]

        # idが連続しているか確認&他の列が空欄でないか確認
        # csv化して""で囲まれた文字があったときうまく消す
        new_augumented_batch = ""
        for i, augumented_line in enumerate(augumented_batch.splitlines()):
            csv_augumented_line = csv_like_split(augumented_line)
            id = csv_augumented_line[0]
            if int(id) != current_number + i:
                print(augumented_line)
                raise ValueError("ID is not sequential.")
            if csv_augumented_line[-1] == "" or csv_augumented_line[-2] == "" or csv_augumented_line[-3] == "" or csv_augumented_line[-4] == "" or csv_augumented_line[-5] == "":
                print(augumented_line)
                raise ValueError("Some columns are empty.")
            if len(csv_augumented_line) != column_num:
                print(augumented_line)
                raise ValueError("irregular column num")
            
            # reviewとreplyContent以外が数値か確認
            float(csv_augumented_line[0])
            float(csv_augumented_line[2])
            float(csv_augumented_line[3])
            float(csv_augumented_line[4])
            match = re.match(r'(\d+)\s*days?\s*(\d{1,2}):?(\d{2})?:?(\d{2})?', csv_augumented_line[6])
            if not match:
                raise ValueError("Time format is invalid.")


            csv_augumented_line[1] = "\"" + csv_augumented_line[1] + "\""
            csv_augumented_line[5] = "\"" + csv_augumented_line[5] + "\""
            new_augumented_batch += ','.join(csv_augumented_line) + "\n"
            new_augumented_batch = new_augumented_batch.replace("\"\"", "\"").replace("\"\"", "\"")
            
        current_number = current_number + batch_size - 1
        augumented_data = augumented_data + new_augumented_batch

        print("Current number:")
        print(current_number)

        # 生成されたデータをCSVとして保存 (オプション)
        header = df.columns.tolist()
        with open("train_augmented_by_llama2.csv", "w") as f:
            write = csv.writer(f)
            write.writerow(header)
            f.write(augumented_data)
    except Exception as e:
        print(f"Error: {e}")
        continue