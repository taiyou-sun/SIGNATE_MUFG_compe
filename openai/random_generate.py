import pandas as pd

# CSVファイルのパス
file_path = '../train.csv'

# データを読み込む
df = pd.read_csv(file_path)

# カスタマイズされた層化抽出関数
def custom_stratified_sample(df, stratify_col, sample_sizes):
    samples = []
    for value, size in sample_sizes.items():
        # 特定の score に対して抽出
        group = df[df[stratify_col] == value]
        samples.append(group.sample(min(len(group), size)))
    return pd.concat(samples)

# サンプルサイズの指定: {score値: 取得したい行数}
sample_sizes = {
    4: 15,  # score = 4 に対して多めにサンプルを取得
    3: 10,  # score = 3 に対して少し少なめにサンプルを取得
    2: 20,  # score = 2 に対して少し少なめにサンプルを取得
    1: 20,   # score = 1 に対して少し少なめにサンプルを取得
    0: 30,
}

# カスタムサンプリング実行
sampled_df = custom_stratified_sample(df, 'score', sample_sizes)

# 検証データの割合
validation_ratio = 0.2

# 検証データと訓練データに分割
validation_df = sampled_df.groupby('score').sample(frac=validation_ratio, random_state=42)
train_df = sampled_df.drop(validation_df.index)

# 新しいCSVファイルとして保存
train_output_path = 'train_sampled.csv'
validation_output_path = 'validation_sampled.csv'
sampled_df.to_csv(train_output_path, index=False)
validation_df.to_csv(validation_output_path, index=False)

print(f"訓練データが {train_output_path} に保存されました。")
print(f"検証データが {validation_output_path} に保存されました。")