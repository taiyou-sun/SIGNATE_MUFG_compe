# SIGNATE_MUFG_compe

## プロジェクト概要

このリポジトリは、SIGNATE × 三菱UFJフィナンシャルグループ主催のコンペティションのコードを管理するためのものです。自然言語処理（NLP）と金融データの分析を組み合わせたアプローチを用いています。

## プロジェクト構成

```
/
├── openai/               # OpenAI APIを使用したモデル関連のコード
│   ├── gpt-finetuning.py       # GPTモデルのファインチューニング
│   ├── gpt-finetuning-batchtest.py # バッチテスト用スクリプト
│   ├── gpt4o-pn.py        # GPT-4oモデルを使用した予測
│   └── gpt-4o_pred.py     # GPT-4oによる予測生成
├── scripts/              # 各種処理スクリプト
├── .env                  # 環境変数設定ファイル（APIキーなど）
└── load_env_helper.py    # 環境変数読み込み用ヘルパー
```

## セットアップと実行方法

### 1. 環境のセットアップ

必要なPythonパッケージをインストールします：

```bash
pip install pandas numpy openai python-dotenv tqdm
```

### 2. 環境変数の設定

`.env`ファイルに必要なAPIキーを設定します：

```
# OpenAI APIキー
OPENAI_API_KEY=あなたのopenai_apiキー

# Hugging Faceトークン
HUGGINGFACE_TOKEN=あなたのhuggingfaceトークン
```

### 3. スクリプトの実行

各スクリプトは以下のように実行できます：

```bash
# GPTファインチューニング
python openai/gpt-finetuning.py

# バッチテスト
python openai/gpt-finetuning-batchtest.py

# GPT-4oを使用した予測
python openai/gpt4o-pn.py

# 予測結果の生成
python openai/gpt-4o_pred.py
```

## Jupyter Notebookでの使用

Jupyterノートブックで環境変数を使用するには、以下のコードを最初のセルに追加します：

```python
# 環境からAPIキーをロード
import sys
sys.path.append('..')  # 必要に応じて調整
from load_env_helper import load_api_keys

# これで.envファイルからAPIキーがロードされます
```

これらのスクリプトは`dotenv`パッケージを使用して環境変数をロードします：

```python
from dotenv import load_dotenv
import os

# .envファイルから環境変数をロード
load_dotenv()

# 環境変数からAPIキーを使用
openai.api_key = os.getenv("OPENAI_API_KEY")
```

### セキュリティに関する考慮事項

- `.env`ファイルは`.gitignore`に追加して、機密キーが誤ってバージョン管理にコミットされないようにしてください。
- コードを共有する際は、APIキーが削除され、環境変数に置き換えられていることを確認してください。
- スクリプトやノートブックに直接APIキーをハードコードしないでください。