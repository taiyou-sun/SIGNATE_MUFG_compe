#!/usr/bin/env python3
"""
Jupyterノートブックで環境変数をロードするためのヘルパースクリプト。
APIキーを安全にロードするために、ノートブックの先頭に以下を追加してください：

```python
# 環境からAPIキーをロード
import sys
sys.path.append('..')  # 必要に応じてこのモジュールを見つけるために調整
from load_env_helper import load_api_keys

# これで.envファイルからAPIキーがロードされます
```
"""

import os
from dotenv import load_dotenv

def load_api_keys():
    """
    .envファイルからAPIキーをロードし、環境変数として設定します。
    これにより既存のノートブックを変更せずに動作させることができます。
    """
    # .envファイルから変数をロード
    load_dotenv()
    
    # OpenAI APIキーが設定されていることを確認
    if "OPENAI_API_KEY" in os.environ:
        print("✓ OpenAI APIキーが環境から読み込まれました")
    else:
        print("⚠️ OpenAI APIキーが環境に見つかりません")
    
    # Hugging Faceトークンが設定されていることを確認
    if "HUGGINGFACE_TOKEN" in os.environ:
        print("✓ Hugging Faceトークンが環境から読み込まれました")
    else:
        print("⚠️ Hugging Faceトークンが環境に見つかりません")
    
    return True

if __name__ == "__main__":
    # 関数をテスト
    load_api_keys()
