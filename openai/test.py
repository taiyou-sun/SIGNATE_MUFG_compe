import torch
import torch.nn as nn
import torch.optim as optim

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデルの定義
class OrdinalRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(OrdinalRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.thresholds = nn.Parameter(torch.randn(1))  # 閾値は1つだけ

    def forward(self, x):
        logits = self.linear(x)
        return logits

# 損失関数の定義
def ordinal_regression_loss(logits, thresholds, targets):
    # 誤差の計算
    error = logits - thresholds - targets.float()

    # 損失の計算 (例: MSE Loss)
    loss = torch.mean(error ** 2)

    return loss

# モデル、オプティマイザ、損失関数のインスタンス化
model = OrdinalRegressionModel(input_size=10).to(device)  # input_size は適宜変更
optimizer = optim.Adam(model.parameters())
criterion = ordinal_regression_loss

# ダミーデータの作成
input_data = torch.randn(100, 10).to(device)  # input_size に合わせて変更
target_labels = torch.randint(0, 5, (100,)).to(device)  # 0~4のラベル

# 学習ループ
epochs = 100
for epoch in range(epochs):
    # 予測
    logits = model(input_data)

    # 損失の計算
    loss = criterion(logits, model.thresholds, target_labels)

    # 勾配の計算とパラメータの更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 損失の出力
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 予測例
with torch.no_grad():
    predicted_values = model(input_data[0:5])
    print(f"Predicted Values: {predicted_values.squeeze().tolist()}")