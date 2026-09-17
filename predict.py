import cv2
import torch
import numpy as np

from unet import UNet

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用デバイス:", device)

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "unet_model.pth",
        map_location=device
    )
)

model.eval()

# テスト画像を読み込む
image = cv2.imread(
    "split/test/images/1015769.png",
    cv2.IMREAD_GRAYSCALE
)

# 正解ラベルを読み込む
label = cv2.imread(
    "split/test/labels/1015769_label.png",
    cv2.IMREAD_GRAYSCALE
)

# 学習時と同じサイズに変更
image = cv2.resize(image, (256, 256))
label = cv2.resize(label, (256,256))

cv2.imwrite("input.png", image)
cv2.imwrite("label.png", label)

# 0～1に正規化
image = image.astype(np.float32) / 255.0

# Tensorへ変換
image = torch.from_numpy(image)

# チャネル次元を追加
image = image.unsqueeze(0)

# バッチ次元を追加
image = image.unsqueeze(0)

# GPU/CPUへ送る
image = image.to(device)

print("モデルの読み込み完了")
print("入力画像サイズ:", image.shape)

# 推論
with torch.no_grad():

    output = model(image)

print("出力サイズ:", output.shape)

print("Logit最小値:", output.min().item())
print("Logit最大値:", output.max().item())

# 確率へ変換
prediction = torch.sigmoid(output)

print(
    "予測最小値:",
    prediction.min().item()
)

print(
    "予測最大値:",
    prediction.max().item()
)

prediction = (prediction > 0.5).float()


# Tensor → NumPy
prediction = prediction.squeeze().cpu().numpy()

# 0,1 → 0,255
prediction = (prediction * 255).astype(np.uint8)

# 保存
cv2.imwrite("prediction.png", prediction)

print("prediction.png を保存しました")

print("予測ラベル:")