import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用デバイス:", device)


# =====================================
# Model
# ResNet34 + FPN
# =====================================

model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

model = model.to(device)


# =====================================
# 学習済みモデルの読み込み
# =====================================

model.load_state_dict(
    torch.load(
        "resnet_fpn_model.pth",
        map_location=device
    )
)

model.eval()

print("モデルの読み込み完了")


# =====================================
# テスト画像
# =====================================

image_path = "split/test/images/1015769.png"
label_path = "split/test/labels/1015769_label.png"

# =====================================
# 画像読み込み
# =====================================

image = Image.open(image_path).convert("L")
label = Image.open(label_path).convert("L")

# 256 × 256にリサイズ
image = image.resize((256, 256))
label = label.resize((256, 256))


# =====================================
# NumPy配列に変換
# =====================================

image_np = np.array(image).astype(np.float32) / 255.0

label_np = np.array(label)

# ラベルを0/1にする
label_np = (label_np > 0).astype(np.float32)


# =====================================
# Tensorに変換
# =====================================

image_tensor = torch.from_numpy(
    image_np
).unsqueeze(0).unsqueeze(0)

image_tensor = image_tensor.to(device)


print("入力画像サイズ:", image_tensor.shape)


# =====================================
# Prediction
# =====================================

with torch.no_grad():

    outputs = model(image_tensor)

    predictions = torch.sigmoid(outputs)


print("出力サイズ:", predictions.shape)

print(
    "予測最小値:",
    predictions.min().item()
)

print(
    "予測最大値:",
    predictions.max().item()
)


# =====================================
# 二値化
# =====================================

prediction_binary = (
    predictions > 0.5
).float()


# =====================================
# NumPyに変換
# =====================================

prediction_np = (
    prediction_binary
    .squeeze()
    .cpu()
    .numpy()
)


# =====================================
# 保存
# =====================================

plt.imsave(
    "prediction_resnet.png",
    prediction_np,
    cmap="gray"
)

print(
    "prediction_resnet.png を保存しました"
)


# =====================================
# 入力・正解・予測を比較
# =====================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)


axes[0].imshow(
    image_np,
    cmap="gray"
)

axes[0].set_title(
    "Input Image"
)

axes[0].axis("off")


axes[1].imshow(
    label_np,
    cmap="gray"
)

axes[1].set_title(
    "Ground Truth"
)

axes[1].axis("off")


axes[2].imshow(
    prediction_np,
    cmap="gray"
)

axes[2].set_title(
    "Prediction"
)

axes[2].axis("off")


plt.tight_layout()

plt.savefig(
    "comparison_resnet.png"
)

print(
    "comparison_resnet.png を保存しました"
)