import torch
import segmentation_models_pytorch as smp


# =====================================
# Device
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用デバイス:", device)


# =====================================
# ResNet34 + FPN
# =====================================

model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

model = model.to(device)


# =====================================
# ダミー画像
# =====================================

x = torch.randn(
    1,
    1,
    256,
    256
).to(device)


# =====================================
# Forward
# =====================================

with torch.no_grad():

    y = model(x)


print("入力サイズ:", x.shape)
print("出力サイズ:", y.shape)