import os
import torch
import numpy as np
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from dataset import BoneDataset


# =========================
# Dice係数
# =========================
def dice_score(pred, target, eps=1e-6):

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()

    dice = (
        2.0 * intersection + eps
    ) / (
        pred.sum() + target.sum() + eps
    )

    return dice


# =========================
# IoU
# =========================
def iou_score(pred, target, eps=1e-6):

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    intersection = (pred * target).sum()

    union = pred.sum() + target.sum() - intersection

    iou = (
        intersection + eps
    ) / (
        union + eps
    )

    return iou


# =========================
# Device
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用デバイス:", device)


# =========================
# Test Dataset
# =========================
test_dataset = BoneDataset(
    "split/test/images",
    "split/test/labels"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)

print("テスト画像数:", len(test_dataset))


# =========================
# Model
# =========================
model = smp.FPN(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=1,
    classes=1
)

model = model.to(device)


# =========================
# 学習済みモデル読み込み
# =========================
model.load_state_dict(
    torch.load(
        "resnet_fpn_model.pth",
        map_location=device
    )
)

model.eval()

print("モデルの読み込み完了")


# =========================
# Evaluation
# =========================
dice_scores = []
iou_scores = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        # モデル出力
        outputs = model(images)

        # Sigmoid
        predictions = torch.sigmoid(outputs)

        # 0.5で2値化
        predictions = (
            predictions > 0.3
        ).float()

        # バッチ内の画像ごとに評価
        for pred, label in zip(predictions, labels):

            dice = dice_score(pred, label)
            iou = iou_score(pred, label)

            dice_scores.append(
                dice.item()
            )

            iou_scores.append(
                iou.item()
            )


# =========================
# 結果
# =========================
average_dice = np.mean(dice_scores)
average_iou = np.mean(iou_scores)

print()
print("===== テスト結果 =====")
print(f"Average Dice: {average_dice:.4f}")
print(f"Average IoU : {average_iou:.4f}")

print()
print("Dice 最大値:", f"{max(dice_scores):.4f}")
print("Dice 最小値:", f"{min(dice_scores):.4f}")