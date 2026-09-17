import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp

from dataset import BoneDataset


# =========================
# Dice
# =========================
def dice_score(pred, target, eps=1e-6):

    pred = pred.reshape(-1).float()
    target = target.reshape(-1).float()

    intersection = (pred * target).sum()

    dice = (
        2 * intersection + eps
    ) / (
        pred.sum() + target.sum() + eps
    )

    return dice.item()


# =========================
# Device
# =========================
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用デバイス:", device)


# =========================
# Dataset
# =========================
test_dataset = BoneDataset(
    "split/test/images",
    "split/test/labels"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
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

model.load_state_dict(
    torch.load(
        "resnet_fpn_model.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("モデルの読み込み完了")


# =========================
# Prediction
# =========================
results = []

threshold = 0.3

with torch.no_grad():

    for index, (images, labels) in enumerate(test_loader):

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        predictions = (
            probabilities > threshold
        ).float()

        dice = dice_score(
            predictions,
            labels
        )

        # CPUへ移動して保存
        image_np = (
            images[0, 0]
            .cpu()
            .numpy()
        )

        label_np = (
            labels[0, 0]
            .cpu()
            .numpy()
        )

        prediction_np = (
            predictions[0, 0]
            .cpu()
            .numpy()
        )

        results.append({
            "index": index,
            "dice": dice,
            "image": image_np,
            "label": label_np,
            "prediction": prediction_np
        })


# =========================
# Dice順に並べる
# =========================
results.sort(
    key=lambda x: x["dice"]
)


# 最低Dice
worst = results[0]

# 中央付近のDice
middle = results[len(results) // 2]

# 最高Dice
best = results[-1]


# =========================
# 保存関数
# =========================
def save_comparison(result, name):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(
        result["image"],
        cmap="gray"
    )

    axes[0].set_title(
        f"Input\nDice: {result['dice']:.4f}"
    )

    axes[1].imshow(
        result["label"],
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        result["prediction"],
        cmap="gray"
    )

    axes[2].set_title(
        "Prediction"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    filename = (
        f"{name}_dice_"
        f"{result['dice']:.4f}.png"
    )

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    print(
        f"{filename} を保存しました"
    )


# =========================
# 保存
# =========================
save_comparison(
    best,
    "best"
)

save_comparison(
    middle,
    "middle"
)

save_comparison(
    worst,
    "worst"
)


# =========================
# 結果表示
# =========================
print()
print("===== 分析結果 =====")

print(
    f"Best   Index: {best['index']} "
    f"Dice: {best['dice']:.4f}"
)

print(
    f"Middle Index: {middle['index']} "
    f"Dice: {middle['dice']:.4f}"
)

print(
    f"Worst  Index: {worst['index']} "
    f"Dice: {worst['dice']:.4f}"
)