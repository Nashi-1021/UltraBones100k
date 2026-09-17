import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from dataset import BoneDataset


# =========================
# Dice計算
# =========================
def dice_score(pred, target, eps=1e-6):

    pred = pred.float()
    target = target.float()

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
# Validation Dataset
# =========================
val_dataset = BoneDataset(
    "split/val/images",
    "split/val/labels"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)

print("Validation画像数:", len(val_dataset))


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
# Threshold候補
# =========================
thresholds = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9
]


# 各thresholdのDice保存
threshold_scores = {
    threshold: []
    for threshold in thresholds
}


# =========================
# Validation
# =========================
with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        # モデル出力
        outputs = model(images)

        # Sigmoid
        probabilities = torch.sigmoid(outputs)

        # thresholdごとに評価
        for threshold in thresholds:

            predictions = (
                probabilities > threshold
            ).float()

            for pred, label in zip(
                predictions,
                labels
            ):

                score = dice_score(
                    pred,
                    label
                )

                threshold_scores[
                    threshold
                ].append(score)


# =========================
# 結果表示
# =========================
print("\n===== Threshold評価結果 =====")

best_threshold = None
best_dice = 0


for threshold in thresholds:

    scores = threshold_scores[
        threshold
    ]

    average_dice = sum(scores) / len(scores)

    print(
        f"Threshold: {threshold:.1f} "
        f"Average Dice: {average_dice:.4f}"
    )

    if average_dice > best_dice:

        best_dice = average_dice
        best_threshold = threshold


print("\n===== 最適Threshold =====")

print(
    f"Best Threshold: {best_threshold}"
)

print(
    f"Best Validation Dice: {best_dice:.4f}"
)