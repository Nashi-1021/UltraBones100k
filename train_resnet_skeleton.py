import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import time

from dataset_skeleton import BoneSkeletonDataset


# =====================================
# Dataset
# =====================================

train_dataset = BoneSkeletonDataset(
    "split/train/images",
    "split/train/labels"
)

val_dataset = BoneSkeletonDataset(
    "split/val/images",
    "split/val/labels"
)


# =====================================
# DataLoader
# =====================================

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=4,
    shuffle=False
)


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
    encoder_weights="imagenet",
    in_channels=1,
    classes=1
)

model = model.to(device)


# =====================================
# Loss
# BCE + Dice + Skeleton
# =====================================

dice_loss = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)

bce_loss = nn.BCEWithLogitsLoss()

skeleton_weight = 0.1


def criterion(outputs, labels, skeletons):

    # Dice Loss
    dice = dice_loss(
        outputs,
        labels
    )

    # BCE Loss
    bce = bce_loss(
        outputs,
        labels
    )

    # -----------------------------
    # Skeleton項
    # -----------------------------
    probabilities = torch.sigmoid(outputs)

    batch_size = outputs.shape[0]

    skeleton_sum = (
        skeletons
        .view(batch_size, -1)
        .sum(dim=1)
    )

    skeleton_score = (
        (probabilities * skeletons)
        .view(batch_size, -1)
        .sum(dim=1)
        /
        (skeleton_sum + 1e-8)
    ).mean()

    # GitHub版と同じ考え方
    loss = (
        dice
        + bce
        - skeleton_weight * skeleton_score
    )

    return loss


# =====================================
# Optimizer
# =====================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# =====================================
# Dice Score
# =====================================

def dice_score(
    outputs,
    labels,
    threshold=0.5
):

    predictions = torch.sigmoid(
        outputs
    )

    predictions = (
        predictions > threshold
    ).float()

    intersection = (
        predictions * labels
    ).sum()

    dice = (
        2.0 * intersection
        + 1e-6
    ) / (
        predictions.sum()
        + labels.sum()
        + 1e-6
    )

    return dice.item()


# =====================================
# Training
# =====================================

num_epochs = 10


for epoch in range(num_epochs):

    print(
        f"Epoch {epoch + 1}/{num_epochs}"
    )

    model.train()

    running_loss = 0.0

    for batch_idx, (
        images,
        labels,
        skeletons
    ) in enumerate(train_loader):

        start_time = time.time()

        images = images.to(device)
        labels = labels.to(device)
        skeletons = skeletons.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels,
            skeletons
        )

        # Gradient reset
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update
        optimizer.step()

        running_loss += loss.item()

        elapsed = (
            time.time()
            - start_time
        )

        if batch_idx < 5:

            print(
                f"Batch {batch_idx + 1}: "
                f"{elapsed:.3f} 秒"
            )

    epoch_loss = (
        running_loss
        / len(train_loader)
    )

    print(
        f"Train Loss: "
        f"{epoch_loss:.4f}"
    )


    # =================================
    # Validation
    # =================================

    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    val_batches = 0

    with torch.no_grad():

        for (
            images,
            labels,
            skeletons
        ) in val_loader:

            images = images.to(device)
            labels = labels.to(device)
            skeletons = skeletons.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
                skeletons
            )

            val_loss += loss.item()

            val_dice += dice_score(
                outputs,
                labels
            )

            val_batches += 1

    val_loss /= len(val_loader)

    val_dice /= val_batches

    print(
        f"Val Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Dice: "
        f"{val_dice:.4f}"
    )


# =====================================
# Save Model
# =====================================

torch.save(
    model.state_dict(),
    "resnet_fpn_skeleton_model.pth"
)

print("モデルを保存しました")