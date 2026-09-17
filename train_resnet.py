import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import time

from dataset import BoneDataset


# =====================================
# Dataset
# =====================================

train_dataset = BoneDataset(
    "split/train/images",
    "split/train/labels"
)

val_dataset = BoneDataset(
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
# BCE + Dice
# =====================================

dice_loss = smp.losses.DiceLoss(
    mode="binary",
    from_logits=True
)

bce_loss = nn.BCEWithLogitsLoss()


def criterion(outputs, labels):

    dice = dice_loss(outputs, labels)
    bce = bce_loss(outputs, labels)

    return dice + bce


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

def dice_score(outputs, labels):

    predictions = torch.sigmoid(outputs)

    predictions = (predictions > 0.5).float()

    intersection = (predictions * labels).sum()

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

    print(f"Epoch {epoch + 1}/{num_epochs}")

    model.train()

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        start_time = time.time()

        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Gradient reset
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update
        optimizer.step()

        running_loss += loss.item()

        elapsed = time.time() - start_time

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

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
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
    "resnet_fpn_model.pth"
)

print("モデルを保存しました")