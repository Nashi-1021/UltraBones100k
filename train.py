import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import time

from dataset import BoneDataset
from unet import UNet

# Dice係数
def dice_score(pred, target):

    pred = torch.sigmoid(pred)

    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()

    dice = (
        2 * intersection
    ) / (
        pred.sum() + target.sum() + 1e-8
    )

    return dice

# Dice Loss
class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        pred = torch.sigmoid(pred)

        intersection = (pred * target).sum()

        dice = (
            2 * intersection + 1e-8
        ) / (
            pred.sum() +
            target.sum() +
            1e-8
        )

        return 1 - dice
    
# BCE + Dice Loss
class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()

        self.dice = DiceLoss()

    def forward(self, pred, target):

        bce_loss = self.bce(
            pred,
            target
        )

        dice_loss = self.dice(
            pred,
            target
        )

        return bce_loss + dice_loss

# Dataset
train_dataset = BoneDataset(
    "split/train/images",
    "split/train/labels"
)

val_dataset = BoneDataset(
    "split/val/images",
    "split/val/labels"
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False
)

# Device設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用デバイス:", device)

# Model
model = UNet().to(device)

# Loss
criterion = BCEDiceLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# 学習回数
num_epochs = 30

for epoch in range(num_epochs):

    print(f"Epoch {epoch+1}/{num_epochs}")

    model.train()

    # エポック全体のLoss
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):

        start_time = time.time()

        images = images.to(device)
        labels = labels.to(device)

        # 順伝播（Forward）
        outputs = model(images)

        # Loss計算
        loss = criterion(outputs, labels)

        # 勾配リセット
        optimizer.zero_grad()

        # 誤差逆伝播
        loss.backward()

        # 重み更新
        optimizer.step()

        # Lossを足し合わせる
        running_loss += loss.item()

        elapsed = time.time() - start_time

        if batch_idx < 5:
            print(f"Batch {batch_idx + 1}: {elapsed:.3f} 秒")

    # 平均Lossを計算
    epoch_loss = running_loss / len(train_loader)

    print(f"Train Loss: {epoch_loss:.4f}")
    
    # Validation
    model.eval()

    val_loss = 0.0
    val_dice = 0.0
    
    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()
            
            dice = dice_score(outputs, labels)

            val_dice += dice.item()
            
    # 平均Validation Loss
    val_loss /= len(val_loader)

    print(f"Val Loss: {val_loss:.4f}")
    
    val_dice /= len(val_loader)

    print(f"Dice: {val_dice:.4f}")
    
# モデルを保存
torch.save(
    model.state_dict(),
    "unet_model.pth"
)

print("モデルを保存しました")