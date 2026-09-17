import torch
from torch.utils.data import DataLoader

from dataset import BoneDataset


# Dataset作成
train_dataset = BoneDataset(
    "split/train/images",
    "split/train/labels"
)

# DataLoader作成
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("バッチ数:", len(train_loader))

# 最初の1バッチだけ取得
for images, labels in train_loader:

    print("画像サイズ:", images.shape)
    print("ラベルサイズ:", labels.shape)

    print("画像型:", images.dtype)
    print("ラベル型:", labels.dtype)

    print("画像最小値:", images.min().item())
    print("画像最大値:", images.max().item())

    print("ラベル値:", torch.unique(labels))

    break