import os
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset
from scipy.ndimage import distance_transform_edt


class BoneSkeletonDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted(
            os.listdir(image_dir)
        )

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        # -------------------------
        # ファイル名
        # -------------------------
        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        # 1015769.png
        # ↓
        # 1015769_label.png
        base_name = os.path.splitext(
            image_name
        )[0]

        label_name = (
            base_name + "_label.png"
        )

        label_path = os.path.join(
            self.label_dir,
            label_name
        )

        # -------------------------
        # 画像読み込み
        # -------------------------
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        label = cv2.imread(
            label_path,
            cv2.IMREAD_GRAYSCALE
        )

        # -------------------------
        # Resize
        # -------------------------
        image = cv2.resize(
            image,
            (256, 256),
            interpolation=cv2.INTER_LINEAR
        )

        label = cv2.resize(
            label,
            (256, 256),
            interpolation=cv2.INTER_NEAREST
        )

        # -------------------------
        # Image normalization
        # -------------------------
        image = (
            image.astype(np.float32)
            / 255.0
        )

        # -------------------------
        # 元のラベルを0/1にする
        # -------------------------
        label_binary = (label > 0).astype(np.float32)

        # -------------------------
        # Distance Transform
        # -------------------------
        distance_map = distance_transform_edt(
            ~(label_binary.astype(bool))
        )

        # -------------------------
        # Skeleton
        # 元ラベルから1pixel程度の周辺を作成
        # -------------------------
        skeleton = np.zeros_like(
            distance_map,
            dtype=np.float32
        )

        skeleton[
            distance_map <= 1
        ] = 1.0

        # -------------------------
        # Tensor
        # -------------------------
        image = torch.from_numpy(
            image
        ).unsqueeze(0)

        label_binary = torch.from_numpy(
            label_binary
        ).unsqueeze(0)

        skeleton = torch.from_numpy(
            skeleton
        ).unsqueeze(0)

        return image, label_binary, skeleton
    
if __name__ == "__main__":

    dataset = BoneSkeletonDataset(
        "split/train/images",
        "split/train/labels"
    )

    image, label, skeleton = dataset[0]

    print(
        "データ数:",
        len(dataset)
    )

    print(
        "画像サイズ:",
        image.shape
    )

    print(
        "ラベルサイズ:",
        label.shape
    )

    print(
        "Skeletonサイズ:",
        skeleton.shape
    )

    print(
        "画像値:",
        image.min().item(),
        image.max().item()
    )

    print(
        "ラベル値:",
        torch.unique(label)
    )

    print(
        "Skeleton値:",
        torch.unique(skeleton)
    )

    print(
        "ラベル白画素数:",
        label.sum().item()
    )

    print(
        "Skeleton白画素数:",
        skeleton.sum().item()
    )
    
import matplotlib.pyplot as plt

# -------------------------
# 画像として保存
# -------------------------

plt.imsave(
    "skeleton_debug.png",
    skeleton.squeeze(0).numpy(),
    cmap="gray"
)

plt.imsave(
    "label_debug.png",
    label.squeeze(0).numpy(),
    cmap="gray"
)

plt.imsave(
    "input_debug.png",
    image.squeeze(0).numpy(),
    cmap="gray"
)

print("debug画像を保存しました")