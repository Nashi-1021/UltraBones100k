import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BoneDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted(os.listdir(image_dir))

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        label_name = image_name.replace(
            ".png",
            "_label.png"
        )

        image = cv2.imread(
            os.path.join(self.image_dir, image_name),
            cv2.IMREAD_GRAYSCALE
        )

        label = cv2.imread(
            os.path.join(self.label_dir, label_name),
            cv2.IMREAD_GRAYSCALE
        )

        image = cv2.resize(image, (256,256))
        label = cv2.resize(label, (256,256))

        image = image.astype(np.float32)/255.0
        label = (label>0).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)

        return image, label
    
if __name__ == "__main__":

    dataset = BoneDataset(
        "split/train/images",
        "split/train/labels"
    )

    print("画像枚数:", len(dataset))

    image, label = dataset[0]

    print("画像サイズ:", image.shape)
    print("ラベルサイズ:", label.shape)

    print("画像の型:", image.dtype)
    print("ラベルの型:", label.dtype)

    print("画像の最小値:", image.min().item())
    print("画像の最大値:", image.max().item())

    print("ラベルの値:", torch.unique(label))