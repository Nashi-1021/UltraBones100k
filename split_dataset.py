import os
import random
import shutil

# 元データ
image_dir = "UltrasoundImages"
label_dir = "Labels_full"

# 出力先
output_dir = "split"

# 比率
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# ファイル取得
images = sorted(os.listdir(image_dir))
random.seed(42)
random.shuffle(images)

n = len(images)

train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_files = images[:train_end]
val_files = images[train_end:val_end]
test_files = images[val_end:]

print("Train:", len(train_files))
print("Val:", len(val_files))
print("Test:", len(test_files))

# フォルダ作成
for split in ["train", "val", "test"]:
    os.makedirs(
        os.path.join(output_dir, split, "images"),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(output_dir, split, "labels"),
        exist_ok=True
    )

# コピー関数
def copy_files(file_list, split_name):

    for img_name in file_list:

        label_name = img_name.replace(
            ".png",
            "_label.png"
        )

        shutil.copy(
            os.path.join(image_dir, img_name),
            os.path.join(
                output_dir,
                split_name,
                "images",
                img_name
            )
        )

        shutil.copy(
            os.path.join(label_dir, label_name),
            os.path.join(
                output_dir,
                split_name,
                "labels",
                label_name
            )
        )

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print("完了")