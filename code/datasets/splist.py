import os
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import zoom
import random


def generate_file_list(preprocessed_dir):
    """
    生成 train.txt(包含 `train_data` 下所有 .npz 文件）
    生成 val.txt / test.txt(包含 `val_data` 和 `test_data` 下所有 .h5 文件）
    """
    train_data_dir = os.path.join(preprocessed_dir, "train_data")
    val_data_dir = os.path.join(preprocessed_dir, "val_data")
    test_data_dir = os.path.join(preprocessed_dir, "test_data")

    # **检查目录是否存在**
    if not os.path.exists(train_data_dir):
        raise ValueError(f"❌ 预处理目录不存在: {train_data_dir}，请先运行数据预处理！")

    # ✅ **生成 `train.txt`（从 train_data 读取所有 `.npz` 文件）**
    train_files = sorted([f[:-4] for f in os.listdir(train_data_dir) if f.endswith(".npz")])
    with open(os.path.join(preprocessed_dir, "train.txt"), "w") as f:
        for item in train_files:
            f.write(f"{item}\n")
    print(f"✅ 生成 train.txt,共包含 {len(train_files)} 个切片")

    # ✅ **生成 `val.txt`（从 val_data 读取所有 `.h5` 文件）**
    val_files = sorted([f[:-3] for f in os.listdir(val_data_dir) if f.endswith(".h5")])
    with open(os.path.join(preprocessed_dir, "val.txt"), "w") as f:
        for item in val_files:
            f.write(f"{item}\n")
    print(f"✅ 生成 val.txt,共包含 {len(val_files)} 个病例")

    # ✅ **生成 `test.txt`（从 test_data 读取所有 `.h5` 文件）**
    test_files = sorted([f[:-3] for f in os.listdir(test_data_dir) if f.endswith(".h5")])
    with open(os.path.join(preprocessed_dir, "test.txt"), "w") as f:
        for item in test_files:
            f.write(f"{item}\n")
    print(f"✅ 生成 test.txt,共包含 {len(test_files)} 个病例")

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    nnunet_root = '/home/xlk/SSL/CPC-SAM/data/brain'
    images_dir = os.path.join(nnunet_root, 'imagesTr')
    labels_dir = os.path.join(nnunet_root, 'labelsTr')
    out_root = '/home/xlk/SSL/CPC-SAM/data/brain/preprocessed'
    
    preprocessed_dir = "/home/xlk/SSL/CPC-SAM/data/brain/preprocessed"
    generate_file_list(preprocessed_dir)
    
    for split in ['train', 'val', 'test']:
        file_list = os.path.join(out_root, f"{split}.txt")
        out_path = os.path.join(out_root, f"{split}_data")
        mode = "train" if split == "train" else "test"

