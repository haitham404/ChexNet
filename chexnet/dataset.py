"""ChestX-ray14 Dataset and Splitting utilities."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PNEUMONIA_IDX = CLASS_TO_IDX["Pneumonia"]


def encode_labels(x: str) -> list:
    """Convert finding labels string to multi-label binary vector."""
    y = [0] * len(CLASSES)
    labels = str(x).split("|")
    if "No Finding" in labels:
        return y
    for label in labels:
        if label in CLASS_TO_IDX:
            y[CLASS_TO_IDX[label]] = 1
    return y


def build_splits(data_dir: str, csv_path: str, train_csv: str = "train.csv",
                 val_csv: str = "val.csv", test_csv: str = "test.csv"):
    """Build train/val/test splits with patient-level stratification.
    
    Ensures no patient appears in both train and val sets,
    and no image overlap between train/val and test.
    """
    df = pd.read_csv(csv_path)
    df["labels"] = df["Finding Labels"].apply(encode_labels)

    train_val_list = open(os.path.join(data_dir, "train_val_list.txt")).read().splitlines()
    test_list = open(os.path.join(data_dir, "test_list.txt")).read().splitlines()

    train_val_df = df[df["Image Index"].isin(train_val_list)].copy()
    test_df = df[df["Image Index"].isin(test_list)].copy()

    train_val_patients = train_val_df["Patient ID"].unique()
    rng = np.random.default_rng(42)
    rng.shuffle(train_val_patients)
    split_idx = int(0.9 * len(train_val_patients))
    train_pts = train_val_patients[:split_idx]
    val_pts = train_val_patients[split_idx:]

    train_df = train_val_df[train_val_df["Patient ID"].isin(train_pts)].copy()
    val_df = train_val_df[train_val_df["Patient ID"].isin(val_pts)].copy()

    assert len(set(train_pts) & set(val_pts)) == 0, "Train/Val patient overlap!"
    assert len(set(train_val_list) & set(test_list)) == 0, "Train/Test image overlap!"

    pneu_train = train_df["Finding Labels"].str.contains("Pneumonia").sum()
    pneu_test = test_df["Finding Labels"].str.contains("Pneumonia").sum()

    print(f"Train: {len(train_df):>6} images | {len(train_pts):>5} patients")
    print(f"Val   : {len(val_df):>6} images | {len(val_pts):>5} patients")
    print(f"Test  : {len(test_df):>6} images")
    print(f"Pneumonia — train: {pneu_train} ({100 * pneu_train / len(train_df):.2f}%)  "
          f"test: {pneu_test} ({100 * pneu_test / len(test_df):.2f}%)")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_df, val_df, test_df


class ChestXrayDataset(Dataset):
    """PyTorch Dataset for ChestX-ray14 images."""

    def __init__(self, csv_path: str, image_dir: str, transform=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self._image_map = None

        self.labels = torch.tensor(
            self.df["labels"].apply(eval).tolist(), dtype=torch.float32
        )

    def _get_image_map(self):
        if self._image_map is None:
            m = {}
            for root, _, files in os.walk(self.image_dir):
                for f in files:
                    if f.endswith(".png") or f.endswith(".jpg"):
                        m[f] = os.path.join(root, f)
            self._image_map = m
        return self._image_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image Index"]
        path = self._get_image_map().get(img_name)
        if path is None:
            raise FileNotFoundError(f"Image not found: {img_name}")
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]