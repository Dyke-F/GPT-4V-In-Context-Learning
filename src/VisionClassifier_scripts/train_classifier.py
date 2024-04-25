import argparse
import ast
import os
import sys
from dataclasses import dataclass

# !{sys.executable} -m pip install timm
# !{sys.executable} -m pip install torchvision
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import tqdm
from datasets import load_dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timm.data import create_transform, resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm
import joblib


# remove the test samples from the train / validation datasets
crc100k_test_path = "/mnt/bulk/dferber/ICL_REVISION/data/CRC100K/knn"
mhist_test_path = "/mnt/bulk/dferber/ICL_REVISION/data/MHIST/knn"
pcam_test_path = "/mnt/bulk/dferber/ICL_REVISION/data/PCam"

crc100k_test = list(Path(crc100k_test_path).glob("**/*.csv"))
mhist_test = list(Path(mhist_test_path).glob("**/*.csv"))
pcam_test = list(Path(pcam_test_path).glob("**/*.csv"))

# crc100k test samples
crc100k_test = [pd.read_csv(csv) for csv in crc100k_test]
crc100k_test_samples = set(sample for csv in crc100k_test for sample in csv["fname"])

# mhist test samples
mhist_test = [pd.read_csv(csv) for csv in mhist_test]
mhist_test_samples = set(sample for csv in mhist_test for sample in csv["fname"])

# pcam test samples
pcam_test = [pd.read_csv(csv) for csv in pcam_test]
pcam_test_samples = set(sample for csv in pcam_test for sample in csv["fname"])


print(f"crc100k test samples: {len(crc100k_test_samples)}")
print(f"mhist test samples: {len(mhist_test_samples)}")
print(f"pcam test samples: {len(pcam_test_samples)}")

pcam_test_sample_paths = [
    os.path.join("/mnt/bulk/dferber/ICL_REVISION/images/PCam/full_imgs", f"{s}.png")
    for s in pcam_test_samples
]

pcam_test_sample_paths = [Path(p) for p in pcam_test_sample_paths]

mhist_test_sample_paths = [
    os.path.join("/mnt/bulk/dferber/ICL_REVISION/images/MHIST/images", f"{s}.png")
    for s in mhist_test_samples
]

mhist_test_sample_paths = [Path(p) for p in mhist_test_sample_paths]


crc100k_test_samples_paths = []
for s in crc100k_test_samples:
    label = s.split("-")[0]
    folder_name = label
    path = os.path.join(
        f"/mnt/bulk/dferber/ICL_REVISION/images/CRC100K/CRC-VAL-HE-7K-png/{label}",
        f"{s}.png",
    )
    crc100k_test_samples_paths.append(path)

crc100k_test_samples_paths = [Path(p) for p in crc100k_test_samples_paths]


# FIXME: paths on Uranus
IMAGE_DIRS = {
    "PCAM": Path("/mnt/bulk/dferber/ICL_REVISION/images/PCam/full_imgs"),
    "MHIST": Path("/mnt/bulk/dferber/ICL_REVISION/images/MHIST/images"),
    "CRC100K": Path(
        "/mnt/bulk/dferber/ICL_REVISION/images/CRC100K/NCT-CRC-HE-100K-png"
    ),
}

# CUDA > MPS > CPU
BACKEND = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not BACKEND.type == "cuda":
    BACKEND = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass
class Conf:
    train_batch_size: int = 32
    valid_batch_size: int = 32
    lr: float = 0.0001
    num_epochs: int = 1
    num_workers: int = 0  # FIXME: uranus
    backend: torch.device = BACKEND
    valid_size: float = 0.1
    random_seed: float = 42


# load data and convert
class MHISTDataset(Dataset):
    def __init__(self, image_paths, img_transform, exclude_filenames=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            img_transform (callable): Transformation to be applied on a sample.
            exclude_filenames (list, optional): List of filenames to exclude from the dataset.
        """
        self.img_transform = img_transform

        if exclude_filenames is not None:
            exclude_filenames = set(exclude_filenames)
            self.image_paths = [p for p in image_paths if p not in exclude_filenames]
        else:
            self.image_paths = image_paths

        self.labels = [p.stem.split("_")[0] for p in self.image_paths]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    @property
    def num_labels(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        label = img_path.stem.split("_")[0]
        encoded_label = self.label_encoder.transform([label])[0]

        return image, encoded_label


class PCAMDataset(Dataset):
    def __init__(self, image_paths, img_transform, exclude_filenames=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            img_transform (callable): Transformation to be applied on a sample.
            exclude_filenames (list, optional): List of filenames to exclude from the dataset.
        """
        self.img_transform = img_transform

        if exclude_filenames is not None:
            exclude_filenames = set(exclude_filenames)
            self.image_paths = [p for p in image_paths if p not in exclude_filenames]
        else:
            self.image_paths = image_paths
        self.labels = [p.stem.split("-")[0] for p in self.image_paths]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    @property
    def num_labels(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        label = img_path.stem.split("-")[0]
        encoded_label = self.label_encoder.transform([label])[0]

        return image, encoded_label


class CRC100KDataset(Dataset):
    def __init__(self, image_paths, img_transform, exclude_filenames=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            img_transform (callable): Transformation to be applied on a sample.
            exclude_filenames (list, optional): List of filenames to exclude from the dataset.
                Should not happen because we only load the train samples
        """
        self.img_transform = img_transform

        if exclude_filenames is not None:
            exclude_filenames = set(exclude_filenames)
            self.image_paths = [p for p in image_paths if p not in exclude_filenames]
        else:
            self.image_paths = image_paths
        self.labels = [p.stem.split("-")[0] for p in self.image_paths]

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)

    @property
    def num_labels(self):
        return len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        label = img_path.stem.split("-")[0]
        encoded_label = self.label_encoder.transform([label])[0]

        return image, encoded_label


# FIXME: use as namedtuple
def resolve_task(task):
    if task == "PCAM":
        return "PCAM", IMAGE_DIRS["PCAM"], PCAMDataset, pcam_test_sample_paths
    elif task == "MHIST":
        return "MHIST", IMAGE_DIRS["MHIST"], MHISTDataset, mhist_test_sample_paths
    elif task == "CRC100K":
        return (
            "CRC100K",
            IMAGE_DIRS["CRC100K"],
            CRC100KDataset,
            crc100k_test_samples_paths,
        )


TASKS = ["PCAM", "MHIST", "CRC100K"]


def train(vision_model: str, task: str, config: dataclass):
    img_config = resolve_data_config({}, model=vision_model)
    img_transform = create_transform(**img_config)

    task_name, image_root_dir, dataset_cls, to_exclude = resolve_task(task)

    image_paths = list(Path(image_root_dir).rglob("*.png"))
    train_file_paths, valid_file_paths = train_test_split(
        image_paths, test_size=config.valid_size, random_state=config.random_seed
    )

    train_file_len_pre = len(train_file_paths)
    valid_file_len_pre = len(valid_file_paths)

    train_dataset = dataset_cls(
        image_paths=train_file_paths,
        img_transform=img_transform,
        exclude_filenames=to_exclude,
    )
    valid_dataset = dataset_cls(
        image_paths=valid_file_paths,
        img_transform=img_transform,
        exclude_filenames=to_exclude,
    )

    assert not any(
        tx in train_dataset.image_paths for tx in to_exclude
    ), "Data Leakage from test set into train set"

    assert not any(
        tx in valid_dataset.image_paths for tx in to_exclude
    ), "Data Leakage from test set into valid set"

    if not task == "CRC100K":
        # we havent excluded any samples yet for CRC100K
        assert len(train_dataset) + len(
            valid_dataset
        ) == train_file_len_pre + valid_file_len_pre - len(to_exclude)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config.train_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    num_classes = train_dataset.num_labels

    model = timm.create_model(vision_model, pretrained=True, num_classes=num_classes)
    model = model.to(config.backend)
    print(
        "Model # Parameters: ",
        sum([p.numel() for p in model.parameters() if p.requires_grad]),
    )
    print(f"Loading model to backend: {config.backend.type}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True, min_lr=0.00001
    )

    best_val_acc = 0.0
    for epoch in range(config.num_epochs):
        model = model.train()

        for (
            batch
        ) in (
            train_dataloader
        ):  # to handle native PyTorch Dataset and HF-Dataset at same time (deprecated)
            if isinstance(batch, dict):
                images, labels = batch["image"].to(config.backend), batch["label"].to(
                    config.backend
                )
            else:
                images, labels = batch
                images, labels = images.to(config.backend), labels.to(config.backend)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            sys.stdout.write(f"\rEpoch {epoch} | Train Loss: {loss.item()}")
            sys.stdout.flush()

        val_loss, val_acc = validate(model, valid_dataloader, criterion, config)
        print(f"\nValidation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{vision_model}_{task_name}_{config.num_epochs}epochs_best.pth")
            print(
                f"Checkpoint saved: Epoch {epoch} with Validation Accuracy {val_acc:.4f}"
            )

    sys.stdout.flush()


def validate(model, valid_dataloader, criterion, config):
    model = model.eval()
    val_loss = 0.0
    correct = 0

    with torch.inference_mode():
        for (
            batch
        ) in (
            valid_dataloader
        ):  # to handle native PyTorch Dataset and HF-Dataset at same time
            if isinstance(batch, dict):
                images, labels = batch["image"].to(config.backend), batch["label"].to(
                    config.backend
                )
            else:
                images, labels = batch
                images, labels = images.to(config.backend), labels.to(config.backend)

            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

    val_loss /= len(valid_dataloader)
    val_accuracy = correct / len(valid_dataloader.dataset) * 100

    return val_loss, val_accuracy


config = Conf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vision model training.")
    parser.add_argument(
        "model",
        choices=["resnet18", "resnet50", "tiny_vit_21m_224", "vit_small_patch8_224"],
        help="Specify the vision model to use.",
    )
    parser.add_argument(
        "num_epochs",
        type=int,
        help="Number of epochs to train",
    )

    args = parser.parse_args()

    config = Conf(num_epochs=args.num_epochs)

    print(config)

    VISION_MODEL = args.model

    print(f"Running {VISION_MODEL}...")

    for task in TASKS:
        print(f"Running task {task} ...")

        train(VISION_MODEL, task, config)

        print(f"### Finished {task}. ###")

    print(f"### Finished {VISION_MODEL}. ###")

print(f"### Finished everything. ###")
