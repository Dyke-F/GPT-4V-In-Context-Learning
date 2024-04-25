import argparse
import ast
import os
import sys
from dataclasses import dataclass

# !{sys.executable} -m pip install timm
# !{sys.executable} -m pip install torchvision
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import timm
import torch
import tqdm
from datasets import load_dataset
from huggingface_hub import notebook_login
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from timm.data import create_transform, resolve_data_config
from timm.data.transforms_factory import create_transform
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

from PIL import Image
import torch
from transformers import AutoImageProcessor, ViTModel
from typing import NamedTuple



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

        return image, encoded_label, img_path.stem


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

        return image, encoded_label, img_path.stem


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

        return image, encoded_label, img_path.stem


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


@dataclass
class Conf:
    eval_batch_size = 1
    num_workers = 4 
    backend = BACKEND


class Features(NamedTuple):
    all_features: torch.tensor
    labels: list


def load_features(task, config):
    path = f"/mnt/bulk/dferber/ICL_REVISION/features/uni_{task}_features.npy"
    np_features = np.load(path)
    all_features = torch.tensor(np_features["vector"])
    all_features = all_features.to(config.backend)

    return Features(all_features=all_features, labels=np_features["name"])


def linear_probing(query_feature, features, task):

    all_features = features.all_features
    labels = features.labels

    similarities = F.cosine_similarity(query_feature, all_features, dim=1)
    top_index = torch.argmax(similarities)
    
    label = labels[top_index]

    if task == "MHIST":
        return [label.split("_")[0]]
    return [label.split("-")[0]]


def inference_uni_lin_probing(task: str, config: dataclass, uni_out_dim: int = 1024):
    
    features = load_features(task, config)
    _, _, dataset_cls, test_samples_paths = resolve_task(task)

    # check if not contaminated

    labels = list(features.labels)
    test_samples_paths2 = [s.stem for s in test_samples_paths]
    if task == "CRC100K":
        test_samples_paths2 = [s.replace("-TCGA", "") for s in test_samples_paths2]
    assert len(set(labels).intersection(set(test_samples_paths2))) == 0, "Overlap between test set and extracted features"

    uni = timm.create_model(
        "hf-hub:MahmoodLab/uni",
        pretrained=True,
        init_values=1e-5,
        dynamic_img_size=True,
    )
    img_transform = create_transform(
        **resolve_data_config(uni.pretrained_cfg, model=uni)
    )
    uni = uni.eval()

    for param in uni.parameters():
        param.requires_grad_(False)


    test_dataset = dataset_cls(
        image_paths=test_samples_paths,
        img_transform=img_transform,
        exclude_filenames=None,
    )

    assert all(Path(tx).resolve() in test_dataset.image_paths for tx in test_samples_paths), "Not all test samples are in the dataset"
    assert len(test_dataset.image_paths) == len(test_samples_paths), "There are more than the test samples in the test dataset"

    eval_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,  # Consider using a larger batch size if possible
        shuffle=False,
        num_workers=config.num_workers,
    )

    print(f"Loading model to backend: {config.backend.type}")
    uni = uni.to(config.backend)

    results = []


    with torch.inference_mode():
        for batch in eval_loader:
            if isinstance(batch, dict):
                images, labels = batch["image"].to(config.backend), batch["label"].to(
                    config.backend
                )
            else:
                images, labels, filenames = batch
                images, labels = images.to(config.backend), labels.to(config.backend)

            uni_outputs = uni(images)

            predicted_labels = linear_probing(query_feature=uni_outputs,
                features=features,
                task=task
            )
            
            true_labels = test_dataset.label_encoder.inverse_transform(
                labels.cpu().numpy()
            )

            results.extend(zip(filenames, true_labels, predicted_labels))

    results_df = pd.DataFrame(results, columns=['file_name', "true_label", "predicted_label"])
    results_df.to_csv(f"UNI_LINEAR_PROBING_{task}_eval.csv", index=False)



if __name__ == "__main__":

    config = Conf()

    TASKS = ["PCAM", "MHIST", "CRC100K"]

    for task in TASKS:
        
        inference_uni_lin_probing(
            task=task,
            config=config,
        )

