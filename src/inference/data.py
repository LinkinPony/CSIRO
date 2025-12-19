from __future__ import annotations

import os
from typing import List, Tuple

import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from src.data.augmentations import build_eval_transform


def resolve_paths(input_path: str) -> Tuple[str, str]:
    """
    Resolve dataset_root + test_csv from a directory (containing test.csv) or a direct test.csv path.
    """
    if os.path.isdir(input_path):
        dataset_root = input_path
        test_csv = os.path.join(input_path, "test.csv")
    else:
        dataset_root = os.path.dirname(os.path.abspath(input_path))
        test_csv = input_path
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"test.csv not found at: {test_csv}")
    return dataset_root, test_csv


class TestImageDataset(Dataset):
    def __init__(self, image_paths: List[str], root_dir: str, transform: T.Compose) -> None:
        self.image_paths = image_paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        rel_path = self.image_paths[index]
        img_path = os.path.join(self.root_dir, rel_path)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, rel_path


def build_transforms(image_size: Tuple[int, int], mean: List[float], std: List[float]) -> T.Compose:
    """
    Build the evaluation transform used for inference.
    """
    return build_eval_transform(image_size=image_size, mean=mean, std=std)


