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


def build_transforms(
    image_size: Tuple[int, int],
    mean: List[float],
    std: List[float],
    *,
    hflip: bool = False,
    vflip: bool = False,
) -> T.Compose:
    """
    Build the evaluation transform used for inference.
    """
    if not bool(hflip) and not bool(vflip):
        return build_eval_transform(image_size=image_size, mean=mean, std=std)

    # Deterministic flip TTA view (applied on PIL before resize).
    # We intentionally do NOT add vertical flips / rotations here since the default
    # AugMix training policy in this repo uses hflip only (plus Resize).
    #
    # Note: vflip is supported as an explicit opt-in (tta_vflip) for users who want it.
    pre: List[object] = []
    if bool(vflip):
        pre.append(T.RandomVerticalFlip(p=1.0))
    if bool(hflip):
        pre.append(T.RandomHorizontalFlip(p=1.0))
    return T.Compose(
        [
            *pre,
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


