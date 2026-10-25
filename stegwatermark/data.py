"""Cover images. MIRFLICKR files are not 400×400; each one is fitted here."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class CoverFolder(Dataset):
    def __init__(self, root, size=400):
        self.size = size
        self.files = [
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in _SUFFIXES
        ]
        if not self.files:
            raise FileNotFoundError("no images under %s" % root)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        try:
            image = Image.open(path).convert("RGB")
            image = ImageOps.fit(image, (self.size, self.size))
            array = np.asarray(image, dtype=np.float32) / 255.0
        except Exception:
            array = np.zeros((self.size, self.size, 3), dtype=np.float32)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor


class SyntheticCovers(Dataset):
    """Random images so the training loop can be checked before MIRFLICKR is downloaded."""

    def __init__(self, n=64, size=400):
        self.n = n
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        return torch.rand(3, self.size, self.size, generator=generator)
