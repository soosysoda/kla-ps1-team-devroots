import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def add_speckle_noise(img, severity=0.1):
    noise = np.random.randn(*img.shape) * severity
    noisy = img + img * noise
    return np.clip(noisy, 0.0, 1.2)


def add_gaussian_noise(img, severity=0.05):
    noise = np.random.randn(*img.shape) * severity
    return np.clip(img + noise, 0.0, 1.0)


def random_blur(img, max_sigma=1.0):
    from scipy.ndimage import gaussian_filter
    sigma = random.uniform(0.0, max_sigma)
    if sigma > 0:
        img = gaussian_filter(img, sigma=sigma)
    return img


def _normalize(arr):
    arr = arr.astype(np.float32)
    if arr.max() <= 2.0 and arr.min() >= -0.5:
        return arr
    return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)


def _load_npy_image(path):
    arr = np.load(path)
    arr = np.squeeze(arr)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    else:
        arr = _normalize(arr)
    return arr


class RestorationDataset(Dataset):
    def __init__(self, root_dir, split="train", augment=True, target_size=None):
        self.root = Path(root_dir)
        self.augment = augment and split == "train"
        self.target_size = target_size

        degraded_dir = self.root / "degraded"
        clean_dir = self.root / "clean"
        stacked_degraded = self.root / "degraded.npy"
        stacked_clean = self.root / "clean.npy"

        if stacked_degraded.exists() and stacked_clean.exists():
            self.mode = "stacked"
            self._degraded_arr = np.load(stacked_degraded, mmap_mode="r")
            self._clean_arr = np.load(stacked_clean, mmap_mode="r")
            assert len(self._degraded_arr) == len(self._clean_arr), (
                f"Mismatched counts: degraded.npy has {len(self._degraded_arr)} "
                f"images, clean.npy has {len(self._clean_arr)}"
            )
            self.filenames = list(range(len(self._degraded_arr)))  # indices
        elif degraded_dir.exists() and clean_dir.exists():
            self.degraded_dir = degraded_dir
            self.clean_dir = clean_dir
            npy_files = sorted(
                f.name for f in degraded_dir.iterdir() if f.suffix.lower() == ".npy"
            )
            if npy_files:
                self.mode = "npy_files"
                self.filenames = npy_files
            else:
                self.mode = "image_files"
                self.filenames = sorted(
                    f.name for f in degraded_dir.iterdir()
                    if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
                )
        elif self._has_suffix_pairs(self.root):
            self.mode = "suffix_pairs"
            noisy_suffix, gt_suffix = self._detect_suffixes(self.root)
            self._noisy_suffix, self._gt_suffix = noisy_suffix, gt_suffix
            noisy_files = sorted(self.root.glob(f"*{noisy_suffix}"))
            self.filenames = [f.name[: -len(noisy_suffix)] for f in noisy_files]
            missing = [
                stem for stem in self.filenames
                if not (self.root / f"{stem}{gt_suffix}").exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} noisy files have no matching GT file "
                    f"(e.g. {missing[0]}{gt_suffix} not found in {self.root})"
                )
        else:
            raise FileNotFoundError(
                f"Could not find a recognized dataset layout under {self.root}. "
                f"Expected one of: degraded.npy+clean.npy, degraded/+clean/ "
                f"subfolders, or a flat folder with *_noisy.npy/*_gt.npy "
                f"(or similar) suffix-paired files."
            )

        if not self.filenames:
            raise FileNotFoundError(
                f"No images found under {self.root} ({self.mode}). "
                f"Check your data path / dataset extraction."
            )

    @staticmethod
    def _has_suffix_pairs(root):
        return len(list(root.glob("*_noisy.npy"))) > 0 or len(list(root.glob("*noisy.npy"))) > 0

    @staticmethod
    def _detect_suffixes(root):
        candidates = [
            ("_noisy.npy", "_gt.npy"),
            ("_noisy.npy", "_clean.npy"),
            ("noisy.npy", "gt.npy"),
        ]
        for noisy_suf, gt_suf in candidates:
            if list(root.glob(f"*{noisy_suf}")):
                return noisy_suf, gt_suf
        raise FileNotFoundError(f"Could not detect noisy/gt suffix convention in {root}")

    def __len__(self):
        return len(self.filenames)

    def _load(self, path):
        img = Image.open(path).convert("L")
        return np.asarray(img, dtype=np.float32) / 255.0

    def __getitem__(self, idx):
        if self.mode == "stacked":
            degraded = np.squeeze(np.array(self._degraded_arr[idx], dtype=np.float32))
            clean = np.squeeze(np.array(self._clean_arr[idx], dtype=np.float32))
            degraded = _normalize(degraded)
            clean = _normalize(clean)
            fname = f"{idx:05d}"
        elif self.mode == "npy_files":
            fname = self.filenames[idx]
            degraded = _load_npy_image(self.degraded_dir / fname)
            clean = _load_npy_image(self.clean_dir / fname)
        elif self.mode == "suffix_pairs":
            stem = self.filenames[idx]
            degraded = _load_npy_image(self.root / f"{stem}{self._noisy_suffix}")
            clean = _load_npy_image(self.root / f"{stem}{self._gt_suffix}")
            fname = stem
        else:
            fname = self.filenames[idx]
            degraded = self._load(self.degraded_dir / fname)
            clean = self._load(self.clean_dir / fname)

        if self.augment:
            if random.random() < 0.5:
                degraded = add_speckle_noise(degraded, severity=random.uniform(0.02, 0.15))
            if random.random() < 0.3:
                degraded = add_gaussian_noise(degraded, severity=random.uniform(0.01, 0.06))
            if random.random() < 0.3:
                degraded = random_blur(degraded, max_sigma=0.8)

            if random.random() < 0.5:
                degraded = np.fliplr(degraded).copy()
                clean = np.fliplr(clean).copy()
            if random.random() < 0.5:
                degraded = np.flipud(degraded).copy()
                clean = np.flipud(clean).copy()

        degraded_t = torch.from_numpy(degraded).unsqueeze(0).float()
        clean_t = torch.from_numpy(clean).unsqueeze(0).float()
        return degraded_t, clean_t, fname


class SizeBucketBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        buckets = {}
        for idx in range(len(dataset)):
            shape = _probe_degraded_shape(dataset, idx)
            buckets.setdefault(shape, []).append(idx)
        self.buckets = buckets
        sizes = {k: len(v) for k, v in buckets.items()}
        print(f"[dataset.py] Size buckets found: {sizes}")

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        all_batches = []
        for shape, indices in self.buckets.items():
            ref_pixels = 128 * 128
            this_pixels = shape[0] * shape[1]
            scale = max(1, round(this_pixels / ref_pixels))
            effective_bs = max(1, self.batch_size // scale)

            idxs = list(indices)
            if self.shuffle:
                rng.shuffle(idxs)
            for i in range(0, len(idxs), effective_bs):
                batch = idxs[i:i + effective_bs]
                if len(batch) < effective_bs and self.drop_last:
                    continue
                all_batches.append(batch)
        if self.shuffle:
            rng.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        total = 0
        for shape, indices in self.buckets.items():
            ref_pixels = 128 * 128
            this_pixels = shape[0] * shape[1]
            scale = max(1, round(this_pixels / ref_pixels))
            effective_bs = max(1, self.batch_size // scale)
            n = len(indices)
            total += (n // effective_bs) if self.drop_last else -(-n // effective_bs)
        return total


def _probe_degraded_shape(dataset, idx):
    base, real_idx = dataset, idx
    while isinstance(base, torch.utils.data.Subset):
        real_idx = base.indices[real_idx]
        base = base.dataset

    if base.mode == "stacked":
        return tuple(np.squeeze(base._degraded_arr[real_idx]).shape)
    elif base.mode == "npy_files":
        fname = base.filenames[real_idx]
        arr = np.load(base.degraded_dir / fname, mmap_mode="r")
        return tuple(np.squeeze(arr).shape)
    elif base.mode == "suffix_pairs":
        stem = base.filenames[real_idx]
        arr = np.load(base.root / f"{stem}{base._noisy_suffix}", mmap_mode="r")
        return tuple(np.squeeze(arr).shape)
    else:  # image_files (PNG etc.)
        fname = base.filenames[real_idx]
        img = Image.open(base.degraded_dir / fname)
        return (img.height, img.width)


def build_dataloaders(data_root, batch_size=8, num_workers=4, val_fraction=0.1, seed=42):
    from torch.utils.data import DataLoader, Subset

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(val_dir):
        train_ds = RestorationDataset(train_dir, split="train")
        val_ds = RestorationDataset(val_dir, split="val")
    else:
        print(
            f"[dataset.py] No {val_dir} found — auto-splitting "
            f"{int(val_fraction*100)}% of train/ as validation (seed={seed})."
        )
        full_train = RestorationDataset(train_dir, split="train", augment=True)
        full_val_noaug = RestorationDataset(train_dir, split="train", augment=False)

        n = len(full_train)
        n_val = max(1, int(n * val_fraction))
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()

        train_ds = Subset(full_train, train_idx)
        val_ds = Subset(full_val_noaug, val_idx)
        print(f"[dataset.py] Split: {len(train_ds)} train / {len(val_ds)} val")

    train_sampler = SizeBucketBatchSampler(train_ds, batch_size, shuffle=True, drop_last=True)
    val_sampler = SizeBucketBatchSampler(val_ds, batch_size, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    print("dataset.py loaded OK. Point build_dataloaders() at your data root, e.g.:")
    print('  train_loader, val_loader = build_dataloaders("data/")')