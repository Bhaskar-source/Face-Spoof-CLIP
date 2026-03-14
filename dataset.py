"""
Dataset loader for UniAttackData
Reads train/dev/test txt files with format:
    <relative_path>  <label>
Labels: 0 = live, 1 = spoof (physical or digital)
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────────────
# CLIP standard pre-processing (224×224, ImageNet-like normalisation)
# ──────────────────────────────────────────────────────────────────────────────
def get_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.48145466, 0.4578275,  0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)),
        ])


class UniAttackDataset(Dataset):
    """
    Args:
        txt_file  : path to the protocol txt file
        data_root : root directory of the dataset
        transform : torchvision transform
        label_col : column index of the label in the txt file (default 1)
        path_col  : column index of the image path (default 0)
    """
    def __init__(
        self,
        txt_file: str,
        data_root: str,
        transform=None,
        label_col: int = 1,
        path_col: int = 0,
    ):
        self.data_root = data_root
        self.transform = transform
        self.samples = []   # list of (abs_path, binary_label)

        with open(txt_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                rel_path = parts[path_col]
                label    = int(parts[label_col])

                # ── Label convention in the txt files ─────────────────────────
                # Detailed:  Live=0, Physical=1, Adversarial=2, Digital=3
                # Binary:    Live=0  →  real (class 0)
                #            Spoof (1,2,3) → fake (class 1)
                binary_label = 0 if label == 0 else 1

                # ── Path fix ──────────────────────────────────────────────────
                # txt lines start with "UniAttackData_P/Data/..."
                # data_root ends   with "...\UniAttackData_P"
                # Strip the leading "UniAttackData_P/" prefix to avoid doubling
                prefix = "UniAttackData_P/"
                if rel_path.startswith(prefix):
                    rel_path = rel_path[len(prefix):]

                # Normalise forward-slashes for os.path.join on Windows
                rel_path = rel_path.replace("/", os.sep)
                abs_path = os.path.join(data_root, rel_path)
                self.samples.append((abs_path, binary_label))

        print(f"[Dataset] Loaded {len(self.samples)} samples from {txt_file}")
        live_count  = sum(1 for _, l in self.samples if l == 0)
        spoof_count = sum(1 for _, l in self.samples if l == 1)
        print(f"          Live: {live_count}  |  Spoof: {spoof_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            # Return a black image on error so training doesn't crash
            img = Image.new("RGB", (224, 224), (0, 0, 0))
            print(f"Warning: could not load {path}: {e}")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    train_txt: str,
    dev_txt:   str,
    test_txt:  str,
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_ds = UniAttackDataset(train_txt, data_root, get_transforms(True))
    dev_ds   = UniAttackDataset(dev_txt,   data_root, get_transforms(False))
    test_ds  = UniAttackDataset(test_txt,  data_root, get_transforms(False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, dev_loader, test_loader
