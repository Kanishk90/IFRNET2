import os
import torch
import numpy as np
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, root):
        self.files = sorted([os.path.join(root, f) for f in os.listdir(root) if f.endswith(".npz")])

    def __len__(self):
        return len(self.files)

    def to_tensor3(self, img):
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = t.repeat(3, 1, 1)
        return t

    def crop_to_multiple(self, img, base=16):
        # img: (C, H, W)
        _, H, W = img.shape
        H_new = (H // base) * base
        W_new = (W // base) * base
        return img[:, :H_new, :W_new]

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        img0 = self.to_tensor3(data["img0"])
        img1 = self.to_tensor3(data["img1"])
        imgt = self.to_tensor3(data["imgt"])

        # Ensure divisible by 16
        img0 = self.crop_to_multiple(img0)
        img1 = self.crop_to_multiple(img1)
        imgt = self.crop_to_multiple(imgt)

        embt = torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1)

        return img0, img1, embt, imgt
