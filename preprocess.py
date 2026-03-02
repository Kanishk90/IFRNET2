# import os
# import nibabel as nib
# import numpy as np


# def normalize(vol):
#     vol = vol.astype(np.float32)
#     vmin, vmax = vol.min(), vol.max()
#     if vmax - vmin < 1e-8:
#         return np.zeros_like(vol, dtype=np.float32)
#     return (vol - vmin) / (vmax - vmin)


# def extract_triplets(vol):
#     """
#     vol: (H,W,D)
#     returns list of (img0, img1, imgt)
#     """
#     H, W, D = vol.shape
#     samples = []
#     for k in range(D - 2):
#         img0 = vol[:, :, k]
#         imgt = vol[:, :, k + 1]
#         img1 = vol[:, :, k + 2]
#         samples.append((img0, img1, imgt))
#     return samples


# def preprocess_volumes(src_dir, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     idx = 0

#     for root, _, files in os.walk(src_dir):
#         for fname in files:
#             if not fname.endswith((".nii", ".nii.gz")):
#                 continue

#             path = os.path.join(root, fname)
#             print("Processing:", path)

#             vol = nib.load(path).get_fdata()
#             vol = normalize(vol)

#             triplets = extract_triplets(vol)

#             for img0, img1, imgt in triplets:
#                 np.savez_compressed(
#                     os.path.join(out_dir, f"{idx:08d}.npz"),
#                     img0=img0,
#                     img1=img1,
#                     imgt=imgt,
#                 )
#                 idx += 1

#     print(f"Saved {idx} triplets to {out_dir}")


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--src", required=True, help="folder with MRI volumes")
#     parser.add_argument("--out", required=True, help="output folder for samples")
#     args = parser.parse_args()

#     preprocess_volumes(args.src, args.out)


import os
import nibabel as nib
import numpy as np
import random
import logging
from datetime import datetime


# -----------------------------
# Logging setup
# -----------------------------
def setup_logger(out_dir):
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "preprocess.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("======================================")
    logging.info("MRI Preprocessing Started")
    logging.info("======================================")


# -----------------------------
# Normalization
# -----------------------------
def normalize(vol):
    vol = vol.astype(np.float32)
    vmin, vmax = vol.min(), vol.max()

    if vmax - vmin < 1e-8:
        logging.warning("Volume has near-constant intensity.")
        return np.zeros_like(vol, dtype=np.float32)

    return (vol - vmin) / (vmax - vmin)


# -----------------------------
# Extract slice triplets
# -----------------------------
def extract_triplets(vol):
    H, W, D = vol.shape
    samples = []

    if D < 3:
        logging.warning(f"Volume depth {D} too small. Skipping.")
        return samples

    for k in range(D - 2):
        img0 = vol[:, :, k]
        imgt = vol[:, :, k + 1]
        img1 = vol[:, :, k + 2]
        samples.append((img0, img1, imgt))

    return samples


# -----------------------------
# Save split
# -----------------------------
def save_split(volume_list, out_dir, split_name):
    split_dir = os.path.join(out_dir, f"{split_name}_npz")
    os.makedirs(split_dir, exist_ok=True)

    idx = 0
    total_triplets = 0

    for path in volume_list:
        try:
            logging.info(f"[{split_name}] Processing: {path}")

            vol = nib.load(path).get_fdata()
            logging.info(f"Shape: {vol.shape}")

            vol = normalize(vol)
            triplets = extract_triplets(vol)

            for img0, img1, imgt in triplets:
                np.savez_compressed(
                    os.path.join(split_dir, f"{idx:08d}.npz"),
                    img0=img0,
                    img1=img1,
                    imgt=imgt,
                )
                idx += 1

            total_triplets += len(triplets)

        except Exception as e:
            logging.error(f"Failed processing {path}: {e}")

    logging.info(f"{split_name.upper()} split complete.")
    logging.info(f"Volumes: {len(volume_list)}")
    logging.info(f"Triplets saved: {total_triplets}")
    logging.info("--------------------------------------")


# -----------------------------
# Main split function
# -----------------------------
def preprocess_with_split(src_dir, out_dir,
                          train_ratio=0.8,
                          val_ratio=0.1,
                          seed=42):

    setup_logger(out_dir)

    volumes = []
    for root, _, files in os.walk(src_dir):
        for f in files:
            if f.endswith((".nii", ".nii.gz")):
                volumes.append(os.path.join(root, f))

    if len(volumes) == 0:
        logging.error("No NIfTI files found in source directory.")
        return

    volumes.sort()
    random.seed(seed)
    random.shuffle(volumes)

    n = len(volumes)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_vols = volumes[:n_train]
    val_vols = volumes[n_train:n_train + n_val]
    test_vols = volumes[n_train + n_val:]

    logging.info(f"Total volumes found: {n}")
    logging.info(f"Train: {len(train_vols)}")
    logging.info(f"Val:   {len(val_vols)}")
    logging.info(f"Test:  {len(test_vols)}")
    logging.info("--------------------------------------")

    save_split(train_vols, out_dir, "train")
    save_split(val_vols, out_dir, "val")
    save_split(test_vols, out_dir, "test")

    logging.info("Preprocessing Finished Successfully")
    logging.info("======================================")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder containing MRI volumes")
    parser.add_argument("--out", required=True, help="Output root folder")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()

    preprocess_with_split(
        args.src,
        args.out,
        args.train_ratio,
        args.val_ratio
    )