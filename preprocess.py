import os
import nibabel as nib
import numpy as np
import random
import logging
import sys
from pathlib import Path
from tqdm import tqdm


# =========================================================
# Logging
# =========================================================
def setup_logger(out_dir):

    out_dir = Path(out_dir)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "preprocess.log"

    if logging.getLogger().hasHandlers():
        logging.getLogger().handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
    )

    logging.info("======================================")
    logging.info("MRI Triplet Preprocessing Started")
    logging.info("======================================")


# =========================================================
# Center Crop 128³
# =========================================================
def center_crop_128(vol):

    if vol.ndim != 3:
        return None

    H, W, D = vol.shape

    if H < 128 or W < 128 or D < 128:
        return None

    x = (H - 128) // 2
    y = (W - 128) // 2
    z = (D - 128) // 2

    return vol[x:x+128, y:y+128, z:z+128]


# =========================================================
# Save Triplets
# =========================================================
def save_split(volume_list, out_dir, split_name):

    split_dir = os.path.join(out_dir, f"{split_name}_npz")
    os.makedirs(split_dir, exist_ok=True)

    idx = 0
    total_triplets = 0

    logging.info(f"Starting {split_name.upper()} split...")

    for path in tqdm(volume_list,
                     desc=f"{split_name.upper()} Volumes",
                     unit="volume"):

        try:

            nii = nib.load(path)
            nii = nib.as_closest_canonical(nii)

            vol = nii.get_fdata().astype(np.float32)

            cropped = center_crop_128(vol)

            if cropped is None:
                logging.warning(f"Skipped (invalid size): {path}")
                continue

            # -------------------------------------------------
            # Percentile Normalization (Robust for MRI)
            # -------------------------------------------------

            p1 = np.percentile(cropped, 1)
            p99 = np.percentile(cropped, 99)

            cropped = np.clip(cropped, p1, p99)
            cropped = (cropped - p1) / (p99 - p1 + 1e-8)

            H, W, D = cropped.shape

            # -------------------------------------------------
            # Generate slice triplets
            # -------------------------------------------------

            for k in range(D - 2):

                img0 = cropped[:, :, k]
                imgt = cropped[:, :, k + 1]
                img1 = cropped[:, :, k + 2]

                # Skip empty slices (background)
                if np.mean(img0) < 0.01 and np.mean(img1) < 0.01:
                    continue

                np.savez_compressed(
                    os.path.join(split_dir, f"{idx:08d}.npz"),
                    img0=img0,
                    img1=img1,
                    imgt=imgt,
                )

                idx += 1
                total_triplets += 1

        except Exception as e:
            logging.error(f"Failed processing {path}: {e}")

    logging.info(f"{split_name.upper()} complete.")
    logging.info(f"Volumes processed: {len(volume_list)}")
    logging.info(f"Triplets saved: {total_triplets}")
    logging.info("--------------------------------------")


# =========================================================
# Main Preprocessing Function
# =========================================================
def preprocess_with_split(src_dir,
                          out_dir,
                          train_ratio=0.8,
                          val_ratio=0.1,
                          seed=42):

    setup_logger(out_dir)

    volumes = []

    # Scan volumes
    for root, _, files in tqdm(os.walk(src_dir), desc="Scanning volumes"):
        for f in files:
            if f.endswith((".nii", ".nii.gz")):
                volumes.append(os.path.join(root, f))

    if len(volumes) == 0:
        logging.error("No NIfTI files found.")
        return

    volumes.sort()

    random.seed(seed)
    random.shuffle(volumes)

    logging.info(f"Total volumes found: {len(volumes)}")

    # -------------------------------------------------
    # Dataset split
    # -------------------------------------------------

    n = len(volumes)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_vols = volumes[:n_train]
    val_vols = volumes[n_train:n_train + n_val]
    test_vols = volumes[n_train + n_val:]

    logging.info(f"Train volumes: {len(train_vols)}")
    logging.info(f"Val volumes:   {len(val_vols)}")
    logging.info(f"Test volumes:  {len(test_vols)}")

    logging.info("--------------------------------------")

    # -------------------------------------------------
    # Save triplets
    # -------------------------------------------------

    save_split(train_vols, out_dir, "train")
    save_split(val_vols, out_dir, "val")
    save_split(test_vols, out_dir, "test")

    logging.info("Preprocessing Finished Successfully")
    logging.info("======================================")


# =========================================================
# Entry Point
# =========================================================
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--src", required=True,
                        help="Folder containing MRI NIfTI volumes")

    parser.add_argument("--out", required=True,
                        help="Output folder")

    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)

    args = parser.parse_args()

    preprocess_with_split(
        args.src,
        args.out,
        args.train_ratio,
        args.val_ratio
    )