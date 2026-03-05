import os
import csv
import time
import socket
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MRIDataset
from model import Model
import mlflow


# ---------------------------------------------------------
# CUDA optimization
# ---------------------------------------------------------
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------
# PSNR Metric
# ---------------------------------------------------------
@torch.no_grad()
def compute_psnr(pred, target, data_range=1.0):

    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])

    psnr = 10 * torch.log10(data_range ** 2 / (mse + 1e-8))

    return psnr.mean().item()


# ---------------------------------------------------------
# SSIM Metric
# ---------------------------------------------------------
@torch.no_grad()
def compute_ssim(pred, target, data_range=1.0, window_size=11):

    C = pred.shape[1]

    w = torch.ones(C,1,window_size,window_size, device=pred.device) / (window_size**2)

    pad = window_size // 2

    mu_p = F.conv2d(pred,w,padding=pad,groups=C)
    mu_t = F.conv2d(target,w,padding=pad,groups=C)

    mu_p_sq = mu_p**2
    mu_t_sq = mu_t**2
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred**2,w,padding=pad,groups=C) - mu_p_sq
    sigma_t_sq = F.conv2d(target**2,w,padding=pad,groups=C) - mu_t_sq
    sigma_pt = F.conv2d(pred*target,w,padding=pad,groups=C) - mu_pt

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
               ((mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2))

    return ssim_map.mean().item()


# ---------------------------------------------------------
# Logger
# ---------------------------------------------------------
def setup_logger(save_dir):

    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")

    logger = logging.getLogger("ifrnet_train")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------
# MLflow Setup
# ---------------------------------------------------------
def setup_mlflow(experiment_name, run_name, params, out_dir, logger,
                 server_uri="http://localhost:10226"):

    try:
        sock = socket.create_connection(("localhost",10226),timeout=2)
        sock.close()

        tracking_uri = server_uri
        logger.info("MLflow server reachable")

    except Exception:

        tracking_uri = f"file://{Path(out_dir).resolve()}/mlruns"
        logger.info("MLflow server not reachable, using local tracking")

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    mlflow.start_run(run_name=run_name)

    mlflow.log_params(params)


# ---------------------------------------------------------
# Training Function
# ---------------------------------------------------------
def train(args):

    logger = setup_logger(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Using device: {device}")

    setup_mlflow(
        args.experiment,
        f"run_bs{args.batch_size}_lr{args.lr}",
        vars(args),
        args.out,
        logger
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    train_ds = MRIDataset(args.train)
    val_ds = MRIDataset(args.val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    logger.info(f"Train samples: {len(train_ds)}")
    logger.info(f"Val samples: {len(val_ds)}")


    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = Model().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler = torch.cuda.amp.GradScaler()

    best_val_loss = float("inf")


    # -----------------------------------------------------
    # Metrics CSV
    # -----------------------------------------------------

    metrics_path = os.path.join(args.out,"metrics.csv")

    with open(metrics_path,"w",newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_psnr",
            "val_ssim",
            "epoch_time"
        ])


    # -----------------------------------------------------
    # Training Loop
    # -----------------------------------------------------

    for epoch in range(1,args.epochs+1):

        epoch_start = time.time()

        model.train()

        train_loss = 0

        for img0,img1,embt,imgt in tqdm(train_loader):

            img0 = img0.to(device)
            img1 = img1.to(device)
            embt = embt.to(device)
            imgt = imgt.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():

                pred,loss_rec,loss_geo,loss_dis = model(img0,img1,embt,imgt)

                loss = loss_rec + loss_geo + loss_dis

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)


        # -------------------------------------------------
        # Validation
        # -------------------------------------------------

        model.eval()

        val_loss = 0
        val_psnr = 0
        val_ssim = 0

        with torch.no_grad():

            for img0,img1,embt,imgt in val_loader:

                img0 = img0.to(device)
                img1 = img1.to(device)
                embt = embt.to(device)
                imgt = imgt.to(device)

                pred,loss_rec,loss_geo,loss_dis = model(img0,img1,embt,imgt)

                loss = loss_rec + loss_geo + loss_dis

                pred = torch.clamp(pred,0,1)

                val_loss += loss.item()

                val_psnr += compute_psnr(pred,imgt)

                val_ssim += compute_ssim(pred,imgt)

        val_loss /= len(val_loader)
        val_psnr /= len(val_loader)
        val_ssim /= len(val_loader)

        epoch_time = time.time() - epoch_start


        # -------------------------------------------------
        # Logging
        # -------------------------------------------------

        logger.info(
            f"Epoch {epoch} | "
            f"Train {train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"PSNR {val_psnr:.2f} | "
            f"SSIM {val_ssim:.4f} | "
            f"Time {epoch_time:.1f}s"
        )


        # CSV Logging
        with open(metrics_path,"a",newline="") as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{val_psnr:.4f}",
                f"{val_ssim:.4f}",
                f"{epoch_time:.1f}"
            ])


        # MLflow Logging
        mlflow.log_metric("train_loss",train_loss,step=epoch)
        mlflow.log_metric("val_loss",val_loss,step=epoch)
        mlflow.log_metric("val_psnr",val_psnr,step=epoch)
        mlflow.log_metric("val_ssim",val_ssim,step=epoch)
        mlflow.log_metric("epoch_time",epoch_time,step=epoch)


        # -------------------------------------------------
        # Save Best Model
        # -------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_path = os.path.join(args.out,"best_model.pth")

            torch.save(model.state_dict(),best_path)

            logger.info(f"New best model saved at epoch {epoch}")

            mlflow.log_artifact(best_path)
            mlflow.log_metric("best_val_loss",best_val_loss,step=epoch)


        # -------------------------------------------------
        # Save Epoch Checkpoint
        # -------------------------------------------------

        ckpt_path = os.path.join(args.out,f"model_{epoch:03d}.pth")

        torch.save(model.state_dict(),ckpt_path)


    mlflow.end_run()


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train",required=True)
    parser.add_argument("--val",required=True)

    parser.add_argument("--out",default="checkpoints_new")

    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--batch_size",type=int,default=8)
    parser.add_argument("--lr",type=float,default=1e-4)

    parser.add_argument("--experiment",default="MRI_ThroughPlane_S")

    args = parser.parse_args()

    train(args)