import os
import time
import socket
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from dataset import MRIDataset
from model import Model
import mlflow


# ---------------------------------
# Logger Setup
# ---------------------------------
def setup_logger(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, "train.log")

    logger = logging.getLogger("ifrnet_train")
    logger.setLevel(logging.INFO)

    # Remove old handlers safely
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.propagate = False  # IMPORTANT

    logger.info(f"Log file created at: {log_file}")

    return logger

# ---------------------------------
# MLflow Setup (always enabled)
# ---------------------------------
def setup_mlflow(experiment_name, run_name, params, out_dir, logger,
                 server_uri="http://localhost:10226"):
    # Quick check if the MLflow server is reachable
    try:
        host, port = "localhost", 10226
        sock = socket.create_connection((host, port), timeout=2)
        sock.close()
        tracking_uri = server_uri
        logger.info(f"MLflow server reachable at {server_uri}")
    except (socket.timeout, ConnectionRefusedError, OSError):
        tracking_uri = f"file://{Path(out_dir).resolve()}/mlruns"
        logger.info(f"MLflow server not reachable, using local tracking: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(params)
    mlflow.set_tag("experiment_dir", str(Path(out_dir).resolve()))

    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {experiment_name}")
    logger.info(f"MLflow run: {run_name}")


# ---------------------------------
# Training Function
# ---------------------------------
def train(args):
    logger = setup_logger(args.out)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("=" * 60)
    logger.info("MRI Through-Plane Super-Resolution Training")
    logger.info("=" * 60)
    logger.info(f"Using device: {device}")

    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Log all arguments
    logger.info("Arguments:")
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")
    logger.info("-" * 60)

    # ---------------- MLflow Setup (always enabled) ----------------
    setup_mlflow(
        experiment_name=args.experiment,
        run_name=f"run_bs{args.batch_size}_lr{args.lr}",
        params=vars(args),
        out_dir=args.out,
        logger=logger,
    )
    logger.info("-" * 60)

    # ---------------- Data ----------------
    logger.info("Loading datasets...")
    train_ds = MRIDataset(args.train)
    val_ds = MRIDataset(args.val)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train samples: {len(train_ds)} ({len(train_loader)} batches)")
    logger.info(f"Val samples:   {len(val_ds)} ({len(val_loader)} batches)")

    # Verify a sample
    sample = train_ds[0]
    logger.info(f"Sample shapes - img0: {sample[0].shape}, embt: {sample[2].shape}")
    logger.info("-" * 60)

    # ---------------- Model ----------------
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    logger.info("-" * 60)

    best_val_loss = float("inf")
    log_interval = max(1, len(train_loader) // 10)  # Log ~10 times per epoch

    # ---------------------------------
    # Training Loop
    # ---------------------------------
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # -------- Train --------
        model.train()
        train_loss = 0.0
        train_loss_rec = 0.0

        for batch_idx, (img0, img1, embt, imgt) in enumerate(train_loader, 1):
            img0 = img0.to(device)
            img1 = img1.to(device)
            imgt = imgt.to(device)
            embt = embt.to(device)

            pred, loss_rec, loss_geo, loss_dis = model(img0, img1, embt, imgt)
            loss = loss_rec + loss_geo + loss_dis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_rec += loss_rec.item()

            # Per-batch logging at intervals
            if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
                gpu_mem = ""
                if device == "cuda":
                    mem_gb = torch.cuda.memory_allocated() / 1e9
                    gpu_mem = f" | GPU mem: {mem_gb:.2f} GB"
                # logger.info(
                #     f"  Epoch [{epoch}/{args.epochs}] "
                #     f"Batch [{batch_idx}/{len(train_loader)}] "
                #     f"Loss: {loss.item():.6f} (rec: {loss_rec.item():.6f})"
                #     f"{gpu_mem}"
                # )

        train_loss /= len(train_loader)
        train_loss_rec /= len(train_loader)

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        val_loss_rec = 0.0

        with torch.no_grad():
            for img0, img1, embt, imgt in val_loader:
                img0 = img0.to(device)
                img1 = img1.to(device)
                imgt = imgt.to(device)
                embt = embt.to(device)

                pred, loss_rec, loss_geo, loss_dis = model(
                    img0, img1, embt, imgt
                )
                loss = loss_rec + loss_geo + loss_dis
                val_loss += loss.item()
                val_loss_rec += loss_rec.item()

        val_loss /= len(val_loader)
        val_loss_rec /= len(val_loader)

        epoch_time = time.time() - epoch_start

        # -------- Epoch Summary --------
        logger.info("-" * 60)
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] Summary | "
            f"Time: {epoch_time:.1f}s | "
            f"Train Loss: {train_loss:.6f} (rec: {train_loss_rec:.6f}) | "
            f"Val Loss: {val_loss:.6f} (rec: {val_loss_rec:.6f})"
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_loss_rec", train_loss_rec, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_loss_rec", val_loss_rec, step=epoch)
        mlflow.log_metric("epoch_time_s", epoch_time, step=epoch)

        # -------- Save best model --------
        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_path = os.path.join(args.out, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(
                f"New best model saved at epoch {epoch} "
                f"(val_loss: {val_loss:.6f}, improved by {improvement:.6f})"
            )
            mlflow.log_artifact(best_path)
            mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

        # -------- Save checkpoint --------
        ckpt_path = os.path.join(args.out, f"model_{epoch:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        logger.info("-" * 60)

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"Training completed in {total_time / 3600:.2f} hours")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info("=" * 60)
    mlflow.end_run()


# ---------------------------------
# Entry Point
# ---------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--experiment", default="MRI_SR")

    args = parser.parse_args()

    train(args)
