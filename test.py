import os
import time
import logging
import argparse
import numpy as np
import torch
from tqdm import tqdm
from model import Model


# ---------------------- Logging ---------------------- #
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    return logging.getLogger(__name__)


# ---------------------- Pad ---------------------- #
def pad_to_multiple(t, base=16):
    _, _, H, W = t.shape
    pad_h = (base - H % base) % base
    pad_w = (base - W % base) % base
    if pad_h == 0 and pad_w == 0:
        return t, H, W
    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, H, W


# ---------------------- Inference ---------------------- #
def infer_pair(img0, img1, model, device):
    t0 = torch.from_numpy(img0).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
    t1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
    embt = torch.tensor([0.5], dtype=torch.float32).view(1,1,1,1).to(device)

    t0, orig_H, orig_W = pad_to_multiple(t0)
    t1, _, _ = pad_to_multiple(t1)

    with torch.no_grad():
        pred = model.inference(t0, t1, embt)

    pred = pred[:, :, :orig_H, :orig_W]
    return pred[0, 0].cpu().numpy()


# ---------------------- Run ---------------------- #
def run(input_path, model_path, output_dir):
    logger = setup_logger()
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Model loaded.")

    if os.path.isdir(input_path):
        files = sorted([f for f in os.listdir(input_path) if f.endswith(".npz")])
        logger.info(f"Found {len(files)} NPZ files.")
    else:
        files = [os.path.basename(input_path)]
        input_path = os.path.dirname(input_path)

    start = time.time()

    for fname in tqdm(files, desc="Testing"):
        data = np.load(os.path.join(input_path, fname))
        img0 = data["img0"]
        img1 = data["img1"]

        pred = infer_pair(img0, img1, model, device)

        # out_path = os.path.join(output_dir, fname.replace(".npz", "_pred.npy"))
        # np.save(out_path, pred)

        from PIL import Image

        # Normalize to 0-255 for PNG
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        pred_uint8 = (pred_norm * 255).astype(np.uint8)

        out_path = os.path.join(output_dir, fname.replace(".npz", "_pred.png"))
        Image.fromarray(pred_uint8).save(out_path)

    logger.info(f"Done. Total time: {time.time()-start:.2f}s")


# ---------------------- Main ---------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.model, args.output)