import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from model import Model


# ------------------ Pad ------------------ #
def pad_to_multiple(t, base=16):
    _, _, H, W = t.shape
    pad_h = (base - H % base) % base
    pad_w = (base - W % base) % base

    if pad_h == 0 and pad_w == 0:
        return t, H, W

    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")

    return t, H, W


# ------------------ Inference ------------------ #
def infer(img0, img1, model, device):

    t0 = torch.from_numpy(img0).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)
    t1 = torch.from_numpy(img1).float().unsqueeze(0).unsqueeze(0).repeat(1,3,1,1).to(device)

    embt = torch.tensor([0.5], dtype=torch.float32).view(1,1,1,1).to(device)

    t0, H, W = pad_to_multiple(t0)
    t1, _, _ = pad_to_multiple(t1)

    with torch.no_grad():
        pred = model.inference(t0, t1, embt)

    pred = pred[:, :, :H, :W]

    return pred[0,0].cpu().numpy()


# ------------------ Normalize ------------------ #
def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# ------------------ Linear Interpolation ------------------ #
def linear_interpolation(img0, img1):
    return 0.5 * (img0 + img1)


# ------------------ Visualization ------------------ #
def save_figure(img0, img1, gt, pred, lin_pred, save_path, title):

    # Metrics
    psnr_model = peak_signal_noise_ratio(gt, pred, data_range=1.0)
    ssim_model = structural_similarity(gt, pred, data_range=1.0)

    psnr_lin = peak_signal_noise_ratio(gt, lin_pred, data_range=1.0)
    ssim_lin = structural_similarity(gt, lin_pred, data_range=1.0)

    error_model = np.abs(gt - pred)
    error_lin = np.abs(gt - lin_pred)

    fig, axes = plt.subplots(2,5,figsize=(16,6))

    # Row 1 (Images)
    axes[0,0].imshow(img0,cmap="gray")
    axes[0,0].set_title("Input k")

    axes[0,1].imshow(gt,cmap="gray")
    axes[0,1].set_title("Ground Truth")

    axes[0,2].imshow(lin_pred,cmap="gray")
    axes[0,2].set_title(f"Linear\nPSNR {psnr_lin:.2f}")

    axes[0,3].imshow(pred,cmap="gray")
    axes[0,3].set_title(f"Model\nPSNR {psnr_model:.2f}")

    axes[0,4].imshow(img1,cmap="gray")
    axes[0,4].set_title("Input k+2")

    # Row 2 (Errors)
    axes[1,0].axis("off")

    axes[1,1].axis("off")

    axes[1,2].imshow(error_lin,cmap="hot")
    axes[1,2].set_title("Linear Error")

    axes[1,3].imshow(error_model,cmap="hot")
    axes[1,3].set_title("Model Error")

    axes[1,4].axis("off")

    for ax in axes.flatten():
        ax.axis("off")

    fig.suptitle(
        f"{title}\n"
        f"Linear PSNR {psnr_lin:.2f} | SSIM {ssim_lin:.4f}    "
        f"Model PSNR {psnr_model:.2f} | SSIM {ssim_model:.4f}",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(save_path,dpi=300)

    plt.close()

    print(f"Saved: {save_path}")

    print(f"Linear  -> PSNR {psnr_lin:.2f} | SSIM {ssim_lin:.4f}")
    print(f"Model   -> PSNR {psnr_model:.2f} | SSIM {ssim_model:.4f}")


# ------------------ Main ------------------ #
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Load data
    data = np.load(args.input)

    img0 = normalize(data["img0"])
    img1 = normalize(data["img1"])
    gt   = normalize(data["imgt"])

    # Predictions
    pred = normalize(infer(img0, img1, model, device))

    lin_pred = normalize(linear_interpolation(img0, img1))

    filename = os.path.basename(args.input).replace(".npz","_comparison.png")

    save_path = os.path.join(args.output, filename)

    save_figure(img0, img1, gt, pred, lin_pred, save_path, filename.replace("_comparison.png",""))