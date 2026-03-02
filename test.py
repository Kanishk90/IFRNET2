import torch
import nibabel as nib
import numpy as np
from model import Model


def normalize(vol):
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(vol)
    return (vol - vmin) / (vmax - vmin)


def pad_to_multiple(t, base=16):
    """Pad tensor (1, C, H, W) so H and W are multiples of base."""
    _, _, H, W = t.shape
    pad_h = (base - H % base) % base
    pad_w = (base - W % base) % base
    if pad_h == 0 and pad_w == 0:
        return t, H, W
    t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
    return t, H, W


def interp_volume(vol, model, device="cuda"):
    H, W, D = vol.shape
    out = np.zeros((H, W, 2 * D - 1), dtype=np.float32)
    out[:, :, 0::2] = vol

    model.eval()

    with torch.no_grad():
        for k in range(D - 1):
            s0 = vol[:, :, k]
            s1 = vol[:, :, k + 1]

            t0 = torch.from_numpy(s0).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            t1 = torch.from_numpy(s1).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
            embt = torch.tensor([0.5], dtype=torch.float32).view(1, 1, 1, 1).to(device)

            # Pad to 16-multiple for encoder compatibility
            t0, orig_H, orig_W = pad_to_multiple(t0)
            t1, _, _ = pad_to_multiple(t1)

            pred = model.inference(t0, t1, embt)
            # Crop back to original size
            pred = pred[:, :, :orig_H, :orig_W]
            pred = pred[0, 0].cpu().numpy()

            out[:, :, 2 * k + 1] = pred

    return out


def run(input_nii, model_path, output_nii):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    nii = nib.load(input_nii)
    vol = nii.get_fdata()
    vol = normalize(vol).astype(np.float32)

    model = Model().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    out = interp_volume(vol, model, device)

    out_nii = nib.Nifti1Image(out, nii.affine)
    nib.save(out_nii, output_nii)
    print(f"Saved interpolated volume: {output_nii} (shape {out.shape})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run(args.input, args.model, args.output)
