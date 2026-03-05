import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Input file
file_path = Path("/home/kanishk/Workspace/IFRNet2/data/train_npz/00000064.npz")

# Output directory
output_dir = Path("figure")
output_dir.mkdir(exist_ok=True)

# Load file
data = np.load(file_path)

img0 = data["img0"]
imgt = data["imgt"]
img1 = data["img1"]

print("File:", file_path.name)
print("Shape:", img0.shape)
print("Intensity range:", img0.min(), img0.max())

# Create figure
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax1.imshow(img0, cmap="gray")
ax1.set_title("img0 (k)")
ax1.axis("off")

ax2 = fig.add_subplot(1, 3, 2)
ax2.imshow(imgt, cmap="gray")
ax2.set_title("imgt (k+1)")
ax2.axis("off")

ax3 = fig.add_subplot(1, 3, 3)
ax3.imshow(img1, cmap="gray")
ax3.set_title("img1 (k+2)")
ax3.axis("off")

plt.tight_layout()

# Save figure
save_path = output_dir / f"{file_path.stem}.png"
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved to: {save_path}")