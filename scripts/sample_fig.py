"""Sample grid for the report: one normal and one defective image per PCB category, defect boxes in red."""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dd.split import CATS, ROOT, load
from dd.yolo import boxes

fig, axes = plt.subplots(2, 4, figsize=(12, 4.6))
for j, cat in enumerate(CATS):
    s = load(cat)
    img, mask = s["pool"][0]
    for i, (path, bx) in enumerate([(s["train_normal"][0], []), (img, boxes(mask))]):
        im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        ax = axes[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for cx, cy, bw, bh in bx:
            ax.add_patch(Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h, fill=False, ec="red", lw=2))
        if i == 0: ax.set_title(cat.upper(), fontsize=12)
axes[0, 0].set_ylabel("normal", fontsize=11); axes[1, 0].set_ylabel("defect (box from mask)", fontsize=11)
fig.tight_layout(pad=0.4)
fig.savefig(ROOT / "report" / "figs" / "samples.jpg", dpi=110, pil_kwargs={"quality": 85})
