"""Qualitative figure: one held-out defect per PCB category, PatchCore map vs YOLO26n at k = 5 and k = 25.

The example per category is drawn at random (numpy seed 0) from the 50 held-out defects, not picked by eye.
All models are seed 0. PatchCore: exported FP32 OpenVINO IR (same scores as PyTorch, see deploy.py).
uv run python scripts/qual_fig.py  -> report/figs/qualitative.jpg
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dd.split import CATS, ROOT, load  # noqa: E402

RNG = np.random.default_rng(0)
IMGSZ = 640
SHOW_CONF = 0.25  # ultralytics' default display threshold; the image score itself is the max over conf >= 0.001


def patchcore_map(cat, path):
    import openvino as ov
    from anomalib.data.utils import read_image
    from torchvision.transforms.v2 import Resize
    xml = ROOT / "runs" / "export" / f"patchcore_{cat}_fp32" / "weights" / "openvino" / "model.xml"
    cm = ov.Core().compile_model(str(xml), "CPU")
    x = Resize((256, 256), antialias=True)(read_image(path, as_tensor=True))[None].numpy()
    out = cm(x)
    return out["anomaly_map"][0, 0], float(out["pred_score"][0])


def yolo_pred(cat, k, path):
    from ultralytics import YOLO
    m = YOLO(str(ROOT / "runs" / "ultralytics" / f"{cat}_k{k}_s0" / "weights" / "best.pt"))
    r = m.predict(path, imgsz=IMGSZ, conf=0.001, verbose=False)[0]
    b = r.boxes
    xyxy, conf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
    return xyxy, conf, float(conf.max()) if len(conf) else 0.0


def crop_box(mask, aspect=4 / 3):
    """Landscape crop (w = aspect * h) centred on the defect, at least 30% of the short side tall."""
    ys, xs = np.nonzero(mask)
    H, W = mask.shape
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    h = max(3 * max(np.ptp(ys), np.ptp(xs) / aspect), 0.3 * min(H, W))
    h = int(min(h, H, W / aspect)); w = int(h * aspect)
    x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
    return x0, y0, w, h


def tag(ax, text):
    ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.65))


def main():
    cols = ["full image + PatchCore map", "crop + ground truth", "PatchCore (0 defects)",
            "YOLO26n, k = 5", "YOLO26n, k = 25"]
    fig, axes = plt.subplots(len(CATS), 5, figsize=(13, 2.15 * len(CATS)),
                             gridspec_kw={"width_ratios": [1.05, 1, 1, 1, 1]})
    picks = {}
    for i, cat in enumerate(CATS):
        held = load(cat)["heldout"]
        img_p, mask_p = held[int(RNG.integers(len(held)))]
        picks[cat] = Path(img_p).name
        im = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) > 0
        H, W = mask.shape
        amap, pscore = patchcore_map(cat, img_p)
        amap = cv2.resize(amap, (W, H), interpolation=cv2.INTER_LINEAR)
        x0, y0, w, h = crop_box(mask)
        sl = (slice(y0, y0 + h), slice(x0, x0 + w))

        ax = axes[i, 0]; ax.imshow(im); ax.imshow(amap, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, ec="white", lw=1.5))
        ax.set_ylabel(cat.upper(), fontsize=11)

        ax = axes[i, 1]; ax.imshow(im[sl])
        ax.contour(mask[sl], levels=[0.5], colors="lime", linewidths=1.4)

        ax = axes[i, 2]; ax.imshow(im[sl]); ax.imshow(amap[sl], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax.contour(mask[sl], levels=[0.5], colors="white", linewidths=0.8, linestyles="--")
        tag(ax, f"score {pscore:.2f}")

        for j, k in [(3, 5), (4, 25)]:
            xyxy, conf, ys = yolo_pred(cat, k, img_p)
            ax = axes[i, j]; ax.imshow(im[sl])
            ax.contour(mask[sl], levels=[0.5], colors="lime", linewidths=0.8, linestyles="--")
            for (a, b, c, d), cf in zip(xyxy, conf):
                if cf >= SHOW_CONF:
                    ax.add_patch(Rectangle((a - x0, b - y0), c - a, d - b, fill=False, ec="red", lw=1.6))
                    ax.text(a - x0, b - y0 - 4, f"{cf:.2f}", color="red", fontsize=9, va="bottom",
                            fontweight="bold", clip_on=True)
            tag(ax, f"score {ys:.2f}")
            ax.set_xlim(0, w); ax.set_ylim(h, 0)
    for j, c in enumerate(cols):
        axes[0, j].set_title(c, fontsize=10.5)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.35)
    out = ROOT / "report" / "figs" / "qualitative.jpg"
    fig.savefig(out, dpi=170, pil_kwargs={"quality": 88})
    print(out, json.dumps(picks))


if __name__ == "__main__":
    main()
