"""Thumbnails for the method figure (report/figs/method.png): normal/defect inputs, PatchCore heat map, YOLO26n (k=25) boxes.
Same model checkpoints and crop rule as scripts/qual_fig.py (PCB4, held-out pick of numpy seed 0)."""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import qual_fig as q
from dd.split import load

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
cat = "pcb4"; s = load(cat)
rng = np.random.default_rng(0)
for c in ["pcb1", "pcb2", "pcb3"]:          # advance the RNG exactly as qual_fig.py does
    rng.integers(len(load(c)["heldout"]))
img_p, mask_p = s["heldout"][int(rng.integers(len(s["heldout"])))]
im = cv2.imread(img_p); mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE) > 0
H, W = mask.shape
x0, y0, w, h = q.crop_box(mask)
crop = lambda a: a[y0:y0 + h, x0:x0 + w]
def save(a, name):
    a = cv2.resize(a, (240, int(240 * a.shape[0] / a.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / name), a, [cv2.IMWRITE_JPEG_QUALITY, 88])
for i, p in enumerate(s["train_normal"][:3]):
    n = cv2.imread(p); save(n, f"pcb-normal{i}.jpg")
save(im, "pcb-defect-full.jpg")
amap, score = q.patchcore_map(cat, img_p)
amap = cv2.resize(amap, (W, H))
heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
save(crop(cv2.addWeighted(im, 0.5, heat, 0.5, 0)), "pcb-heat.jpg")
xyxy, conf, ys = q.yolo_pred(cat, 25, img_p)
box = im.copy()
for (a, b, c, d), cf in zip(xyxy.astype(int), conf):
    if cf >= q.SHOW_CONF: cv2.rectangle(box, (a, b), (c, d), (0, 0, 255), 6)
save(crop(box), "pcb-yolo.jpg")
print(Path(img_p).name, round(score, 2), round(ys, 2))
