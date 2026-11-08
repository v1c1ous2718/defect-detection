"""YOLO26n trained on k defect images (+200 normal backgrounds); scores every held-out image.

uv run python -m dd.yolo <cat> <k> <seed>   -> runs/yolo/<cat>_k<k>_s<seed>.json
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .split import ROOT, load, normals

OUT = ROOT / "runs" / "yolo"
N_BG, EPOCHS, IMGSZ = 200, 100, 640


def boxes(mask_path: str) -> list[tuple[float, float, float, float]]:
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    h, w = m.shape
    n, _, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    out = []
    for x, y, bw, bh, area in st[1:]:
        out.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return out


def make_dataset(cat: str, k: int, seed: int) -> Path:
    s = load(cat)
    defects = random.Random(1000 + seed).sample(s["pool"], k)
    bg = normals(cat, seed)
    d = ROOT / "data" / "yolo" / f"{cat}_k{k}_s{seed}"
    if d.exists():
        shutil.rmtree(d)
    (d / "images" / "train").mkdir(parents=True)
    (d / "labels" / "train").mkdir(parents=True)
    nbox = 0
    for i, (img, mask) in enumerate(defects):
        name = f"d{i:03d}{Path(img).suffix}"
        os.symlink(img, d / "images" / "train" / name)
        bs = boxes(mask)
        nbox += len(bs)
        (d / "labels" / "train" / f"d{i:03d}.txt").write_text("".join(f"0 {a:.6f} {b:.6f} {c:.6f} {e:.6f}\n" for a, b, c, e in bs))
    for i, img in enumerate(bg):
        os.symlink(img, d / "images" / "train" / f"n{i:03d}{Path(img).suffix}")
        (d / "labels" / "train" / f"n{i:03d}.txt").write_text("")
    (d / "data.yaml").write_text(f"path: {d}\ntrain: images/train\nval: images/train\nnames:\n  0: defect\n")
    (d / "meta.json").write_text(json.dumps({"defects": [a for a, _ in defects], "boxes": nbox}))
    return d


def run(cat: str, k: int, seed: int) -> dict:
    out = OUT / f"{cat}_k{k}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    import torch
    from ultralytics import YOLO
    d = make_dataset(cat, k, seed)
    model = YOLO("yolo26n.pt")
    t0 = time.time()
    model.train(data=str(d / "data.yaml"), epochs=EPOCHS, imgsz=IMGSZ, seed=seed, deterministic=True, val=False,
                plots=False, workers=8, batch=16, project=str(ROOT / "runs" / "ultralytics"),
                name=f"{cat}_k{k}_s{seed}", exist_ok=True, verbose=False)
    fit_s = time.time() - t0
    w = ROOT / "runs" / "ultralytics" / f"{cat}_k{k}_s{seed}" / "weights" / "last.pt"
    m = YOLO(str(w))
    s = load(cat)
    items = [(p, "good") for p in s["test_normal"]] + [(a, "bad") for a, _ in s["heldout"]]
    scores = {}
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        res = m.predict([p for p, _ in chunk], imgsz=IMGSZ, conf=0.001, verbose=False)
        for (p, lab), r in zip(chunk, res):
            c = r.boxes.conf
            scores[Path(p).name + "|" + lab] = float(c.max()) if len(c) else 0.0
    r = {"cat": cat, "k": k, "seed": seed, "fit_s": fit_s, "boxes": json.loads((d / "meta.json").read_text())["boxes"],
         "weights": str(w), "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r))
    return r


if __name__ == "__main__":
    r = run(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
    print(r["cat"], r["k"], r["seed"], f"{r['fit_s']:.0f}s")
