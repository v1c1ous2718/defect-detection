"""Fixed splits for VisA PCB categories (see DESIGN.md).

data/split/<cat>.json : {"train_normal": [...], "test_normal": [...], "pool": [...], "heldout": [...]}
each anomaly entry is [image_path, mask_path]; paths are absolute.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "datasets" / "visa_raw"
SPLIT = ROOT / "data" / "split"
CATS = ["pcb1", "pcb2", "pcb3", "pcb4"]


def build(seed: int = 0) -> None:
    rows = defaultdict(list)
    with open(RAW / "split_csv" / "1cls.csv") as f:
        for r in csv.DictReader(f):
            if r["object"] in CATS:
                rows[r["object"]].append(r)
    SPLIT.mkdir(parents=True, exist_ok=True)
    for cat in CATS:
        rs = rows[cat]
        p = lambda s: str(RAW / s)
        train_normal = sorted(p(r["image"]) for r in rs if r["split"] == "train" and r["label"] == "normal")
        test_normal = sorted(p(r["image"]) for r in rs if r["split"] == "test" and r["label"] == "normal")
        anom = sorted([p(r["image"]), p(r["mask"])] for r in rs if r["split"] == "test" and r["label"] == "anomaly")
        assert not [r for r in rs if r["split"] == "train" and r["label"] != "normal"]
        random.Random(seed).shuffle(anom)
        half = len(anom) // 2
        out = {"train_normal": train_normal, "test_normal": test_normal, "pool": anom[:half], "heldout": anom[half:]}
        (SPLIT / f"{cat}.json").write_text(json.dumps(out, indent=0))
        print(cat, {k: len(v) for k, v in out.items()})


def load(cat: str) -> dict:
    return json.loads((SPLIT / f"{cat}.json").read_text())


def normals(cat: str, seed: int, n: int = 200) -> list[str]:
    """The seed's normal train images, shared by PatchCore (memory bank) and YOLO (background)."""
    return random.Random(seed).sample(load(cat)["train_normal"], n)


if __name__ == "__main__":
    build()
