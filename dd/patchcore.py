"""PatchCore (anomalib defaults) on one VisA PCB category; scores every held-out image.

uv run python -m dd.patchcore <cat> <seed>   -> runs/patchcore/<cat>_s<seed>.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from .split import ROOT, load, normals

OUT = ROOT / "runs" / "patchcore"


def link_dir(d: Path, files: list[str]) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        dst = d / Path(f).name
        if not dst.exists():
            os.symlink(f, dst)
    return d


def run(cat: str, seed: int) -> dict:
    out = OUT / f"{cat}_s{seed}.json"
    if out.exists():
        return json.loads(out.read_text())
    import torch
    from lightning.pytorch import seed_everything
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    s = load(cat)
    base = ROOT / "data" / "pc" / f"{cat}_s{seed}"
    link_dir(base / "train" / "good", normals(cat, seed))
    link_dir(base / "test" / "good", s["test_normal"])
    link_dir(base / "test" / "bad", [a for a, _ in s["heldout"]])
    seed_everything(seed, workers=True)
    dm = Folder(name=cat, root=base, normal_dir="train/good", abnormal_dir="test/bad", normal_test_dir="test/good",
                val_split_mode="same_as_test", train_batch_size=32, eval_batch_size=32, num_workers=8, seed=seed)
    dm.setup()
    model = Patchcore()
    eng = Engine(default_root_dir=str(ROOT / "runs" / "anomalib"), logger=False, devices=1, accelerator="gpu")
    t0 = time.time()
    eng.fit(model=model, datamodule=dm)
    fit_s = time.time() - t0
    preds = eng.predict(model=model, dataloaders=dm.test_dataloader(), return_predictions=True)
    scores = {}
    for b in preds:
        for p, sc in zip(b.image_path, b.pred_score.flatten().tolist()):
            scores[Path(p).name + ("|bad" if "/test/bad/" in str(p) else "|good")] = sc
    res = {"cat": cat, "seed": seed, "fit_s": fit_s, "n_train": len(normals(cat, seed)),
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, "scores": scores}
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    return res


if __name__ == "__main__":
    r = run(sys.argv[1], int(sys.argv[2]))
    print(r["cat"], r["seed"], len(r["scores"]), f"{r['fit_s']:.0f}s")
