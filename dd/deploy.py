"""CPU deployment: OpenVINO FP32 vs INT8 (NNCF PTQ) for YOLO26n (k = 50, seed 0) and PatchCore (seed 0).

For every PCB category: held-out AUROC of each exported model. Latency is measured on pcb1 only
(the networks are identical across categories), batch 1, median of 200 runs after 20 warm-ups.

uv run python -m dd.deploy   -> runs/deploy.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import median

import numpy as np

from .evaluate import metrics
from .patchcore import link_dir
from .split import CATS, ROOT, load, normals

OUT = ROOT / "runs" / "deploy.json"
EXP = ROOT / "runs" / "export"


def held(cat):
    s = load(cat)
    return [(p, "good") for p in s["test_normal"]] + [(a, "bad") for a, _ in s["heldout"]]


def timeit(fn, n=200, warm=20) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t = time.perf_counter(); fn(); ts.append((time.perf_counter() - t) * 1e3)
    return median(ts)


def yolo(cat: str, lat: bool) -> dict:
    from ultralytics import YOLO
    run = f"{cat}_k50_s0"
    w = ROOT / "runs" / "ultralytics" / run / "weights" / "last.pt"
    data = ROOT / "data" / "yolo" / run / "data.yaml"
    out = {}
    for prec in ("fp32", "int8"):
        path = Path(YOLO(str(w)).export(format="openvino", int8=(prec == "int8"), data=str(data), imgsz=640,
                                        project=str(EXP), name=f"yolo_{run}_{prec}"))
        m = YOLO(str(path), task="detect")
        items, scores = held(cat), {}
        for p, lab in items:
            r = m.predict(p, imgsz=640, conf=0.001, device="cpu", verbose=False)[0]
            scores[Path(p).name + "|" + lab] = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
        out[prec] = {**metrics(scores), "size_mb": sum(f.stat().st_size for f in path.rglob("*.bin")) / 2**20}
        if lat:
            import openvino as ov
            core = ov.Core()
            xml = next(path.glob("*.xml"))
            x = np.random.rand(1, 3, 640, 640).astype(np.float32)
            for th in (1, 6):
                cm = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": th})
                req = cm.create_infer_request()
                out[prec][f"ms_{th}t"] = timeit(lambda: req.infer({0: x}))
    return out


def patchcore(cat: str, lat: bool) -> dict:
    from lightning.pytorch import seed_everything
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.data.utils import read_image
    from anomalib.models import Patchcore
    from torchvision.transforms.v2 import Resize
    import nncf
    import openvino as ov
    base = ROOT / "data" / "pc" / f"{cat}_s0"
    link_dir(base / "train" / "good", normals(cat, 0))
    seed_everything(0, workers=True)
    dm = Folder(name=cat, root=base, normal_dir="train/good", abnormal_dir="test/bad", normal_test_dir="test/good",
                val_split_mode="same_as_test", num_workers=8, seed=0)
    dm.setup()
    model = Patchcore()
    eng = Engine(default_root_dir=str(ROOT / "runs" / "anomalib"), logger=False, devices=1, accelerator="gpu")
    eng.fit(model=model, datamodule=dm)
    prep = lambda f: Resize((256, 256), antialias=True)(read_image(f, as_tensor=True))  # = Patchcore's own resize
    # anomalib exports with a symbolic batch of 2 baked into its resize nodes; export a static batch-1 graph instead.
    fp32 = eng.export(model=model, export_type=ExportType.OPENVINO, export_root=EXP / f"patchcore_{cat}_fp32",
                      input_size=(256, 256), onnx_kwargs={"dynamic_shapes": None})
    # INT8: NNCF PTQ with default settings, calibrated on the 200 normal train images only.
    # (anomalib's INT8_PTQ calibrates on the val loader, which here is the held-out test set.)
    cal = nncf.Dataset(normals(cat, 0), lambda f: prep(f)[None].numpy())
    # int8: NNCF defaults (pre-registered). int8_bb (post-hoc, see DESIGN.md): the three memory-bank distance MatMuls
    # stay FP32, only the WideResNet backbone is quantized.
    variants = {"int8": None, "int8_bb": nncf.IgnoredScope(types=["MatMul"])}
    xmls = {"fp32": Path(fp32)}
    for name, ign in variants.items():
        xmls[name] = EXP / f"patchcore_{cat}_{name}" / "model.xml"
        if not xmls[name].exists():
            kw = {"ignored_scope": ign} if ign else {}
            q = nncf.quantize(ov.Core().read_model(str(fp32)), cal, subset_size=len(normals(cat, 0)), **kw)
            xmls[name].parent.mkdir(parents=True, exist_ok=True)
            ov.save_model(q, str(xmls[name]), compress_to_fp16=False)
    out = {}
    for prec, xml in xmls.items():
        # raw compiled model: anomalib's OpenVINOInferencer returned wrong scores after its first call here
        cm = ov.Core().compile_model(str(xml), "CPU")
        scores = {Path(p).name + "|" + lab: float(cm(prep(p)[None].numpy())["pred_score"][0]) for p, lab in held(cat)}
        out[prec] = {**metrics(scores), "size_mb": xml.with_suffix(".bin").stat().st_size / 2**20}
        if lat:
            core = ov.Core()
            x = np.random.rand(1, 3, 256, 256).astype(np.float32)  # timed pre-resized, like YOLO at 640
            for th in (1, 6):
                cm = core.compile_model(str(xml), "CPU", {"INFERENCE_NUM_THREADS": th})
                req = cm.create_infer_request()
                out[prec][f"ms_{th}t"] = timeit(lambda: req.infer({0: x}))
    return out


def main():
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for cat in CATS:
        for name, fn in (("yolo26n_k50", yolo), ("patchcore", patchcore)):
            key = f"{cat}/{name}"
            if key not in res:
                res[key] = fn(cat, lat=(cat == "pcb1"))
                OUT.write_text(json.dumps(res, indent=1))
            print(key, {p: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k in
                            ("auroc", "size_mb", "ms_1t", "ms_6t")} for p, d in res[key].items()}, flush=True)


if __name__ == "__main__":
    main()
