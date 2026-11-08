"""Aggregate held-out image-level AUROC / AP for PatchCore and YOLO -> runs/summary.json."""
from __future__ import annotations

import json
from statistics import mean, stdev

from sklearn.metrics import average_precision_score, roc_auc_score

from .split import CATS, ROOT

KS, SEEDS = [5, 10, 25, 50], [0, 1, 2]


def metrics(scores: dict) -> dict:
    keys = sorted(scores)
    y = [k.endswith("|bad") for k in keys]
    s = [scores[k] for k in keys]
    return {"auroc": roc_auc_score(y, s), "ap": average_precision_score(y, s), "n_pos": sum(y), "n_neg": len(y) - sum(y)}


def agg(rows: list[dict]) -> dict:
    out = {}
    for m in ("auroc", "ap"):
        v = [r[m] for r in rows]
        out[m] = v
        out[m + "_mean"] = mean(v)
        out[m + "_std"] = stdev(v) if len(v) > 1 else 0.0
    return out


def main() -> dict:
    summ = {}
    for cat in CATS:
        c = {}
        pc = [ROOT / "runs" / "patchcore" / f"{cat}_s{s}.json" for s in SEEDS]
        if all(p.exists() for p in pc):
            rs = [json.loads(p.read_text()) for p in pc]
            c["patchcore"] = {**agg([metrics(r["scores"]) for r in rs]), "fit_s": mean(r["fit_s"] for r in rs),
                              "n": metrics(rs[0]["scores"])}
        for k in KS:
            ps = [ROOT / "runs" / "yolo" / f"{cat}_k{k}_s{s}.json" for s in SEEDS]
            if all(p.exists() for p in ps):
                rs = [json.loads(p.read_text()) for p in ps]
                c[f"yolo_k{k}"] = {**agg([metrics(r["scores"]) for r in rs]), "fit_s": mean(r["fit_s"] for r in rs)}
        summ[cat] = c
        line = "  ".join(f"{m}: {v['auroc_mean']:.3f}±{v['auroc_std']:.3f}" for m, v in c.items())
        print(cat, line)
    (ROOT / "runs" / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


if __name__ == "__main__":
    main()
