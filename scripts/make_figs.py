"""Figures for the report: AUROC vs number of labelled defects (YOLO26n) against PatchCore (0 defects)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "runs" / "summary.json").read_text())
KS = [5, 10, 25, 50]

fig, axes = plt.subplots(1, 4, figsize=(12, 3.1), sharey=True)
for ax, cat in zip(axes, S):
    c = S[cat]
    m = [c[f"yolo_k{k}"]["auroc_mean"] for k in KS]
    s = [c[f"yolo_k{k}"]["auroc_std"] for k in KS]
    ax.errorbar(KS, m, yerr=s, marker="o", color="#2a6fdb", capsize=3, label="YOLO26n (k defects)")
    pm, ps = c["patchcore"]["auroc_mean"], c["patchcore"]["auroc_std"]
    ax.axhline(pm, color="#d1495b", ls="--", label="PatchCore (0 defects)")
    ax.axhspan(pm - ps, pm + ps, color="#d1495b", alpha=0.15)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.minorticks_off()
    ax.set_title(cat.upper()); ax.set_xlabel("labelled defect images k"); ax.grid(alpha=0.3)
axes[0].set_ylabel("held-out image AUROC"); axes[0].set_ylim(0.55, 1.0)
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "report" / "figs" / "crossover.png", dpi=160)
print("ok")
