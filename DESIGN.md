# Design (fixed before running)

Question: on PCB images, how many labelled defect images does a supervised detector (YOLO26) need
before it beats an unsupervised anomaly detector (PatchCore) that never sees a defect?

## Data
- VisA (Amazon, CC BY 4.0), categories pcb1, pcb2, pcb3, pcb4.
- Official `split_csv/1cls.csv`: train = normal images only; test = normal + anomaly.
- Each category's test anomalies are split once (seed 0) into two halves:
  - **pool** (50%): YOLO may draw its k training defects from here.
  - **held-out** (50%): used for evaluation only, together with **all** test normals.
- Both methods are scored on exactly the same held-out images.

## Methods
- **PatchCore** (anomalib 2.7 defaults: WideResNet-50 layers 2+3, coreset 10%, 256 px).
  Trained on 200 normal train images, the same 200 that YOLO uses as background for that seed. Never sees a defect.
  Seeds 0-2 (different 200-image draws and coreset sampling).
- **YOLO26n** (ultralytics 8.4, COCO-pretrained, default hyperparameters, 640 px, 100 epochs).
  One class "defect"; boxes = connected components of the VisA masks.
  Train set = k defect images from the pool + the seed's 200 normal train images as background.
  k ∈ {5, 10, 25, 50 (= whole pool)}; seeds 0-2 (different draws of the k images and of the 200 normals).
  No checkpoint selection: the final-epoch weights are used (with k = 5 a validation split would remove training defects).

## Metrics (held-out set)
- Image-level AUROC and AP (anomaly = positive).
  PatchCore image score = its anomaly score; YOLO image score = max box confidence (0 if no box).
- Reported: mean ± std over 3 seeds. No hyperparameter is tuned on the held-out set.

## Deployment (last)
- CPU latency (batch 1, i5-12400, median of 200 runs) of both, FP32 vs INT8 (OpenVINO / NNCF PTQ),
  and the AUROC change after INT8.

## Known limits, stated up front
- 640 px for YOLO and 256 px for PatchCore down-sample ~1.5k px PCB images; small defects may vanish.
  Both use their library defaults; resolution is not tuned.
- Image-level metrics only; localisation quality is not compared.

## Change log
- 2026-11-08, before any result: PatchCore originally used all ~900 normal train images. Its memory bank
  (~925k patches x 1,536 dims, ~5.7 GB, doubled while concatenating) overflowed the 12 GB GPU and the run
  stalled. PatchCore now uses 200 normal images per seed, and YOLO's background images were changed to the
  same draw, so both methods see identical normal data.
- 2026-11-08, after the accuracy grid, before any deployment result was used: anomalib's `INT8_PTQ` calibrates
  on the datamodule's val loader, which here is the held-out test set. PatchCore INT8 is therefore quantized
  directly with `nncf.quantize` (same default settings) on the 200 normal train images. YOLO is unaffected
  (ultralytics calibrates on `data.yaml` val = its own training images).
- 2026-11-08, **post-hoc**, added after seeing PatchCore INT8 lose AUROC on pcb1: a second PatchCore INT8 variant
  (`int8_bb`) that keeps the three memory-bank distance MatMuls in FP32 and quantizes only the backbone.
  It is reported separately and labelled post-hoc.
