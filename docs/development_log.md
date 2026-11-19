
## 2026-11-15 - Resolution test on grouped split (YOLO26n-OBB, batch 2, nbs 4, 60 epochs)
- 640:  val mAP50 0.811, mAP50-95 0.630, P 0.846, R 0.768 (runs/train/grouped_640)
- 1024: val mAP50 0.853, mAP50-95 0.695, P 0.894, R 0.832 (runs/train/grouped_1024, 4.67 h, best epoch 54)
- Biggest gains: spur, mouse_bite, missing_pad. short slightly lower (45 boxes).
- Laptop CPU inference: ~183 ms/img at 1024 vs ~82 ms at 640 (not phone numbers).
- v1 weights frozen in detection/weights/ with SHA256SUMS.txt. Test split still unused.

## 2026-11-15 - Test split evaluation (once, frozen v1 weights, grouped split: 33 images, 377 boxes)
- 1024: P 0.900, R 0.762, mAP50 0.828, mAP50-95 0.655 (val: 0.894 / 0.832 / 0.853 / 0.695)
- 640:  P 0.887, R 0.702, mAP50 0.782, mAP50-95 0.601 (val via evaluate.py: 0.849 / 0.757 / 0.811 / 0.630)
- 1024 advantage reproduced on test: +0.046 mAP50, +0.054 mAP50-95, +0.060 recall.
- Weakest class: mouse_bite (1024 recall 0.50, mAP50 0.653).
- Laptop CPU inference: 104 ms (1024) vs 44 ms (640), not phone numbers.
- Correction: earlier 640 val P/R (0.846/0.768) came from batch-4 validation; evaluate.py gives 0.849/0.757.
- Limits: scanner images only, 3 test groups, dataset has no defect-free boards.

## 2026-11-15 - Operating point (val, rotated IoU 0.5) and QC rules v1
- 1024: best F1 0.843 at conf 0.40 (P 0.898, R 0.795, 0.94 false alarms and 2.14 missed per board).
- 640: best F1 0.772 at conf 0.25 (P 0.819, R 0.730). 1024 is better at every threshold.
- Detections below 0.40 are ~80% false alarms; 0.40-0.50 about 50% real; >=0.60 about 95% real.
- QC v1 (quality_control/rules.py): HOLD >= 0.50, REVIEW 0.40-0.50, PASS otherwise. No severity yet.
- Limit: dataset has no clean boards, so false alarms on good boards are unmeasured.

## 2026-11-19 - Phone-like degradation test (frozen 1024 model, val copies, labels unchanged)
- mAP50-95: control 0.694, color -0.027, blur -0.003, noise +0.001, glare 0.000, jpeg -0.005, all -0.036.
- Under 'all': recall 0.825 -> 0.750; mouse_bite 0.716 -> 0.585, spur 0.797 -> 0.667. Precision steady.
- Proxy only: no geometric change, mild blur, no real boards or phone photos.
