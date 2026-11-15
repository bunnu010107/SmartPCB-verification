
## 2026-11-15 - Resolution test on grouped split (YOLO26n-OBB, batch 2, nbs 4, 60 epochs)
- 640:  val mAP50 0.811, mAP50-95 0.630, P 0.846, R 0.768 (runs/train/grouped_640)
- 1024: val mAP50 0.853, mAP50-95 0.695, P 0.894, R 0.832 (runs/train/grouped_1024, 4.67 h, best epoch 54)
- Biggest gains: spur, mouse_bite, missing_pad. short slightly lower (45 boxes).
- Laptop CPU inference: ~183 ms/img at 1024 vs ~82 ms at 640 (not phone numbers).
- v1 weights frozen in detection/weights/ with SHA256SUMS.txt. Test split still unused.
