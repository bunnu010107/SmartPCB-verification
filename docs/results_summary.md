# SmartPCB results summary

All numbers come from runs logged in this repository (docs/development_log.md).

## System
Capture -> detection (YOLO26n-OBB, ONNX) -> rotated NMS -> deterministic QC rules (HOLD / REVIEW / PASS)
-> JSON report, annotated image, SQLite inspection history.
Implemented and tested on a laptop. The ONNX model was benchmarked on the target phone (Alcatel T519H,
Dimensity 6300). There is no Android app yet.

## Data and split
- Dataset: PCB-Defect V2 Reannotated (Mendeley Data): 230 scanner images of lab-made single-layer boards,
  2,506 oriented boxes, 6 classes (missing_pad, mouse_bite, open_circuit, short, spur, spurious_copper).
- Grouped split (images sharing the first filename part stay together):
  train 162 images / 16 groups, val 35 / 3, test 33 / 3. The test split was evaluated once.

## Model and training
- YOLO26n-OBB, 2.45 M parameters, 5.6 GFLOPs at 640 px and 14.7 GFLOPs at 1024 px.
- AdamW, lr 0.001, 60 epochs, seed 42, CPU only (1024 px: 4.67 h with batch 2 / nbs 4; 640 px: 1.03 h).
- Early runs failed because Ultralytics accumulated 64 images per update at batch 1 (about 25 updates in
  total). Setting nbs equal to the real batch fixed it.

## Detection accuracy (best checkpoint)
| Split | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| val | 640 | 0.849 | 0.757 | 0.811 | 0.630 |
| val | 1024 | 0.894 | 0.832 | 0.853 | 0.695 |
| test | 640 | 0.887 | 0.702 | 0.782 | 0.601 |
| test | 1024 | 0.900 | 0.762 | 0.828 | 0.655 |

Test per class, 1024 px:
| Class | Boxes | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| missing_pad | 50 | 0.819 | 0.880 | 0.762 |
| mouse_bite | 126 | 0.500 | 0.653 | 0.473 |
| open_circuit | 43 | 0.907 | 0.914 | 0.711 |
| short | 45 | 0.717 | 0.807 | 0.641 |
| spur | 72 | 0.681 | 0.742 | 0.557 |
| spurious_copper | 41 | 0.951 | 0.973 | 0.786 |

## ONNX export
- ONNX validation mAP50-95: 1024 px 0.699 (PyTorch 0.695), 640 px 0.635 (PyTorch 0.630).
- Model output is (1, 11, N): cx, cy, w, h, 6 class scores, angle (radians). Thresholding and rotated NMS
  are done outside the model (detection/onnx_infer.py).
- Standalone decode vs PyTorch on the 35 val images at confidence 0.40: 317 of 324 PyTorch detections found,
  317 of 331 ONNX detections confirmed, mean confidence difference 0.016.

## Quality control (validation split, 1024 px)
- Rules v1: HOLD if any detection >= 0.50, REVIEW if the best is 0.40-0.50, otherwise PASS. No severity assigned.
- At confidence 0.40: precision 0.898, recall 0.795, 0.94 false alarms and 2.14 missed defects per board.
- At confidence 0.50: precision 0.936, recall 0.757, 0.54 false alarms and 2.54 missed defects per board.
- Detections below 0.40 were about 80% false alarms on clean validation images.

## Robustness proxy (simulated phone degradation, validation copies, labels unchanged)
| Variant | mAP50-95 | Change |
|---|---:|---:|
| control | 0.694 | |
| color | 0.667 | -0.027 |
| blur | 0.691 | -0.003 |
| noise | 0.695 | +0.001 |
| glare | 0.694 | 0.000 |
| jpeg | 0.689 | -0.005 |
| all combined | 0.658 | -0.036 |
Under all effects combined, about 1 extra defect per board is missed at confidence 0.40 (3.11 vs 2.11) and the
best-F1 threshold moves from 0.40 to 0.30.

## Speed and memory
| Device | Model | Threads | Mean latency | Memory (RSS) |
|---|---|---:|---:|---:|
| Laptop (Ryzen 5 5600H, ONNX Runtime CPU) | 640 | 4 | 36.2 ms | 168 MB |
| Laptop | 1024 | 4 | 128.1 ms | 323 MB |
| Alcatel T519H (Dimensity 6300: 2 x A76 + 6 x A55) | 640 | 2 | 189 ms | 145 MB |
| Alcatel T519H | 640 | 4 | 228 ms | 145 MB |
| Alcatel T519H | 1024 | 2 | 569 ms | 249 MB |
| Alcatel T519H | 1024 | 4 | 589 ms | 249 MB |
Phone measurements: Termux, ONNX Runtime 1.32.0 CPU execution provider, random input, model only, 20 timed runs.
Two threads is fastest on the phone (the two fast cores).

## Limitations (not claimed)
- All accuracy numbers are on scanner images of lab-made single-layer boards; 3 validation and 3 test groups only.
- No physical-board phone photos have been evaluated. One photo of an on-screen image produced a false alarm on
  a viewer icon outside the board, so a board-localisation step is needed.
- The dataset has no defect-free boards, so false alarms on good boards are unmeasured. PASS means "no detection
  above threshold", not "board is good".
- Thresholds were tuned on scanner images and need re-calibration for phone captures.
- Phone latency is model-only; camera, decoding, resizing, NMS and UI time are not included.
- No Android app has been built; the Python reference (detection/onnx_infer.py) specifies the port.
