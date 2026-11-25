# SmartPCB – Intelligent PCB Defect Detection & Edge AI Inspection

**AI-Based PCB Defect Detection | Android Edge AI | ONNX Runtime | FPGA Acceleration**

SmartPCB is an intelligent Printed Circuit Board (PCB) inspection system that combines deep-learning-based defect detection, automated quality control, mobile-based inspection, and embedded AI deployment.

The project aims to develop a portable and efficient PCB inspection platform capable of identifying PCB defects, visualizing their locations, assessing defect severity, and eventually performing hardware-accelerated inference using FPGA technology.

---

# 1. Project Overview

SmartPCB integrates three major development components:

1. **AI-Based PCB Defect Detection:** Deep-learning-based object detection for identifying PCB defects.
2. **Android-Based PCB Inspection:** A mobile application using ONNX Runtime for on-device AI inference.
3. **FPGA-Based AI Acceleration:** Planned deployment on the Digilent Zybo Z7-10 using its programmable logic and ARM Cortex-A9 processor.

The ultimate objective is to create an embedded AI inspection system that can perform PCB quality assessment with low inference latency and reduced dependence on cloud computing.

---

# 2. System Architecture

## 2.1 Current Software Architecture

```text
                  PCB IMAGE
                      |
                      v
              IMAGE PREPROCESSING
                      |
                      v
               AI DEFECT DETECTION
                      |
                 ONNX MODEL
                      |
             +--------+--------+
             |                 |
             v                 v
        BOUNDING BOXES    CONFIDENCE SCORES
             |                 |
             +--------+--------+
                      |
                      v
             SEVERITY CLASSIFICATION
                      |
                      v
               INSPECTION RESULTS
                      |
              +-------+-------+
              |               |
              v               v
         PYTHON PIPELINE   ANDROID APP
```

## 2.2 Planned FPGA Architecture

```text
                 PCB IMAGE
                     |
                     v
             ARM CORTEX-A9
              PROCESSING SYSTEM
                     |
                     v
             FPGA PROGRAMMABLE LOGIC
                     |
                     v
               AI ACCELERATOR
                     |
                     v
              DEFECT DETECTION
                     |
                     v
             INSPECTION RESULTS
```

The FPGA implementation will explore hardware acceleration using the programmable logic resources of the Zynq-7000 SoC.

The ARM Cortex-A9 processor will handle system control, data management, and communication with the hardware accelerator.

---

# 3. Project Components

## 3.1 AI-Based PCB Defect Detection

The Python-based detection pipeline supports:

- Dataset analysis and preprocessing.
- YOLO-format annotation preparation.
- Deep-learning model training.
- Model validation and evaluation.
- ONNX model export.
- ONNX inference.
- Inference benchmarking.
- Confidence threshold optimization.
- Prediction visualization.

Important files:

```text
detection/
├── train.py
├── predict.py
├── evaluate.py
├── export_onnx.py
├── onnx_infer.py
├── benchmark_onnx.py
├── operating_point.py
├── overfit_test.py
├── summarize_run.py
└── weights/
```

## 3.2 Dataset Processing

The preprocessing pipeline prepares PCB images and annotations for model training.

Features include:

- Dataset inspection.
- Image preprocessing.
- Annotation visualization.
- Defect crop inspection.
- Grouped dataset splitting.
- Dataset quality verification.
- PCB image inspection and analysis.

Important files:

```text
preprocessing/
├── analyze_dataset.py
├── image_loader.py
├── visualize_annotations.py
├── inspect_defect_crops.py
├── inspect_single_image.py
├── make_grouped_split.py
├── make_phone_like.py
└── create_paired_inspection.py
```

## 3.3 Android PCB Inspection Application

The Android application provides a mobile interface for PCB defect inspection.

Technologies:

- Kotlin
- Android Studio
- Jetpack Compose
- ONNX Runtime
- On-device AI inference

Application objectives:

- PCB image selection.
- AI-based defect detection.
- Bounding-box visualization.
- Confidence-based predictions.
- Mobile PCB inspection.
- On-device model execution.

Model location:

```text
app/src/main/assets/smartpcb_model.onnx
```

The Android application uses the exported ONNX model to perform inference locally.

## 3.4 Quality Control Module

The quality-control subsystem provides rule-based inspection and defect severity assessment.

```text
quality_control/
├── __init__.py
├── database.py
├── rules.py
└── config_v1_1024.json
```

Automated tests:

```text
tests/
├── test_database.py
└── test_rules.py
```

---

# 4. Technologies Used

| Category                | Technology                  |
| ----------------------- | --------------------------- |
| Programming Languages   | Python, Kotlin              |
| Deep Learning           | YOLO-based Object Detection |
| Computer Vision         | OpenCV, Pillow              |
| Model Deployment        | ONNX Runtime                |
| Mobile Development      | Android Studio              |
| UI Framework            | Jetpack Compose             |
| Dataset Format          | COCO, YOLO                  |
| Version Control         | Git, GitHub                 |
| Development Environment | Linux, Windows              |
| GPU Computing           | NVIDIA CUDA                 |
| Target FPGA             | Digilent Zybo Z7-10         |
| Embedded Processor      | ARM Cortex-A9               |
| FPGA Architecture       | Xilinx Zynq-7000            |
| Hardware Acceleration   | Programmable Logic          |

---

# 5. Repository Structure

```text
SmartPCB-verification/
│
├── app/
│   └── src/
│       └── main/
│           ├── assets/
│           │   └── smartpcb_model.onnx
│           └── java/
│               └── ... Android source files
│
├── detection/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── export_onnx.py
│   ├── onnx_infer.py
│   ├── benchmark_onnx.py
│   ├── operating_point.py
│   └── weights/
│
├── preprocessing/
│   ├── analyze_dataset.py
│   ├── image_loader.py
│   ├── visualize_annotations.py
│   ├── inspect_defect_crops.py
│   └── make_grouped_split.py
│
├── quality_control/
│   ├── database.py
│   ├── rules.py
│   └── config_v1_1024.json
│
├── datasets/
│   ├── README.md
│   └── splits/
│       └── v2_obb_grouped_manifest.csv
│
├── docs/
│   ├── architecture.md
│   ├── development_log.md
│   └── results_summary.md
│
├── tests/
│   ├── test_database.py
│   └── test_rules.py
│
├── inspect_board.py
├── README.md
└── .gitignore
```

---

# 6. Dataset Statistics

The project uses PCB defect datasets, including the DsPCBSD dataset, for training and evaluation.

## 6.1 Dataset Distribution

| Parameter                          |       Value |
| ---------------------------------- | ----------: |
| Primary Dataset                    |     DsPCBSD |
| Training Images                    |       8,208 |
| Validation Images                  |       2,051 |
| Total Images                       |      10,259 |
| Annotation Format                  | COCO / YOLO |
| Unified YOLO Annotations Inspected |       2,506 |
| Annotation Coordinate Errors       |           0 |
| Annotation Validation Status       |      PASSED |

**Note:** The 2,506 annotations represent the inspected unified YOLO annotation set. They should not be interpreted as the total number of annotations in the complete dataset.

## 6.2 Dataset Quality Verification

| Validation Parameter                 | Result |
| ------------------------------------ | -----: |
| Total Annotations Inspected          |  2,506 |
| Invalid Annotations                  |      0 |
| Coordinates Outside Normalized Range |      0 |
| Annotation Validation Errors         |      0 |
| Bounding Box Visual Inspection       | PASSED |

All 2,506 inspected annotations passed the coordinate validation checks.

Bounding-box visualization was also performed to verify the annotation placement.

---

# 7. Experimental Results and Performance Analysis

This section summarizes the initial model training experiment, computational requirements, dataset validation, and deployment performance.

## 7.1 Initial Model Training Results

The initial defect detection model was trained for 30 epochs using a CPU-based training experiment.

| Parameter                |  Experimental Result |
| ------------------------ | -------------------: |
| Training Epochs          |                   30 |
| Total Training Time      |         11.259 hours |
| Total Runtime            |     40,532.4 seconds |
| Average Time per Epoch   |        22.52 minutes |
| Training Configuration   |                  CPU |
| Best Model Checkpoint    |              best.pt |
| Last Model Checkpoint    |              last.pt |
| Reported Checkpoint Size | Approximately 5.5 MB |
| Training Status          |            COMPLETED |

The training time is calculated from the recorded experimental duration.

## 7.2 Computational Performance

| Parameter              | Result               |
| ---------------------- | -------------------- |
| CPU Training           | Completed            |
| CPU Training Runtime   | 11.259 hours         |
| GPU Availability       | NVIDIA GPU available |
| NVIDIA Driver Version  | 580.82.07            |
| Reported CUDA Version  | 13.0                 |
| ONNX Export Pipeline   | Implemented          |
| ONNX Inference Script  | Implemented          |
| ONNX Benchmark Script  | Available            |
| CPU Inference Latency  | Pending Measurement  |
| GPU Inference Latency  | Pending Measurement  |
| Peak RAM Consumption   | Pending Measurement  |
| GPU Memory Consumption | Pending Measurement  |

**Important:** GPU availability in the development environment does not mean that the reported 30-epoch experiment used GPU acceleration

## 7.3 Model Evaluation Metrics

The following metrics will be populated using the final validation and evaluation reports.

| Evaluation Metric            | Result                   |
| ---------------------------- | ------------------------ |
| Precision                    | Pending Final Evaluation |
| Recall                       | Pending Final Evaluation |
| mAP\@50                      | Pending Final Evaluation |
| mAP\@50-95                   | Pending Final Evaluation |
| F1-Score                     | Pending Final Evaluation |
| Confusion Matrix             | Pending Final Evaluation |
| Optimal Confidence Threshold | Pending Final Evaluation |

## 7.4 Training Time Analysis

The initial CPU experiment establishes a baseline for future optimization.

| Performance Parameter |            Value |
| --------------------- | ---------------: |
| Total Epochs          |               30 |
| Total Training Time   |     11.259 hours |
| Average Epoch Time    |    22.52 minutes |
| Total Runtime         | 40,532.4 seconds |

Future experiments will investigate model optimization and accelerated training configurations.

## 7.5 Inference Performance Comparison

The project targets multiple deployment platforms.

| Performance Metric  | Python CPU             | NVIDIA GPU               | Android                | Zybo Z7-10          |
| ------------------- | ---------------------- | ------------------------ | ---------------------- | ------------------- |
| Inference Framework | ONNX Runtime / PyTorch | ONNX Runtime / PyTorch   | ONNX Runtime           | Planned Accelerator |
| Inference Latency   | Pending                | Pending                  | Pending                | Pending             |
| FPS                 | Pending                | Pending                  | Pending                | Pending             |
| CPU Utilization     | Pending                | Pending                  | Pending                | Pending             |
| Memory Usage        | Pending                | Pending                  | Pending                | Pending             |
| Power Consumption   | Pending                | Pending                  | Pending                | Pending             |
| Deployment Status   | Available              | Available for evaluation | Initial implementation | Planned             |

All inference comparisons will use the same model, input dimensions, and equivalent measurement methodology wherever possible.

## 7.6 Android Application Performance

| Parameter            | Result               |
| -------------------- | -------------------- |
| Application Platform | Android              |
| Programming Language | Kotlin               |
| UI Framework         | Jetpack Compose      |
| Inference Framework  | ONNX Runtime         |
| Model Format         | ONNX                 |
| Model Location       | app/src/main/assets/ |
| Image Processing     | Mobile Device        |
| Inference Latency    | Pending Measurement  |
| Memory Consumption   | Pending Measurement  |
| Battery Consumption  | Pending Measurement  |

## 7.7 FPGA Hardware Performance

The Digilent Zybo Z7-10 is the target platform for future hardware acceleration.

| Hardware Metric                    | Target Measurement |
| ---------------------------------- | ------------------ |
| ARM Cortex-A9 Inference Latency    | To Be Measured     |
| FPGA Accelerated Inference Latency | To Be Measured     |
| LUT Utilization                    | To Be Measured     |
| Flip-Flop Utilization              | To Be Measured     |
| BRAM Utilization                   | To Be Measured     |
| DSP Utilization                    | To Be Measured     |
| Accelerator Clock Frequency        | To Be Measured     |
| Power Consumption                  | To Be Measured     |
| CPU Workload Reduction             | To Be Measured     |
| FPGA Speedup Factor                | To Be Calculated   |

The hardware speedup factor will be calculated as:

**Speedup Factor = ARM Software Inference Time / FPGA Accelerated Inference Time**

The comparison will be performed using equivalent inference workloads.

---

# 8. FPGA-Based AI Acceleration – Future Development

A major objective of SmartPCB is to deploy the trained AI model on the **Digilent Zybo Z7-10 development board**.

The board integrates a Xilinx Zynq-7000 SoC containing:

- Dual-core ARM Cortex-A9 Processing System.
- Programmable Logic based on Artix-7 FPGA fabric.

## 8.1 Planned Hardware Architecture

| Component               | Responsibility                         |
| ----------------------- | -------------------------------------- |
| ARM Cortex-A9           | System control and software processing |
| FPGA Programmable Logic | Hardware acceleration                  |
| DDR Memory              | Image and model data storage           |
| AXI Interconnect        | Communication between PS and PL        |
| AI Accelerator          | Neural network computation             |
| Processing System       | Input/output and system management     |

## 8.2 FPGA Development Plan

1. Optimize the trained neural network.
2. Investigate model quantization.
3. Convert the model into a hardware-compatible representation.
4. Design FPGA acceleration modules.
5. Integrate the accelerator with the ARM processing system.
6. Implement communication between PS and PL.
7. Benchmark software inference against hardware inference.
8. Measure FPGA resource utilization and power consumption.

## 8.3 Target Outcomes

- Reduced inference latency.
- Improved computational efficiency.
- Reduced ARM processor workload.
- Efficient embedded AI deployment.
- Hardware/software co-design.
- Real-time PCB inspection capability.

**Note:** FPGA acceleration is planned future work. Hardware deployment and performance improvements have not yet been demonstrated.

---

# 9. Development Roadmap

| Phase | Development Activity                            | Status                 |
| ----- | ----------------------------------------------- | ---------------------- |
| 1     | PCB Dataset Collection and Analysis             | Completed              |
| 2     | Dataset Preprocessing                           | Completed              |
| 3     | YOLO Annotation Validation                      | Completed              |
| 4     | Initial AI Model Training                       | Completed              |
| 5     | Model Evaluation and ONNX Export Pipeline       | Implemented            |
| 6     | Android Application Integration                 | Initial Implementation |
| 7     | Android On-Device Inference Testing             | In Progress            |
| 8     | Model Optimization for Different PCB Conditions | Planned                |
| 9     | Zybo Z7-10 FPGA Accelerator Development         | Planned                |
| 10    | ARM and FPGA Integration                        | Planned                |
| 11    | Hardware Performance Benchmarking               | Planned                |
| 12    | Final Embedded System Demonstration             | Planned                |

---

# 10. Getting Started

## 10.1 Clone the Repository

```bash
git clone https://github.com/bunnu010107/SmartPCB-verification.git

cd SmartPCB-verification
```

## 10.2 Python Environment Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Linux:

```bash
source .venv/bin/activate
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies specified by the project's dependency configuration.

## 10.3 Android Application Setup

1. Install Android Studio.
2. Open the cloned repository.
3. Allow Gradle synchronization to complete.
4. Connect an Android device or launch an emulator.
5. Build the application.
6. Run the application.
7. Test PCB image inspection and model inference.

---

# 11. Future Scope

The project can be extended in several directions:

- Improved detection across different PCB colours and layouts.
- Detection under varying illumination conditions.
- Improved generalization to unseen PCB designs.
- Model quantization and pruning.
- Optimized ONNX inference.
- FPGA-based neural network acceleration.
- Real-time embedded PCB inspection.
- Industrial quality-control integration.
- Hardware/software co-design using ARM and FPGA.

---

# 12. Project Objectives

The ultimate goal is to develop a complete intelligent PCB inspection platform combining:

1. Deep-learning-based PCB defect detection.
2. Automated defect severity classification.
3. Portable Android-based inspection.
4. ONNX-based edge inference.
5. FPGA hardware acceleration.
6. ARM-FPGA heterogeneous computing.

The project explores the integration of artificial intelligence, computer vision, mobile computing, embedded systems, and reconfigurable hardware for industrial PCB quality inspection.

---

# 13. Repository Information

| Parameter            | Details                                      |
| -------------------- | -------------------------------------------- |
| Project Name         | SmartPCB                                     |
| Repository           | SmartPCB-verification                        |
| Repository Owner     | bunnu010107                                  |
| Primary Domain       | Artificial Intelligence and Embedded Systems |
| Application Platform | Android                                      |
| AI Deployment        | ONNX Runtime                                 |
| Target Hardware      | Digilent Zybo Z7-10                          |
| Embedded Processor   | ARM Cortex-A9                                |
| Hardware Accelerator | FPGA Programmable Logic                      |

**GitHub Repository:**

[https://github.com/bunnu010107/SmartPCB-verification](https://github.com/bunnu010107/SmartPCB-verification)

---

**SmartPCB – Intelligent PCB Defect Detection & Edge AI Inspection**

*Combining AI, Android Edge Computing, and FPGA Hardware Acceleration for Intelligent PCB Quality Inspection.*
