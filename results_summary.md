# Results summary (real measured numbers) - PVEL-AD EL inspection on Raspberry Pi 5

Status: 20 Nov 2026, 21:00. All numbers below were measured by us. Items marked TODO are not measured yet.

## Dataset (PVEL-AD, EL2021) and splits
- 36,543 images total: trainval 4,500 labelled defective; test 19,150 (all defective, labels from test_annotation); good 11,353; "meterial" 1,540 (not used).
- 12 original defect classes grouped into 7: crack (crack + star_crack), finger, black_core, thick_line, short_circuit, dislocation (horizontal + vertical), other (corner, fragment, scratch, printing_error).
- Train 6,375 images (3,825 defective, 2,550 good). Validation 1,125 (675 defective, 450 good). Seeded 85/15 split of trainval + 3,000 good cells.
- Test 5,500 images: 4,000 defective sampled from the official test split + 1,500 good cells that training never used. The official test split has no defect-free cells, so good cells came from the `good` folder (a different source: a limitation).
- Images per class, train: crack 878, finger 1,272, black_core 859, thick_line 659, short_circuit 418, dislocation 255, other 16.
- Images per class, test: crack 491, finger 2,503, black_core 783, thick_line 257, short_circuit 251, dislocation 144, other 5.
- Images are 1024x1024 grayscale, resized to 384x384 (most defects cover only 0.2-0.6% of a cell).

## Model and training
- MobileNetV3-Small (ImageNet-pretrained, first conv adapted to 1 channel), 8 sigmoid outputs (7 defect groups + "defective").
- BCE with class weights, AdamW, one-cycle LR 1e-3, batch 32, 12 epochs on a laptop CPU (about 2.5 hours). Best epoch chosen by mean validation AUC (epoch 11, 0.9875). Per-class thresholds tuned on validation only.

## Accuracy on the test split (5,500 cells; thresholds from validation)
Defective vs good: accuracy 93.1%, AUC 0.980, precision 0.984, recall 0.920, F1 0.951.
False alarms: 60 of 1,500 good cells (4.0%). Missed: 318 of 4,000 defective cells (8.0%).

| Class | Test support | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| crack | 491 | 0.839 | 0.560 | 0.487 | 0.521 |
| finger | 2,503 | 0.978 | 0.958 | 0.891 | 0.924 |
| black_core | 783 | 0.998 | 0.990 | 0.908 | 0.947 |
| thick_line | 257 | 0.969 | 0.666 | 0.782 | 0.719 |
| short_circuit | 251 | 1.000 | 0.869 | 1.000 | 0.930 |
| dislocation | 144 | 1.000 | 0.911 | 1.000 | 0.954 |
| other | 5 | 0.822 | 0.000 | 0.000 | 0.000 (too few images to judge) |

Weakest: crack (F1 0.52; validation F1 was 0.76, so it generalises poorly) and thick_line (0.72).

### Same test split (5,500 cells), FP32 vs INT8 ONNX (INT8 uses its own validation-tuned thresholds)
| Defective vs good | FP32 | INT8 |
|---|---|---|
| Accuracy | 93.1% | 93.2% |
| DEFECTIVE AUC | 0.980 | 0.975 |
| Precision / recall / F1 (defective) | 0.984 / 0.920 / 0.951 | 0.977 / 0.929 / 0.952 |
| False alarms (of 1,500 good) | 60 (4.0%) | 88 (5.9%) |
| Missed (of 4,000 defective) | 318 (8.0%) | 285 (7.1%) |

| Class (F1, test) | FP32 | INT8 |
|---|---|---|
| crack | 0.521 | 0.405 |
| finger | 0.924 | 0.923 |
| black_core | 0.947 | 0.880 |
| thick_line | 0.719 | 0.702 |
| short_circuit | 0.930 | 0.881 |
| dislocation | 0.954 | 0.986 |
INT8 matches FP32 on the headline defective-vs-good decision but is weaker on the hard classes (crack most). The INT8 model's scores are slightly shifted, so INT8 needs its own thresholds: with the FP32 thresholds, black_core recall was only 0.29 on a 300-cell sample; with INT8-specific thresholds tuned on validation it was 0.91.

## Edge optimisation: ONNX and INT8
- Exported to ONNX (opset 17), input 1x1x384x384.
- Naive INT8 (all layers, ONNX Runtime static quantisation) collapsed: AUC 0.98 -> about 0.7 (0.67 to 0.72 depending on the run and split).
- Cause found by a per-layer scan: only the first conv and the first depthwise conv are sensitive (AUC drop 0.25 and 0.30). All other 50 convs lose nothing measurable.
- Fix: keep those two layers in FP32, quantise the rest (static, QDQ, per-channel weights, percentile 99.999 calibration on 128 validation images). Calibration images were never used to choose or report results.
- Another pitfall found: ONNX Runtime's quant_pre_process step corrupted quantisation for this model; quantising from the original export fixed it.

## Raspberry Pi 5 benchmark (4 GB, CPU only, ONNX Runtime 1.32, 300-image test sample: 150 defective + 150 good)
| | FP32 | INT8 |
|---|---|---|
| Model size | 5.83 MB | 1.77 MB (3.3x smaller) |
| Latency, 4 threads | 14.8 ms = 67 FPS | 11.4 ms = 87 FPS (1.3x faster) |
| Latency, 1 thread | 29.3 ms = 34 FPS | 16.0 ms = 62 FPS (1.8x faster) |
| Process RAM | 290 MB | 289 MB |
| Defective-vs-good accuracy (300 sample) | 97.0% | 95.0% |
| DEFECTIVE AUC (300 sample) | 0.993 | 0.988 |
INT8 agrees with FP32 on the "defective" decision for 96% of images.
Same models on a Windows PC (x86, 4 threads): FP32 2.4 ms, INT8 2.9 ms. INT8 is slower on x86 but faster on the Pi's Arm CPU.
Pi health during the test: 38.4 C idle, no throttling (throttled=0x0).
Notes: latency is model inference only (no camera capture, resize or report drawing). RAM includes the 44 MB sample array held by the benchmark script.
The 300-image sample accuracy (97.0% / 95.0%) differs from the 5,500-image figure (93.1%) because the sample is balanced 50/50 and the test set is 73% defective.

## Hardware
Raspberry Pi 5 Model B Rev 1.1, 4 GB RAM, Debian (kernel 6.18, aarch64), Python 3.13. Camera: Raspberry Pi Camera Module 3 (imx708), standard (not NoIR), so it cannot capture real EL.

## TODO (not measured yet)
- End-to-end Pi frame rate with camera capture + preprocessing + report.
- Camera capture on the Pi (cable problem being fixed).
- Demo video, GitHub repo link.
