# Solar-cell EL inspection assistant on Raspberry Pi 5

Arm Bharat AI-SoC Challenge 2026-27, problem statement PS5: *Real-time industrial defect detection and quality control assistant* (theme: Applied AI on SoC).

**Demo video:** https://drive.google.com/file/d/1EX1ci5akcCfGneSNC4i1A7nLsg_3FYG2/view?usp=drivesdk

An edge AI assistant that looks at an electroluminescence (EL) image of a solar cell, decides whether it is defective, names the defect type(s), and produces a health report (fault type, severity, suggested action). Everything runs on-device, CPU-only, on a Raspberry Pi 5 (4 GB, no accelerator), with no cloud.

It is two pipelines on one board:
- **Pipeline A, vision (ONNX Runtime):** a MobileNetV3-Small classifier finds the defect types. We compare an FP32 model with an INT8-quantised model.
- **Pipeline B, advisory (Llama 3.2 1B Instruct on ExecuTorch):** the vision result is turned into a small structured JSON (no pixels). Llama 3.2 1B (SpinQuant INT4, `.pte`) runs through the ExecuTorch `llama_main` runner on the Pi CPU and writes a short Cause / Impact / Action explanation.

Rules, not Llama, decide pass/fail and severity (Low / Medium / High). Llama only explains. If Llama is missing, too slow or its text fails a check (see "AI advisory"), the dashboard falls back to rule-based text and says so.

Built with Llama.

Team: Pragadeesh and teammate, PSG Institute of Technology and Applied Research, Coimbatore.

## Results (measured)
Test split: 5,500 cells never used for training or threshold tuning (4,000 defective from the official PVEL-AD test split + 1,500 defect-free cells).

| | FP32 | INT8 |
|---|---|---|
| Defective-vs-good accuracy | 93.1% | 93.2% |
| AUC | 0.980 | 0.975 |
| False alarms (of 1,500 good) | 60 (4.0%) | 88 (5.9%) |
| Missed (of 4,000 defective) | 318 (8.0%) | 285 (7.1%) |
| Model size | 5.83 MB | 1.77 MB |
| Raspberry Pi 5 latency, 4 threads | 14.8 ms (67 FPS) | 11.4 ms (87 FPS) |
| Raspberry Pi 5 latency, 1 thread | 29.3 ms (34 FPS) | 16.0 ms (62 FPS) |
| Process RAM on the Pi | 290 MB | 289 MB |

Latency is model inference only (no camera capture or drawing). Per-class results, the dataset splits and notes are in `results_summary.md`.

Weak spots (be aware): `crack` (F1 0.52 FP32 / 0.41 INT8) and `thick_line` (0.72 / 0.70). The `other` group has only 16 training images, so its scores are not meaningful.

## AI advisory results (measured on the Pi 5, Llama 3.2 1B Instruct SpinQuant INT4, ExecuTorch, CPU)
| Measure | Value |
|---|---|
| Model file | 1.1 GB `.pte` |
| Model load | about 2.7 s per advisory |
| Prompt size | about 365 to 430 tokens |
| Time to first token | about 3.3 to 4.1 s |
| Generation speed | about 6.1 to 7 tok/s with the full prompt (about 12.9 tok/s with a 13-token prompt) |
| Total per new advisory | about 10 to 12 s (identical findings reuse a cached advisory) |
| Llama peak RAM | about 1.64 GB |
| Vision app running together | 12.9 ms inference, 66.7 FPS pipeline, 183 MB app memory |
| Board | 46 to 52 C, no throttling (`throttled=0x0`), swap not used |

Generation is slower with a longer prompt (12.9 tok/s at 13 prompt tokens, 11.7 at 63, about 7 at 317). We believe the longer context is the cause but did not isolate it. Numbers are from our own runs and logs (`advisory_log.jsonl`).

Limits of the advisor: a 1B model can still write a vague or slightly wrong sentence. We add safeguards (below), but its text is advice for a human, not a measurement. When the classifier says "defective" but names no defect type, Llama can only say the type is unclear and recommend manual inspection.

## How it works
1. **Data**: PVEL-AD (1024x1024 EL cell images, XML boxes). The 12 defect classes are grouped into 7: crack, finger, black_core, thick_line, short_circuit, dislocation, other. Images are resized to 384x384 grayscale (defects are small).
2. **Model**: MobileNetV3-Small, 1-channel input, 8 sigmoid outputs (7 defect groups + "defective"), trained on a laptop CPU.
3. **INT8**: ONNX Runtime static quantisation. Naive INT8 collapsed (AUC 0.98 to 0.70). A per-layer scan found only the first conv and the first depthwise conv are sensitive; keeping those two in FP32 and using percentile calibration keeps accuracy. INT8 uses its own validation-tuned thresholds.
4. **Health report**: severity from defect type (black_core, short_circuit = High; crack, dislocation, thick_line, other = Medium; finger = Low; 3 or more types raise it one level), plus a suggested action. These are our own engineering rules, not an industry standard.

## AI advisory (pipeline B)
1. `pvel_app.py` builds a findings JSON per defective cell: result, severity, confidence, defect types with a plain meaning and a typical cause from our own table, and the rule-based action. Scope is stated as "whole-cell classification, defect position unknown". The image is never sent to Llama.
2. `advisor.py` builds a Llama 3 chat prompt (system rules, one worked example, the findings) and runs `llama_main` as a subprocess (temperature 0, so the same findings give the same text).
3. The answer must contain Cause, Impact and Action lines. A **number guard** rejects any line that contains a number not present in the findings. Failed lines are replaced by rule text, marked "partly rule-based"; a timeout or missing runner gives full "rule-based" text.
4. Results are cached per (severity, fault types), run latest-first so the dashboard never waits on old cells, and written to `advisory_log.jsonl` with timings.
5. The dashboard shows the advisory, the source badge (llama / llama (cached) / rule-based), the timings and the exact JSON that was sent.

### Set up ExecuTorch + Llama (build on a Linux x86_64 PC, run on the Pi)
We followed the ExecuTorch Raspberry Pi route (release/1.0): on an x86_64 Linux host, clone ExecuTorch and run `examples/raspberry_pi/setup.sh pi5` to cross-compile `llama_main` and `libllama_runner.so` for the Pi 5. With CMake 4.x we needed `CMAKE_POLICY_VERSION_MINIMUM=3.5`. On Debian 13 (trixie) do not apply the GLIBC workaround from the tutorial.

Download the ready-made model from Hugging Face: `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` (`Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte`, `tokenizer.model`, `params.json`). Put `llama_main`, `libllama_runner.so`, the `.pte` and `tokenizer.model` in `~/executorch-deployment` on the Pi (the app's default `--llama-dir`).

    python llama_selftest.py            # 5 fixed cases through Llama, checks format and guard

### App options
    --llama-dir DIR           folder with llama_main, .pte and tokenizer.model (default ~/executorch-deployment)
    --llama-model PATH        .pte file      --llama-tokenizer PATH
    --llama-timeout SEC       per-advisory limit before falling back (default 120)
    --llama-temperature T     default 0.0
    --llama-wait              pause the loop until each advisory is ready (best for recording a demo)
    --no-llama                rule-based text only

## Reproduce (laptop, Windows or Linux)
Dataset: download PVEL-AD (see "Data" below) and extract so the folder `.../PVELAD/EL2021` has `trainval`, `test`, `othertypes`. Test XMLs come from `test_annotation.rar` (extract to a separate folder).

    python -m venv .venv
    .venv\Scripts\activate            (Linux: source .venv/bin/activate)
    pip install -r requirements.txt

    python prep_pvel.py --root <...>/PVELAD/EL2021 --out data/pvel --test-ann <test_annotation folder>
    python prep_pvel.py --root <...>/PVELAD/EL2021 --out data/pvel_test --test-ann <test_annotation folder> --only-test
    python train_pvel.py --data data/pvel --epochs 12        # ~2.5 h on a laptop CPU
    python eval_pvel.py --data data/pvel_test                # FP32 PyTorch test results
    python export_onnx.py --data data/pvel_test --calib data/pvel
    python quantize_sweep.py --val data/pvel --test data/pvel_test    # builds the INT8 model + thresholds
    python eval_onnx.py --data data/pvel_test --bundle pi_bundle      # FP32 vs INT8 on the full test split

## Run on the Raspberry Pi 5
Copy the `pi_bundle` folder to the Pi (`scp -r pi_bundle <user>@<pi>:~/`), then on the Pi:

    python3 -m venv --system-site-packages ~/venv
    source ~/venv/bin/activate
    pip install -r requirements-pi.txt
    cd ~/pi_bundle
    python pi_check.py                                              # health, camera and runtime check
    python bench_onnx.py --models pvel_fp32.onnx pvel_int8.onnx --threads 4 --tag pi5
    python pvel_app.py --app --serve 8000 --smooth 3                # dashboard buttons: Camera / Demo cells / Upload image
    python pvel_app.py --demo --serve 8000                          # bundled test cells, results page on port 8000
    python pvel_app.py --camera --serve 8000                        # live camera (see the note below)
    bash install_autostart.sh                                       # optional: start the app at every boot (service el-inspect)

Open `http://<pi>.local:8000` on another device to see the live result page.

**Important limitation:** a standard Raspberry Pi camera cannot capture real EL (that needs a near-infrared camera and a forward-biased cell). In `--camera` mode the camera photographs an EL image shown on a screen or print. This demonstrates the edge pipeline, not a real EL capture station.

## Known limitations
- No defect location: the model classifies the whole cell, so the app does not draw boxes.
- The camera shows an EL image on a screen; non-EL scenes are out of distribution and are shown as defective.
- INT8 misses some defects (about 7% of defective cells in the test split).
- The 1B advisor can be vague or occasionally wrong; the dashboard labels its source and keeps the rule-based action beside it.
- Only 4 GB RAM: Llama needs about 1.6 GB while running, which fits next to the vision app (swap stayed at 0 in our tests).

## Files
- `prep_pvel.py`, `train_pvel.py`, `eval_pvel.py`: data preparation, training, evaluation (PyTorch)
- `export_onnx.py`, `quantize_sweep.py`, `eval_onnx.py`, `bench_onnx.py`: ONNX export, INT8 quantisation, evaluation, benchmark
- `pvel_app.py`: the inspection assistant (health report, results page)
- `advisor.py`: pipeline B, Llama 3.2 1B on ExecuTorch (prompt, runner, number guard, fallback, log)
- `llama_selftest.py`: runs 5 fixed cases through the advisor on the Pi
- `install_autostart.sh`: systemd service so the app starts at boot
- `pi_check.py`: Pi health / camera / runtime check
- `pi_bundle/`: models, thresholds and sample for the Pi
- `results_summary.md`: all measured numbers
- Earlier experiments (not part of the final system): `train_el.py`, `train_el_mixed.py`, `eval_el_external.py`, `build_demo_pairs.py`, `prep_rgb.py`, `train_rgb.py` (ELPV / Roboflow / Mendeley EL tests and an RGB fault classifier we dropped)

## Data and credits
- PVEL-AD: B. Su, Z. Zhou, H. Chen, "PVEL-AD: A Large-Scale Open-World Dataset for Photovoltaic Cell Anomaly Detection", IEEE Transactions on Industrial Informatics, 2023, DOI 10.1109/TII.2022.3162846. Check the dataset's own terms before reuse. **The dataset is not included in this repository.**
- MobileNetV3: A. Howard et al., "Searching for MobileNetV3", ICCV 2019. Pretrained ImageNet weights from torchvision.
- ONNX Runtime and PyTorch are used under their own licences.
- **Built with Llama.** Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright (c) Meta Platforms, Inc. All Rights Reserved. We use the community ExecuTorch export `executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET` and the ExecuTorch runtime (see their own licences).
