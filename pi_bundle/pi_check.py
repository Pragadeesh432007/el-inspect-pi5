"""Run this ON THE Raspberry Pi 5 to confirm the stack works (Day 1, takes 2 min).

    pip install onnxruntime numpy pillow          # in a venv
    python pi_check.py            # checks ONNX Runtime + a dummy model + camera

Checks: 1) onnxruntime imports and lists providers  2) a tiny ONNX model runs
        3) camera capture works (picamera2 or OpenCV)  4) CPU temp / throttling
"""
import platform
import subprocess
import time

import numpy as np


def sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "n/a"


print("machine:", platform.machine(), "| python", platform.python_version())
print("model  :", sh("cat /proc/device-tree/model | tr -d '\\0'"))
print("RAM    :", sh("free -m | awk '/Mem:/ {print $2\" MB\"}'"))
print("temp   :", sh("vcgencmd measure_temp"), "| throttled:", sh("vcgencmd get_throttled"))

# 1) + 2) ONNX Runtime with a dummy conv model (built with onnx, no torch needed)
import onnxruntime as ort
print("onnxruntime", ort.__version__, "providers:", ort.get_available_providers())
try:
    import onnx
    from onnx import TensorProto, helper
    W = helper.make_tensor("W", TensorProto.FLOAT, [8, 3, 3, 3],
                           np.random.randn(8, 3, 3, 3).astype(np.float32).flatten().tolist())
    node = helper.make_node("Conv", ["x", "W"], ["y"], pads=[1, 1, 1, 1])
    g = helper.make_graph([node], "dummy",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 224, 224])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 224, 224])],
                          [W])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    sess.run(None, {"x": x})
    t = time.perf_counter()
    for _ in range(20):
        sess.run(None, {"x": x})
    print(f"dummy model OK: {(time.perf_counter()-t)/20*1000:.1f} ms / run")
except ImportError:
    print("(pip install onnx to run the dummy-model check; onnxruntime itself imported fine)")

# 3) camera
try:
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_still_configuration(main={"size": (1280, 720)}))
    cam.start(); time.sleep(1)
    frame = cam.capture_array(); cam.stop()
    print("camera OK (picamera2):", frame.shape)
except Exception as e:
    print("picamera2 not usable:", repr(e)[:120])
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ok, frame = cap.read(); cap.release()
        print("camera (OpenCV):", "OK " + str(frame.shape) if ok else "no frame")
    except Exception as e2:
        print("OpenCV camera not usable:", repr(e2)[:120])

print("temp after:", sh("vcgencmd measure_temp"), "| throttled:", sh("vcgencmd get_throttled"))
