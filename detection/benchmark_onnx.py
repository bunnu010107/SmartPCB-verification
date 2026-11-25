"""Time ONNX Runtime inference on the CPU (laptop baseline for the later phone test).

Usage (from the project root):
    python detection/benchmark_onnx.py --model detection/weights/smartpcb_v1_1024_grouped.onnx --threads 4
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
import psutil


def parse_args():
    p = argparse.ArgumentParser(description="ONNX Runtime CPU benchmark")
    p.add_argument("--model", required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--runs", type=int, default=30)
    return p.parse_args()


def main():
    a = parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = a.threads
    sess = ort.InferenceSession(a.model, sess_options=opts, providers=["CPUExecutionProvider"])

    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.rand(*shape).astype(np.float32)  # random input is fine for timing

    for _ in range(a.warmup):
        sess.run(None, {inp.name: x})

    times = []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        sess.run(None, {inp.name: x})
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)

    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    print(f"model: {a.model}")
    print(f"input shape: {shape} | threads: {a.threads} | runs: {a.runs} (after {a.warmup} warm-up)")
    print(f"latency ms: mean {t.mean():.1f} | median {np.median(t):.1f} | p95 {np.percentile(t, 95):.1f} "
          f"| min {t.min():.1f} | max {t.max():.1f}")
    print(f"process memory (RSS): {rss_mb:.0f} MB")


if __name__ == "__main__":
    main()
