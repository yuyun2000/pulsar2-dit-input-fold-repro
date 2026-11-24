#!/usr/bin/env python3
"""Run the compiled axmodel on an AX650 board and check whether it uses its x input.

Usage (on the board, python3 + numpy only):
    python3 test_axmodel.py --model assets/dit_p11.axmodel \
        --onnx-ref assets/p11_ref.npy --input assets/input.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ax_engine import AxEngine  # noqa: E402


def cos(a, b):
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--onnx-ref", required=True)
    ap.add_argument("--input", default="assets/input.npz")
    args = ap.parse_args()

    d = np.load(args.input)
    ref = np.load(args.onnx_ref)
    eng = AxEngine(args.model)

    def run(x):
        shape, data = eng.run([x, d["cond"], d["mu"], d["te"]])[0]
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()

    real = run(d["x"])
    zero = run(np.zeros_like(d["x"]))

    print("model                :", os.path.basename(args.model))
    print("axmodel vs ONNX ref  : %.6f" % cos(real, ref))
    print("axmodel real-vs-zero : %.6f" % cos(real, zero))
    print("std axmodel / ONNX   : %.4f / %.4f" % (float(real.std()), float(ref.std())))
    print()
    print("expected (dit_p11)   : ~0.15 vs ONNX, ~0.99996 real-vs-zero  <-- x ignored")
    print("control  (dit_p8)    : ~0.64 vs ONNX, ~0.481   real-vs-zero  <-- x used")


if __name__ == "__main__":
    main()
