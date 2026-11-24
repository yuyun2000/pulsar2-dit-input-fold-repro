# Pulsar2 8.0: DiT axmodel ignores its main input (input branch folded)

Minimal reproduction for a Pulsar2 8.0 compiler issue on AX650/AX8850.

## Summary

A DiT step graph (22 transformer blocks) compiled by Pulsar2 8.0 produces an
axmodel whose output is **independent of the main input `x`**.  An 8-block
control from the same graph is still input-dependent, so the problem appears
between depth 8 and depth 11.

| model | blocks | ONNX real-vs-zero-x | axmodel real-vs-zero-x | axmodel vs ONNX |
|---|---|---|---|---|
| dit_p8 (control) | 8 | 0.473 | 0.481 | 0.641 |
| dit_p11 | 11 | 0.466 | **0.99996** | **0.149** |

`real-vs-zero-x` is the cosine between the output for the real `x` and the
output for an all-zero `x`.  On p11 the compiled model returns (almost) the
same output for both, i.e. `x` does not reach the graph.  p11's output std is
0.120 while ONNX is 2.272.

## Environment

- Pulsar2 8.0, `pulsar2 version` -> `commit: e9a1aa15`
- target: `AX650` (`target_hardware: AX650`, `npu_mode: NPU1`)
- board: AX650 (marketed AX8850), SDK `libax_engine.so` from 2024-09,
  runtime called through ctypes (`AX_ENGINE_CreateHandle` / `CreateContext` /
  `GetIOInfo` / `RunSync`)
- host: Ubuntu 24.04 x86_64

## Files

- `assets/dit_p11.onnx` + `.onnx.data`  (661 MB, 11 blocks)
- `assets/dit_p8.onnx` + `.onnx.data`   (493 MB, 8 blocks, control)
- `assets/dit_p11.axmodel`, `assets/dit_p8.axmodel` (built by us, for reference)
- `assets/input.npz`  (x, cond, mu, te; same tensors for both models)
- `assets/p11_ref.npy`, `assets/p8_ref.npy`  (ONNX outputs)
- `calib/{x,cond,mu,te}.tar.gz`  (4 real samples each, `0000.npy`..)
- `prepare.py`, `test_axmodel.py`, `ax_engine.py`

Download the large assets from the release:

```bash
gh release download v1 -R yuyun2000/pulsar2-dit-input-fold-repro -D assets
# or direct URLs:
# https://github.com/yuyun2000/pulsar2-dit-input-fold-repro/releases/download/v1/dit_p11.onnx
# https://github.com/yuyun2000/pulsar2-dit-input-fold-repro/releases/download/v1/dit_p11.onnx.data
# https://github.com/yuyun2000/pulsar2-dit-input-fold-repro/releases/download/v1/dit_p8.onnx
# https://github.com/yuyun2000/pulsar2-dit-input-fold-repro/releases/download/v1/dit_p8.onnx.data
```

## Reproduce

```bash
python3 prepare.py
pulsar2 build --config config_p11.json
pulsar2 build --config config_p8.json
```

On the board:

```bash
python3 test_axmodel.py --model assets/dit_p11.axmodel \
    --onnx-ref assets/p11_ref.npy --input assets/input.npz
python3 test_axmodel.py --model assets/dit_p8.axmodel \
    --onnx-ref assets/p8_ref.npy --input assets/input.npz
```

Expected:

```
dit_p11: axmodel vs ONNX ~0.15, real-vs-zero ~0.99996   (x ignored)
dit_p8 : axmodel vs ONNX ~0.64, real-vs-zero ~0.481     (x used)
```

## What we already tried (all give the same p11 output)

- INT8 MinMax / MSE / Percentile, real calibration
- U16 / S16 activations; LN U16, Softmax U16, MatMul+Softmax U16,
  Conv+MatMul+Softmax+LN all U16; `te` pathway U16
- `disable_onnx_optimization: true`
- baking `cond`/`mu`/`te` as constants, leaving only `x` as input (the output
  is still constant: real-vs-zero 0.999978)
- adding an identity tie `v_pred + x.transpose(1,2) * 1e-7` (folded away)

The issue is therefore not a quantization-width problem.  We are looking for
either a known limitation of the graph optimizer for deep transformer graphs,
or a config switch that disables the input folding.
