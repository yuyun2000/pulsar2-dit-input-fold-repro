### Environment

- Pulsar2 8.0, `pulsar2 version` -> `commit: e9a1aa15`
- `target_hardware: AX650`, `npu_mode: NPU1`
- Board: AX650 (AX8850), SDK `libax_engine.so` 2024-09, runtime called through ctypes (`AX_ENGINE_CreateHandle/CreateContext/GetIOInfo/RunSync`)
- Host: Ubuntu 24.04 x86_64

### Problem

Compiling an 11-block DiT step graph produces an axmodel whose output is **independent of the main input `x`**. An 8-block control from the same graph is still input-dependent, so this appears between depth 8 and 11.

| model | blocks | ONNX real-vs-zero-x | axmodel real-vs-zero-x | axmodel vs ONNX |
|---|---|---|---|---|
| dit_p8 (control) | 8 | 0.473 | 0.481 | 0.641 |
| dit_p11 | 11 | 0.466 | **0.99996** | **0.149** |

`real-vs-zero-x` = cosine(output(real x), output(zero x)). For p11 the compiled model gives almost the same output for real and zero `x` (cos 0.99996), and its output std is 0.120 while ONNX is 2.272.

The full 22-block model behaves the same (real-vs-zero 0.9708, cosine vs ONNX 0.314). Baking `cond`/`mu`/`te` as constants and leaving only `x` as input gives real-vs-zero **0.999978**.

### Reproduction

Repo: https://github.com/yuyun2000/pulsar2-dit-input-fold-repro

```bash
git clone https://github.com/yuyun2000/pulsar2-dit-input-fold-repro
cd pulsar2-dit-input-fold-repro
gh release download v1 -R yuyun2000/pulsar2-dit-input-fold-repro -D assets
# direct: https://github.com/yuyun2000/pulsar2-dit-input-fold-repro/releases/tag/v1
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

Actual output:

```
dit_p11: axmodel vs ONNX 0.1485, real-vs-zero 0.99996   <-- x ignored
dit_p8 : axmodel vs ONNX 0.6410, real-vs-zero 0.4810    <-- x used
```

The release contains the ONNX models (+ external data), the calibration tars, the ONNX reference outputs, and the axmodels we built.

### What we tried (p11 output is identical in all cases)

- MinMax / MSE / Percentile calibration with real data
- U16/S16 activations; LN U16/S16, Softmax U16, MatMul+Softmax U16, Conv+MatMul+Softmax+LN all U16, `te` pathway U16
- `disable_onnx_optimization: true`
- baking `cond`/`mu`/`te` as constants (single input `x`): real-vs-zero 0.999978
- adding an identity tie `v_pred + x.transpose(1,2) * 1e-7` (folded away)

This is not a quantization-width issue. Is this a known graph-optimizer/DCE limitation for deep transformer graphs? Is there a config switch to disable the input folding? We can provide the full 22-block ONNX and the export scripts (GPT-SoVITS v5turbo CFM DiT, public checkpoint) if useful.
