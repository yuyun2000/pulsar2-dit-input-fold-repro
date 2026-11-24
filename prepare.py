#!/usr/bin/env python3
"""Write pulsar2 configs with absolute paths for the current checkout."""
import json
import os

base = os.path.dirname(os.path.abspath(__file__))
shapes = "x:1x100x700,cond:1x100x700,mu:1x512x700,te:1x1024"
names = ["x", "cond", "mu", "te"]


def cfg(n):
    return {
        "input": f"{base}/assets/dit_p{n}.onnx",
        "output_dir": f"{base}/out",
        "output_name": f"dit_p{n}.axmodel",
        "work_dir": f"{base}/out/work_p{n}",
        "model_type": "ONNX",
        "target_hardware": "AX650",
        "npu_mode": "NPU1",
        "input_shapes": shapes,
        "onnx_opt": {"disable_onnx_optimization": False, "enable_onnxsim": False,
                     "model_check": True},
        "quant": {
            "input_configs": [{
                "tensor_name": k,
                "calibration_dataset": f"{base}/calib/{k}.tar.gz",
                "calibration_format": "Numpy",
                "calibration_size": 4,
                "calibration_mean": [],
                "calibration_std": [],
            } for k in names],
            "calibration_method": "MinMax",
            "precision_analysis": False,
            "highest_mix_precision": False,
        },
        "input_processors": [{"tensor_name": k, "tensor_layout": "NCHW",
                              "src_dtype": "FP32", "src_layout": "NCHW"} for k in names],
    }


U16_OPS = ["Add","Concat","Conv","Cos","Div","Erf","Gemm","Greater","LayerNormalization",
           "MatMul","Mul","Neg","Pow","ReduceL2","ReduceMean","Reshape","Sigmoid","Sin",
           "Slice","Softmax","Softplus","Split","Squeeze","Tanh","Transpose","Unsqueeze","Where"]

os.makedirs(f"{base}/out", exist_ok=True)
for n in (8, 11):
    c = cfg(n)
    with open(f"{base}/config_p{n}.json", "w") as f:
        json.dump(c, f, indent=2)
    c["output_name"] = f"dit_p{n}_u16.axmodel"
    c["work_dir"] = f"{base}/out/work_p{n}_u16"
    c["quant"]["layer_configs"] = [{"op_types": U16_OPS, "data_type": "U16",
                                    "weight_data_type": "S8", "output_data_type": "U16"}]
    with open(f"{base}/config_p{n}_u16.json", "w") as f:
        json.dump(c, f, indent=2)
    print("wrote config_p%d.json / config_p%d_u16.json" % (n, n))
