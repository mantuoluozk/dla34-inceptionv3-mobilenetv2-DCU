#!/usr/bin/env python3
"""DLA-34 optimization benchmark for Hygon DCU.

The original model.py and benchmark.py are intentionally left unchanged.
This script exposes reversible optimization switches for controlled A/B tests.
"""

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--batch-size", type=int, choices=(16, 128), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument(
        "--best",
        action="store_true",
        help="enable the best tested BW150 optimization combination",
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--miopen-find-mode", choices=("1", "3"), default="3")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
    )
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        parser.error("warmup cannot be negative and iterations must be positive")
    if args.tf32 and args.dtype != "fp32":
        parser.error("TF32 only applies to FP32; remove --tf32 for FP16")
    if args.best:
        args.channels_last = True
        args.tf32 = args.dtype == "fp32"
        args.miopen_find_mode = "3"
        args.compile_mode = "reduce-overhead"
        args.fullgraph = False
    return args


ARGS = parse_args()

# These variables must be set before importing torch/MIOpen.
os.environ["MIOPEN_FIND_MODE"] = ARGS.miopen_find_mode
if ARGS.channels_last:
    os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "1"
if ARGS.tf32:
    os.environ["MIOPEN_PRECISION_FP32_FP32_FP32_TF32_FP32"] = "1"
    os.environ["HIPBLASLT_ALLOW_TF32"] = "1"

import torch  # noqa: E402

from model import create_model  # noqa: E402


DTYPES = {"fp32": torch.float32, "fp16": torch.float16}
IMAGE_SIZE = 224


def synchronize(device):
    if device.type == "cpu":
        return
    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "synchronize"):
        try:
            backend.synchronize(device)
        except TypeError:
            backend.synchronize()


def percentile(values, ratio):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * ratio)]


def main():
    args = ARGS
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device or default_device)
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)

    model = create_model(pretrained=args.pretrained).eval()
    model = model.to(device=device, dtype=dtype)
    sample = torch.randn(
        args.batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=dtype,
    )

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        sample = sample.to(memory_format=torch.channels_last)

    parameter_count = sum(value.numel() for value in model.parameters())
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
    )

    if args.compile_mode != "none":
        model = torch.compile(
            model,
            backend="inductor",
            mode=args.compile_mode,
            fullgraph=args.fullgraph,
        )

    backend = getattr(torch, device.type, None)
    if backend is not None and hasattr(backend, "reset_peak_memory_stats"):
        backend.reset_peak_memory_stats(device)

    latencies_ms = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(sample)
        synchronize(device)

        for _ in range(args.iterations):
            start = time.perf_counter()
            output = model(sample)
            synchronize(device)
            latencies_ms.append((time.perf_counter() - start) * 1000)

    mean_ms = statistics.fmean(latencies_ms)
    result = {
        "model": "dla34",
        "variant": "optimized",
        "device": str(device),
        "device_name": (
            backend.get_device_name(device)
            if backend is not None and hasattr(backend, "get_device_name")
            else str(device)
        ),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "dtype": args.dtype,
        "precision_mode": "native",
        "batch_size": args.batch_size,
        "input_shape": list(sample.shape),
        "input_stride": list(sample.stride()),
        "output_shape": list(output.shape),
        "parameters": parameter_count,
        "model_state_size_mb": state_bytes / 1024**2,
        "optimizations": {
            "best_preset": args.best,
            "channels_last": args.channels_last,
            "channels_last_active": sample.is_contiguous(
                memory_format=torch.channels_last
            ),
            "tf32": args.tf32,
            "miopen_find_mode": args.miopen_find_mode,
            "compile_mode": args.compile_mode,
            "fullgraph": args.fullgraph,
        },
        "batch_latency_ms": {
            "mean": mean_ms,
            "min": min(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "p99": percentile(latencies_ms, 0.99),
            "max": max(latencies_ms),
        },
        "sample_latency_ms": mean_ms / args.batch_size,
        "throughput_images_per_second": args.batch_size * 1000 / mean_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "python_version": platform.python_version(),
    }
    if backend is not None and hasattr(backend, "max_memory_allocated"):
        result["peak_memory_mb"] = backend.max_memory_allocated(device) / 1024**2

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
