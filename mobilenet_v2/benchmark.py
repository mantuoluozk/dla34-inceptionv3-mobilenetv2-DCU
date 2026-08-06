#!/usr/bin/env python3
"""Benchmark mobilenet_v2 with random input."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from model import create_model


MODEL_NAME = "mobilenet_v2"
IMAGE_SIZE = 224
DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


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
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("batch-size/iterations must be positive and warmup cannot be negative")

    device = torch.device(args.device)
    compute_dtype = DTYPES[args.dtype]
    model = (
        create_model(pretrained=args.pretrained)
        .eval()
        .to(device=device, dtype=compute_dtype)
    )
    sample = torch.randn(
        args.batch_size,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        device=device,
        dtype=compute_dtype,
    )
    parameter_count = sum(value.numel() for value in model.parameters())
    trainable_count = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
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
        "model": MODEL_NAME,
        "pretrained": args.pretrained,
        "device": str(device),
        "device_name": (
            backend.get_device_name(device)
            if backend is not None and hasattr(backend, "get_device_name")
            else str(device)
        ),
        "hip_version": torch.version.hip,
        "dtype": args.dtype,
        "precision_mode": "native",
        "batch_size": args.batch_size,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "parameters": parameter_count,
        "trainable_parameters": trainable_count,
        "model_state_size_mb": state_bytes / 1024**2,
        "latency_ms": {
            "mean": mean_ms,
            "min": min(latencies_ms),
            "p50": percentile(latencies_ms, 0.50),
            "p90": percentile(latencies_ms, 0.90),
            "p99": percentile(latencies_ms, 0.99),
            "max": max(latencies_ms),
        },
        "throughput_images_per_second": args.batch_size * 1000 / mean_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    if backend is not None and hasattr(backend, "max_memory_allocated"):
        result["peak_memory_mb"] = backend.max_memory_allocated(device) / 1024**2

    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
