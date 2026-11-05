"""Benchmark CPU vs MPS inference and training throughput for PP-OCRv6."""

from __future__ import annotations

import json
import time
import numpy as np
import torch

from ocrtorch.models.ppocrv6 import PP_OCRv6MediumRec
from ocrtorch.train.loss import CTCLossStep


def sync(device: torch.device):
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_inference(device_name: str, batch_sizes=[1, 8, 16]):
    device = torch.device(device_name)
    model = PP_OCRv6MediumRec().to(device)
    model.eval()

    num_steps = 10 if device_name == "mps" else 5
    warmup = 3 if device_name == "mps" else 2

    results = {}
    print(f"[{device_name.upper()}] Starting inference benchmark...", flush=True)
    for b in batch_sizes:
        x = torch.randn(b, 3, 48, 320, device=device)
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
                sync(device)

            times = []
            for _ in range(num_steps):
                t0 = time.perf_counter()
                _ = model(x)
                sync(device)
                times.append(time.perf_counter() - t0)

        avg_ms = float(np.mean(times) * 1000)
        ms_per_img = avg_ms / b
        img_sec = b / (avg_ms / 1000)
        results[b] = {
            "batch_latency_ms": round(avg_ms, 2),
            "latency_per_img_ms": round(ms_per_img, 2),
            "throughput_img_per_sec": round(img_sec, 2),
        }
        print(f"  Batch {b:2d}: {avg_ms:6.2f} ms/batch | {ms_per_img:6.2f} ms/img | {img_sec:6.2f} img/s", flush=True)
    return results


def benchmark_training(device_name: str, batch_sizes=[8, 16]):
    device = torch.device(device_name)
    model = PP_OCRv6MediumRec().to(device)
    model.train()
    criterion = CTCLossStep().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    num_steps = 10 if device_name == "mps" else 3
    warmup = 2 if device_name == "mps" else 1

    results = {}
    print(f"[{device_name.upper()}] Starting training step benchmark...", flush=True)
    for b in batch_sizes:
        x = torch.randn(b, 3, 48, 320, device=device)
        targets = torch.randint(1, 18700, (b, 15))
        input_lengths = torch.full((b,), 40, dtype=torch.long)
        target_lengths = torch.full((b,), 15, dtype=torch.long)

        for _ in range(warmup):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            sync(device)

        times = []
        for _ in range(num_steps):
            t0 = time.perf_counter()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            sync(device)
            times.append(time.perf_counter() - t0)

        avg_step_ms = float(np.mean(times) * 1000)
        throughput = b / (avg_step_ms / 1000)
        results[b] = {
            "step_latency_ms": round(avg_step_ms, 2),
            "throughput_img_per_sec": round(throughput, 2),
        }
        print(f"  Batch {b:2d}: {avg_step_ms:6.2f} ms/step | {throughput:6.2f} img/s", flush=True)
    return results


def main():
    mps_inf = benchmark_inference("mps")
    cpu_inf = benchmark_inference("cpu")

    mps_train = benchmark_training("mps")
    cpu_train = benchmark_training("cpu")

    summary = {
        "inference": {
            "mps": mps_inf,
            "cpu": cpu_inf,
        },
        "training": {
            "mps": mps_train,
            "cpu": cpu_train,
        },
    }
    with open("artifacts/benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nBenchmark saved to artifacts/benchmark_results.json")


if __name__ == "__main__":
    main()
