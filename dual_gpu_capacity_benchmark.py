"""Measure aggregate capacity from independent replicas on multiple GPUs.

Each GPU receives one model replica and an equal share of the offered request
rate. GPU memory is not combined: every worker owns its model and KV cache.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_rate(value):
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive")
    return parsed


def percentile(values, percent):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values):
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "p95": percentile(values, 95),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark data-parallel inference replicas on two T4 GPUs"
    )
    parser.add_argument("--engine", choices=("pagedserve", "vllm"), required=True)
    parser.add_argument("--pagedserve-strategy", choices=("orca", "sarathi"), default="orca")
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--input-length", type=positive_int, required=True)
    parser.add_argument("--output-length", type=positive_int, required=True)
    parser.add_argument("--num-requests-per-replica", type=positive_int, default=120)
    parser.add_argument("--request-rate", type=positive_rate, action="append", required=True)
    parser.add_argument("--gpu", action="append", help="physical GPU id (default: 0 and 1)")
    parser.add_argument("--max-batch-size", type=positive_int, default=64)
    parser.add_argument("--kv-cache-memory-mb", type=positive_int)
    parser.add_argument("--kv-cache-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-safety-mb", type=positive_int, default=3072)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/pagedserve-dual-gpu"),
    )
    return parser.parse_args()


def build_worker_command(args, worker_index, output_path):
    replica_rate_divisor = len(args.gpu)
    command = [
        sys.executable,
        str(Path(__file__).with_name("comparison_benchmark.py")),
        "--engine",
        args.engine,
        "--model-id",
        args.model_id,
        "--dtype",
        args.dtype,
        "--input-length",
        str(args.input_length),
        "--output-length",
        str(args.output_length),
        "--num-requests",
        str(args.num_requests_per_replica),
        "--max-batch-size",
        str(args.max_batch_size),
        "--seed",
        str(1234 + worker_index),
        "--json-output",
        str(output_path),
    ]
    for total_rate in args.request_rate:
        command.extend(("--request-rate", str(total_rate / replica_rate_divisor)))

    if args.engine == "pagedserve":
        command.extend(
            (
                "--pagedserve-strategy",
                args.pagedserve_strategy,
                "--kv-cache-memory-utilization",
                str(args.kv_cache_memory_utilization),
                "--kv-cache-safety-mb",
                str(args.kv_cache_safety_mb),
            )
        )
        if args.kv_cache_memory_mb is not None:
            command.extend(("--kv-cache-memory-mb", str(args.kv_cache_memory_mb)))
    else:
        command.extend(
            ("--gpu-memory-utilization", str(args.gpu_memory_utilization))
        )

    for option, value in (
        ("--ttft-slo-ms", args.ttft_slo_ms),
        ("--tpot-slo-ms", args.tpot_slo_ms),
        ("--e2e-slo-ms", args.e2e_slo_ms),
    ):
        if value is not None:
            command.extend((option, str(value)))
    return command


def combine_rate(worker_results, total_offered_rate):
    raw_requests = [
        request
        for result in worker_results
        for request in result["raw_requests"]
        if request["error"] is None
    ]
    ttfts = [request["ttft_seconds"] for request in raw_requests]
    tpots = [request["tpot_seconds"] for request in raw_requests]
    e2es = [request["e2e_seconds"] for request in raw_requests]
    goodputs = [result["goodput_requests_per_second"] for result in worker_results]
    return {
        "offered_request_rate": total_offered_rate,
        "achieved_request_throughput": sum(
            result["achieved_request_throughput"] for result in worker_results
        ),
        "output_token_throughput": sum(
            result["output_token_throughput"] for result in worker_results
        ),
        "goodput_requests_per_second": (
            sum(goodputs) if all(value is not None for value in goodputs) else None
        ),
        "successful_requests": sum(
            result["successful_requests"] for result in worker_results
        ),
        "failed_requests": sum(
            len(result["failed_requests"]) for result in worker_results
        ),
        "ttft_seconds": latency_summary(ttfts),
        "tpot_seconds": latency_summary(tpots),
        "e2e_seconds": latency_summary(e2es),
        "per_gpu_telemetry": [
            result["gpu_telemetry"] for result in worker_results
        ],
    }


def pair(value):
    if value is None:
        return "n/a"
    return f"{value['median'] * 1000:.2f}/{value['p95'] * 1000:.2f}"


def main():
    args = parse_args()
    args.gpu = args.gpu or ["0", "1"]
    if len(args.gpu) < 2 or len(args.gpu) != len(set(args.gpu)):
        raise ValueError("Provide at least two unique GPU ids")
    if not 0 < args.kv_cache_memory_utilization <= 1:
        raise ValueError("kv_cache_memory_utilization must be in (0, 1]")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    log_handles = []
    output_paths = []
    try:
        for worker_index, gpu_id in enumerate(args.gpu):
            output_path = args.output_dir / f"worker-{worker_index}.json"
            log_path = args.output_dir / f"worker-{worker_index}.log"
            command = build_worker_command(args, worker_index, output_path)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu_id
            log_handle = log_path.open("w")
            process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                cwd=Path(__file__).parent,
            )
            processes.append((gpu_id, process, log_path))
            log_handles.append(log_handle)
            output_paths.append(output_path)

        failed = []
        for gpu_id, process, log_path in processes:
            return_code = process.wait()
            if return_code != 0:
                failed.append((gpu_id, return_code, log_path))
    finally:
        for log_handle in log_handles:
            log_handle.close()

    if failed:
        for gpu_id, return_code, log_path in failed:
            print(f"GPU {gpu_id} worker failed with code {return_code}: {log_path}")
            print("\n".join(log_path.read_text().splitlines()[-40:]))
        raise SystemExit(1)

    reports = [json.loads(path.read_text()) for path in output_paths]
    for gpu_id, report in zip(args.gpu, reports):
        metadata = report["engine_metadata"]
        if args.engine == "pagedserve":
            print(
                f"GPU {gpu_id}: model={metadata['model_parameter_bytes'] / (1024 ** 2):.1f} MiB, "
                f"KV={metadata['kv_cache_memory_bytes'] / (1024 ** 2):.1f} MiB, "
                f"capacity={metadata['kv_cache_capacity_tokens']:,} tokens / "
                f"{metadata['kv_cache_capacity_max_length_requests']:,} max-length requests"
            )
        else:
            print(f"GPU {gpu_id}: vLLM {metadata['vllm_version']}")

    combined = [
        combine_rate(
            [report["results"][index] for report in reports],
            total_rate,
        )
        for index, total_rate in enumerate(args.request_rate)
    ]
    print(
        "total offered RPS | aggregate achieved RPS | SLO goodput RPS | output tok/s | "
        "TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | E2E p50/p95 (ms) | failures"
    )
    print("-" * 150)
    for result in combined:
        goodput = result["goodput_requests_per_second"]
        goodput_text = "n/a" if goodput is None else f"{goodput:.3f}"
        print(
            f"{result['offered_request_rate']:.1f} | "
            f"{result['achieved_request_throughput']:.3f} | "
            f"{goodput_text} | "
            f"{result['output_token_throughput']:.2f} | "
            f"{pair(result['ttft_seconds'])} | "
            f"{pair(result['tpot_seconds'])} | "
            f"{pair(result['e2e_seconds'])} | "
            f"{result['failed_requests']}"
        )
        for gpu_id, telemetry in zip(args.gpu, result["per_gpu_telemetry"]):
            if not telemetry or not telemetry.get("sample_count"):
                continue
            utilization = telemetry["gpu_kernel_active_percent"]
            memory = telemetry["gpu_memory_used_mb"]
            print(
                f"  GPU {gpu_id}: kernel mean/p95/max "
                f"{utilization['mean']:.1f}/{utilization['p95']:.1f}/"
                f"{utilization['maximum']:.1f}% | VRAM mean/max "
                f"{memory['mean']:.0f}/{memory['maximum']:.0f} MiB"
            )


if __name__ == "__main__":
    main()
