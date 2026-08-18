"""Common open-loop benchmark for HF, PagedServe, and vLLM.

Run one engine per process so model memory and CUDA state cannot leak between
comparisons. Every engine receives the same deterministic token IDs and arrival
trace. Hugging Face is intentionally a sequential FCFS baseline; PagedServe and
vLLM use their native continuous-batching schedulers.
"""

import argparse
import asyncio
import hashlib
import json
import math
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark import positive_integer
from main import GPT2_MODEL, SUPPORTED_DTYPES, build_scheduler


SUPPORTED_ENGINES = ("hf", "pagedserve", "vllm")


@dataclass
class RequestRecord:
    request_index: int
    scheduled_arrival: float
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    error: str | None = None


class NvidiaSMIMonitor:
    """Sample whole-device telemetry while only the measured workload runs."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,"
                "memory.used,memory.total,power.draw,power.limit",
                "--format=csv,noheader,nounits",
                f"--loop-ms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def stop(self):
        if self.process is None:
            return None
        self.process.terminate()
        try:
            output, error = self.process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            output, error = self.process.communicate()

        samples = []
        for line in output.splitlines():
            try:
                samples.append(
                    [float(value.strip()) for value in line.split(",")]
                )
            except ValueError:
                continue
        if not samples:
            return {"sample_count": 0, "error": error.strip() or None}

        gpu_utilization = [sample[0] for sample in samples]
        memory_activity = [sample[1] for sample in samples]
        memory_used_mb = [sample[2] for sample in samples]
        memory_total_mb = [sample[3] for sample in samples]
        power_draw_watts = [sample[4] for sample in samples]
        power_limit_watts = [sample[5] for sample in samples]
        return {
            "sample_count": len(samples),
            "sample_interval_ms": self.interval_ms,
            "gpu_kernel_active_percent": summarize(gpu_utilization),
            "memory_active_percent": summarize(memory_activity),
            "gpu_memory_used_mb": summarize(memory_used_mb),
            "gpu_memory_total_mb": max(memory_total_mb),
            "power_draw_watts": summarize(power_draw_watts),
            "power_limit_watts": max(power_limit_watts),
        }


class NullMonitor:
    def start(self):
        return None

    def stop(self):
        return None


def create_monitor(args):
    if torch.cuda.is_available():
        return NvidiaSMIMonitor(args.telemetry_interval_ms)
    return NullMonitor()


def percentile(values, percent):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize(values):
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def torch_dtype(dtype_name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def synchronize_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def arrival_offsets(request_rate, num_requests):
    if math.isinf(request_rate):
        return [0.0] * num_requests
    return [request_index / request_rate for request_index in range(num_requests)]


def deterministic_prompts(tokenizer, input_length, num_requests, seed):
    special_ids = set(getattr(tokenizer, "all_special_ids", ()))
    vocab_size = len(tokenizer)
    prompts = []
    for request_index in range(num_requests):
        generator = random.Random(seed + request_index)
        prompt = []
        while len(prompt) < input_length:
            token_id = generator.randrange(vocab_size)
            if token_id not in special_ids:
                prompt.append(token_id)
        prompts.append(prompt)
    return prompts


def request_metrics(record):
    if record.error or not record.token_times:
        return None
    ttft = record.token_times[0] - record.scheduled_arrival
    e2e = record.token_times[-1] - record.scheduled_arrival
    if len(record.token_times) > 1:
        intervals = [
            current - previous
            for previous, current in zip(record.token_times, record.token_times[1:])
        ]
        tpot = (record.token_times[-1] - record.token_times[0]) / len(intervals)
    else:
        intervals = []
        tpot = 0.0
    return {"ttft": ttft, "e2e": e2e, "tpot": tpot, "itls": intervals}


def summarize_scenario(
    engine,
    request_rate,
    records,
    duration,
    output_length,
    telemetry,
    ttft_slo_ms,
    tpot_slo_ms,
    e2e_slo_ms,
):
    completed = []
    failed = []
    for record in records:
        metrics = request_metrics(record)
        if (
            metrics is None
            or len(record.token_times) != output_length
            or len(record.token_ids) != output_length
        ):
            failed.append(
                {
                    "request_index": record.request_index,
                    "generated_tokens": len(record.token_times),
                    "error": record.error,
                }
            )
        else:
            completed.append(metrics)

    ttfts = [metrics["ttft"] for metrics in completed]
    e2es = [metrics["e2e"] for metrics in completed]
    tpots = [metrics["tpot"] for metrics in completed]
    itls = [interval for metrics in completed for interval in metrics["itls"]]

    def meets_slo(metrics):
        checks = []
        if ttft_slo_ms is not None:
            checks.append(metrics["ttft"] * 1000 <= ttft_slo_ms)
        if tpot_slo_ms is not None:
            checks.append(metrics["tpot"] * 1000 <= tpot_slo_ms)
        if e2e_slo_ms is not None:
            checks.append(metrics["e2e"] * 1000 <= e2e_slo_ms)
        return all(checks)

    good_requests = sum(meets_slo(metrics) for metrics in completed)
    generated_tokens = sum(len(record.token_times) for record in records)
    raw_requests = []
    for record in records:
        metrics = request_metrics(record)
        raw_requests.append(
            {
                "request_index": record.request_index,
                "scheduled_arrival_seconds": record.scheduled_arrival,
                "generated_tokens": len(record.token_times),
                "output_token_ids": record.token_ids,
                "ttft_seconds": metrics["ttft"] if metrics else None,
                "tpot_seconds": metrics["tpot"] if metrics else None,
                "e2e_seconds": metrics["e2e"] if metrics else None,
                "inter_token_seconds": metrics["itls"] if metrics else [],
                "error": record.error,
            }
        )

    return {
        "engine": engine,
        "offered_request_rate": "inf" if math.isinf(request_rate) else request_rate,
        "duration_seconds": duration,
        "successful_requests": len(completed),
        "failed_requests": failed,
        "achieved_request_throughput": len(completed) / duration,
        "goodput_requests_per_second": good_requests / duration,
        "output_token_throughput": generated_tokens / duration,
        "generated_tokens": generated_tokens,
        "ttft_seconds": summarize(ttfts),
        "tpot_seconds": summarize(tpots),
        "itl_seconds": summarize(itls),
        "e2e_seconds": summarize(e2es),
        "gpu_telemetry": telemetry,
        "raw_requests": raw_requests,
    }


def run_pagedserve_scenario(
    scheduler,
    prompts,
    output_length,
    request_rate,
    args,
):
    offsets = arrival_offsets(request_rate, len(prompts))
    records = [RequestRecord(index, offset) for index, offset in enumerate(offsets)]
    scheduler_ids = {}
    next_request = 0
    monitor = create_monitor(args)
    synchronize_cuda()
    monitor.start()
    benchmark_start = time.perf_counter()

    while next_request < len(prompts) or scheduler.waiting or scheduler.active:
        elapsed = time.perf_counter() - benchmark_start
        while next_request < len(prompts) and offsets[next_request] <= elapsed:
            scheduler_id = scheduler.add_token_request(
                prompts[next_request],
                max_new_tokens=output_length,
            )
            scheduler_ids[scheduler_id] = next_request
            next_request += 1

        if scheduler.waiting or scheduler.active:
            emitted = scheduler.step()
            synchronize_cuda()
            token_time = time.perf_counter() - benchmark_start
            for scheduler_id, token_id in emitted.items():
                record = records[scheduler_ids[scheduler_id]]
                record.token_times.append(token_time)
                record.token_ids.append(token_id)
        elif next_request < len(prompts):
            sleep_for = offsets[next_request] - (time.perf_counter() - benchmark_start)
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.001))

    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop()
    return records, duration, telemetry


def hf_generate_one(model, prompt, output_length, record, benchmark_start):
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=True)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        past_key_values = output.past_key_values
        synchronize_cuda()
        record.token_times.append(time.perf_counter() - benchmark_start)
        record.token_ids.append(int(next_token.item()))

        for _ in range(output_length - 1):
            output = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            past_key_values = output.past_key_values
            synchronize_cuda()
            record.token_times.append(time.perf_counter() - benchmark_start)
            record.token_ids.append(int(next_token.item()))


def run_hf_scenario(model, prompts, output_length, request_rate, args):
    offsets = arrival_offsets(request_rate, len(prompts))
    records = [RequestRecord(index, offset) for index, offset in enumerate(offsets)]
    monitor = create_monitor(args)
    synchronize_cuda()
    monitor.start()
    benchmark_start = time.perf_counter()

    for prompt, record in zip(prompts, records):
        wait_for = record.scheduled_arrival - (time.perf_counter() - benchmark_start)
        if wait_for > 0:
            time.sleep(wait_for)
        try:
            hf_generate_one(model, prompt, output_length, record, benchmark_start)
        except Exception as error:
            record.error = f"{type(error).__name__}: {error}"

    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop()
    return records, duration, telemetry


async def consume_vllm_request(
    engine,
    sampling_params,
    prompt,
    record,
    benchmark_start,
):
    wait_for = record.scheduled_arrival - (time.perf_counter() - benchmark_start)
    if wait_for > 0:
        await asyncio.sleep(wait_for)
    try:
        async for output in engine.generate(
            request_id=f"benchmark-{record.request_index}-{benchmark_start}",
            prompt={"prompt_token_ids": prompt},
            sampling_params=sampling_params,
        ):
            if not output.outputs:
                continue
            new_token_ids = output.outputs[0].token_ids
            if new_token_ids:
                token_time = time.perf_counter() - benchmark_start
                record.token_times.extend([token_time] * len(new_token_ids))
                record.token_ids.extend(int(token_id) for token_id in new_token_ids)
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"


async def run_vllm_scenario(
    engine,
    prompts,
    output_length,
    request_rate,
    args,
):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    offsets = arrival_offsets(request_rate, len(prompts))
    records = [RequestRecord(index, offset) for index, offset in enumerate(offsets)]
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=output_length,
        ignore_eos=True,
        output_kind=RequestOutputKind.DELTA,
    )
    monitor = create_monitor(args)
    synchronize_cuda()
    monitor.start()
    benchmark_start = time.perf_counter()
    await asyncio.gather(
        *[
            consume_vllm_request(
                engine,
                sampling_params,
                prompt,
                record,
                benchmark_start,
            )
            for prompt, record in zip(prompts, records)
        ]
    )
    synchronize_cuda()
    duration = time.perf_counter() - benchmark_start
    telemetry = monitor.stop()
    return records, duration, telemetry


def warmup_pagedserve(scheduler, prompt, output_length):
    request_id = scheduler.add_token_request(
        prompt,
        max_new_tokens=min(output_length, 8),
    )
    while request_id not in scheduler.finished:
        scheduler.step()
    synchronize_cuda()


def warmup_hf(model, prompt, output_length):
    record = RequestRecord(0, 0.0)
    hf_generate_one(model, prompt, min(output_length, 8), record, time.perf_counter())
    synchronize_cuda()


async def warmup_vllm(engine, prompt, output_length):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    params = SamplingParams(
        temperature=0.0,
        max_tokens=min(output_length, 8),
        ignore_eos=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
    )
    async for _ in engine.generate(
        request_id="benchmark-warmup",
        prompt={"prompt_token_ids": prompt},
        sampling_params=params,
    ):
        pass
    synchronize_cuda()


def system_metadata():
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
        git_dirty = None
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "device": "cuda",
                "cuda_device": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "gpu_memory_mb": properties.total_memory / (1024 * 1024),
            }
        )
    else:
        metadata.update(
            {
                "device": "cpu",
                "cuda_device": None,
                "compute_capability": None,
                "gpu_memory_mb": None,
            }
        )
    return metadata


def parse_request_rate(value):
    if value.lower() in ("inf", "infinity", "burst"):
        return math.inf
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("request rate must be positive or 'inf'")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare HF, PagedServe, and vLLM with one common load generator"
    )
    parser.add_argument("--engine", choices=SUPPORTED_ENGINES, required=True)
    parser.add_argument("--model-id", default="openai-community/gpt2")
    parser.add_argument("--dtype", choices=SUPPORTED_DTYPES, default="float32")
    parser.add_argument("--input-length", type=positive_integer, required=True)
    parser.add_argument("--output-length", type=positive_integer, required=True)
    parser.add_argument("--num-requests", type=positive_integer, default=100)
    parser.add_argument(
        "--request-rate",
        type=parse_request_rate,
        action="append",
        required=True,
        help="offered requests/second; repeat for a sweep or use inf for a burst",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-model-len", type=positive_integer, default=1024)
    parser.add_argument("--max-batch-size", type=positive_integer, default=64)
    parser.add_argument("--kv-cache-memory-mb", type=positive_integer, default=6144)
    parser.add_argument("--prefill-chunk-size", type=positive_integer, default=128)
    parser.add_argument(
        "--pagedserve-strategy",
        choices=("orca", "sarathi"),
        default="sarathi",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-enforce-eager", action="store_true")
    parser.add_argument("--telemetry-interval-ms", type=positive_integer, default=200)
    parser.add_argument("--ttft-slo-ms", type=float)
    parser.add_argument("--tpot-slo-ms", type=float)
    parser.add_argument("--e2e-slo-ms", type=float)
    parser.add_argument("--json-output", type=Path, required=True)
    return parser.parse_args()


def validate_args(args):
    if args.engine == "vllm" and not torch.cuda.is_available():
        raise RuntimeError("The vLLM comparison requires a supported GPU environment")
    if not torch.cuda.is_available() and args.dtype != "float32":
        raise ValueError("CPU comparison runs must use float32")
    if args.input_length + args.output_length > args.max_model_len:
        raise ValueError("input_length + output_length exceeds max_model_len")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")


def base_report(args):
    return {
        "system": system_metadata(),
        "settings": {
            "engine": args.engine,
            "model_id": args.model_id,
            "dtype": args.dtype,
            "input_length": args.input_length,
            "output_length": args.output_length,
            "num_requests": args.num_requests,
            "request_rates": [
                "inf" if math.isinf(rate) else rate for rate in args.request_rate
            ],
            "arrival_pattern": "fixed_interval_open_loop",
            "seed": args.seed,
            "max_model_len": args.max_model_len,
            "max_batch_size": args.max_batch_size,
            "kv_cache_memory_mb": args.kv_cache_memory_mb,
            "prefill_chunk_size": args.prefill_chunk_size,
            "pagedserve_strategy": args.pagedserve_strategy,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "vllm_enforce_eager": args.vllm_enforce_eager,
            "ttft_slo_ms": args.ttft_slo_ms,
            "tpot_slo_ms": args.tpot_slo_ms,
            "e2e_slo_ms": args.e2e_slo_ms,
        },
        "results": [],
    }


def add_summary(report, args, request_rate, records, duration, telemetry):
    report["results"].append(
        summarize_scenario(
            engine=args.engine,
            request_rate=request_rate,
            records=records,
            duration=duration,
            output_length=args.output_length,
            telemetry=telemetry,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            e2e_slo_ms=args.e2e_slo_ms,
        )
    )


def run_pagedserve(args, tokenizer, prompts, report):
    scheduler = build_scheduler(
        GPT2_MODEL,
        gpt2_model_id=args.model_id,
        scheduling_strategy=args.pagedserve_strategy,
        prefill_chunk_size=args.prefill_chunk_size,
        max_batch_size=args.max_batch_size,
        kv_cache_memory_mb=args.kv_cache_memory_mb,
        execution_dtype=args.dtype,
    )
    scheduler.eos_token_id = None
    warmup_pagedserve(scheduler, prompts[0], args.output_length)
    report["engine_metadata"] = {
        "policy": "continuous_batching",
        "model_dtype": str(next(scheduler.model_engine.parameters()).dtype),
        "kv_cache_blocks": scheduler.kv_manager.total_available_blocks,
    }
    for request_rate in args.request_rate:
        records, duration, telemetry = run_pagedserve_scenario(
            scheduler,
            prompts,
            args.output_length,
            request_rate,
            args,
        )
        add_summary(report, args, request_rate, records, duration, telemetry)


def run_hf(args, tokenizer, prompts, report):
    import transformers
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=torch_dtype(args.dtype),
    ).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    warmup_hf(model, prompts[0], args.output_length)
    report["engine_metadata"] = {
        "policy": "sequential_fcfs_no_continuous_batching",
        "transformers_version": transformers.__version__,
        "model_dtype": str(next(model.parameters()).dtype),
    }
    for request_rate in args.request_rate:
        records, duration, telemetry = run_hf_scenario(
            model,
            prompts,
            args.output_length,
            request_rate,
            args,
        )
        add_summary(report, args, request_rate, records, duration, telemetry)


async def run_vllm(args, prompts, report):
    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine_args = AsyncEngineArgs(
        model=args.model_id,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.vllm_enforce_eager,
        enable_prefix_caching=False,
        disable_log_stats=True,
        seed=args.seed,
    )
    engine = AsyncLLM.from_engine_args(engine_args)
    report["engine_metadata"] = {
        "policy": "vllm_async_continuous_batching",
        "vllm_version": vllm.__version__,
    }
    try:
        await warmup_vllm(engine, prompts[0], args.output_length)
        for request_rate in args.request_rate:
            records, duration, telemetry = await run_vllm_scenario(
                engine,
                prompts,
                args.output_length,
                request_rate,
                args,
            )
            add_summary(report, args, request_rate, records, duration, telemetry)
    finally:
        engine.shutdown()


def print_report(report):
    print(
        "engine | offered RPS | achieved RPS | goodput RPS | output tok/s | "
        "TTFT p50/p95 (ms) | TPOT p50/p95 (ms) | E2E p50/p95 (ms) | failures"
    )
    print("-" * 145)

    def latency_pair(summary):
        if summary is None:
            return "n/a"
        return f"{summary['median'] * 1000:.2f}/{summary['p95'] * 1000:.2f}"

    for result in report["results"]:
        ttft = result["ttft_seconds"]
        tpot = result["tpot_seconds"]
        e2e = result["e2e_seconds"]
        print(
            f"{result['engine']} | {result['offered_request_rate']} | "
            f"{result['achieved_request_throughput']:.3f} | "
            f"{result['goodput_requests_per_second']:.3f} | "
            f"{result['output_token_throughput']:.2f} | "
            f"{latency_pair(ttft)} | "
            f"{latency_pair(tpot)} | "
            f"{latency_pair(e2e)} | "
            f"{len(result['failed_requests'])}"
        )


def main():
    args = parse_args()
    validate_args(args)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompts = deterministic_prompts(
        tokenizer,
        args.input_length,
        args.num_requests,
        args.seed,
    )
    report = base_report(args)
    prompt_digest = hashlib.sha256(
        json.dumps(prompts, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report["model_metadata"] = {
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "vocab_size": len(tokenizer),
        "prompt_token_ids_sha256": prompt_digest,
    }
    if args.engine == "pagedserve":
        run_pagedserve(args, tokenizer, prompts, report)
    elif args.engine == "hf":
        run_hf(args, tokenizer, prompts, report)
    else:
        asyncio.run(run_vllm(args, prompts, report))

    print_report(report)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Raw results written to {args.json_output}")


if __name__ == "__main__":
    main()
