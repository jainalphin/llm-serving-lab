# Benchmarks

## Apple M4 CPU — 17 August 2026

These are measurements from this repository, not estimates.

### Environment

- MacBook Pro (Mac16,1), Apple M4 (10 CPU cores), 16 GB memory
- macOS 15.7.7 (arm64)
- Python 3.12.13, PyTorch 2.13.0
- CPU execution with 4 PyTorch intra-op threads
- MPS was built into PyTorch but unavailable to this process, so these are **not GPU/MPS results**

### Workload

- Prompt: `Once upon a time`
- Batch size: 1
- Requested output: 16 tokens per request
- 3 warm-up runs followed by 20 measured runs per model
- Seed: 1234
- Every model produced all 320 expected measured tokens

The reference byte tokenizer produces 16 prompt tokens; the shared DistilGPT-2/GPT-2 tokenizer produces 4. TTFT therefore does not use the same prompt-token count. Timing starts immediately before the first scheduler step and ends after the request completes, so generation latency excludes model loading, tokenization, and queue submission. Throughput counts generated output tokens only.

### Results

| Model | Parameters | Observed load | Latency median | Latency p95 | TTFT median | TTFT p95 | Aggregate output throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference Transformer | 141,441 | 0.000980 s | 0.004060 s | 0.004390 s | 0.000440 s | 0.000503 s | 3,891.09 tokens/s |
| DistilGPT-2 | 81,912,576 | 0.883926 s | 0.118528 s | 0.145756 s | 0.017537 s | 0.027226 s | 131.46 tokens/s |
| GPT-2 | 124,439,808 | 5.661764 s | 0.177429 s | 0.195045 s | 0.027505 s | 0.032408 s | 89.14 tokens/s |

The load column is one observed in-process model construction/load, not a stable cold-start benchmark. CPU generation also varies with system conditions. An immediate independent repeat measured 3,791.09, 139.72, and 95.80 tokens/s for Reference, DistilGPT-2, and GPT-2 respectively; its median latencies were 0.004167, 0.113801, and 0.165170 seconds.

Do not interpret these rows as a model-quality comparison. The Reference Transformer is a tiny, randomly initialized test model, while DistilGPT-2 and GPT-2 are pretrained models with very different parameter counts.

### Reproduce

After both pretrained checkpoints have been downloaded once:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. \
  .venv/bin/python benchmark.py \
  --model gpt2 \
  --model distilgpt2 \
  --model reference \
  --warmup-runs 3 \
  --runs 20 \
  --batch-size 1 \
  --max-new-tokens 16 \
  --seed 1234 \
  --json-output /tmp/mac_m4_cpu.json
```

Raw JSON output is intentionally not tracked. This document is the repository's
single benchmark record; temporary JSON output is useful only for validating a run
before copying its verified summary here.

## Scheduler strategy comparison

Sarathi-style scheduling targets online decode stalls, not closed static batches. Both workload shapes therefore need to be measured.

### Closed batch

This workload submits four identical 75-token prompts together, requests 16 output tokens per prompt, and uses a 16-token Sarathi chunk. Results use 3 warm-ups and 10 measured runs.

| Strategy | Median batch latency | Median TTFT | Aggregate output throughput |
| --- | ---: | ---: | ---: |
| Orca | 0.407724 s | 0.058271 s | 154.20 tokens/s |
| Sarathi | 0.620729 s | 0.072626 s | 102.04 tokens/s |

On this CPU, chunking a closed batch adds overhead and is not beneficial.

### Long-prompt arrival during active decoding

This workload first establishes three active DistilGPT-2 decode requests, then introduces one 526-token prompt. Sarathi uses 128-token chunks. Results use 2 warm-ups and 10 measured runs.

| Strategy | New-prompt TTFT median/p95 | Maximum decode interruption median/p95 |
| --- | ---: | ---: |
| Orca | 108.628 / 115.252 ms | 108.626 / 115.250 ms |
| Sarathi | 180.891 / 273.680 ms | 43.766 / 80.299 ms |

Sarathi reduced the median prefill-induced decode interruption by 59.7%, while increasing the new prompt's median TTFT by 66.5%. An immediate independent repeat measured 105.034 ms versus 41.530 ms median decode interruption and 105.036 ms versus 171.695 ms prompt TTFT, confirming the tradeoff.

These CPU figures should not be extrapolated to GPU serving. Sarathi-Serve was designed around GPU compute utilization, and the optimal chunk size depends on the model, hardware, latency target, and arrival pattern.

## Common-engine Mac smoke comparison

The common comparison harness was also run on the same Apple M4 CPU with GPT-2,
FP32, deterministic token IDs, exact output lengths, and no quantization. Hugging
Face is a sequential FCFS baseline; PagedServe uses its paged KV cache and either
Orca or Sarathi scheduling. Native vLLM is not included because its CUDA engine
does not run on this Mac. These small samples validate the harness and expose CPU
behavior; they are not production capacity measurements or GPU predictions.

### Short requests

This workload uses 128 input tokens, 16 output tokens, and eight requests. The
finite-rate row uses a fixed-interval 4 RPS open-loop arrival trace; the burst row
submits all eight requests at once. Every engine completed every request, and both
PagedServe strategies produced exactly the same token IDs as Hugging Face.

| Offered load | Engine | Achieved RPS | Output tokens/s | TTFT median/p95 | TPOT median/p95 | E2E median/p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4 RPS | Hugging Face | 3.45 | 55.23 | 87.92 / 372.90 ms | 12.59 / 21.01 ms | 356.20 / 575.34 ms |
| 4 RPS | PagedServe Orca | 4.06 | 64.88 | 44.81 / 60.50 ms | 12.72 / 16.48 ms | 237.71 / 296.03 ms |
| 4 RPS | PagedServe Sarathi (64-token chunks) | 3.38 | 54.15 | 124.99 / 197.12 ms | 41.77 / 48.89 ms | 758.53 / 922.09 ms |
| Burst | Hugging Face | 2.31 | 37.03 | 1323.55 / 2562.80 ms | 16.29 / 41.39 ms | 1560.07 / 3142.61 ms |
| Burst | PagedServe Orca | 5.83 | 93.34 | 333.27 / 333.27 ms | 69.20 / 69.20 ms | 1371.25 / 1371.25 ms |
| Burst | PagedServe Sarathi (64-token chunks) | 4.90 | 78.36 | 370.85 / 724.46 ms | 54.41 / 64.07 ms | 1182.90 / 1618.45 ms |

At 4 offered RPS, Orca increased achieved throughput by 17.5% and reduced median
TTFT by 49.0% and median E2E latency by 33.3% relative to sequential Hugging Face.
Under the burst, batching raised throughput but also raised per-output-token time
on this CPU. Sarathi's chunking overhead was not beneficial for the finite-rate
CPU case.

### Near-maximum GPT-2 context

This burst workload uses 900 input tokens plus 64 output tokens (964 total), with
four requests. GPT-2's configured context limit is 1024 tokens. All engines again
completed every request with token-identical outputs.

| Engine | Achieved RPS | Output tokens/s | TTFT median/p95 | TPOT median/p95 | E2E median/p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hugging Face | 0.706 | 45.18 | 2552.48 / 4446.80 ms | 16.07 / 16.41 ms | 3565.40 / 5453.64 ms |
| PagedServe Orca | 0.527 | 33.71 | 1161.46 / 1161.46 ms | 102.11 / 102.11 ms | 7594.44 / 7594.44 ms |
| PagedServe Sarathi (128-token chunks) | 0.495 | 31.70 | 1647.05 / 2775.95 ms | 94.72 / 97.99 ms | 7614.51 / 8047.70 ms |

PagedServe delivered first tokens earlier under simultaneous arrivals, but its
batched paged decode was slower than sequential Hugging Face on the CPU, so TPOT,
E2E latency, and total throughput regressed. The intended high-concurrency test is
therefore the CUDA matrix, where batching can use GPU parallelism.
