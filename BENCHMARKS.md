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
  --json-output benchmarks/mac_m4_cpu.json
```

The committed [JSON result](benchmarks/mac_m4_cpu.json) contains system metadata, summary statistics, and all 60 raw measured runs.

## Scheduler strategy comparison

Sarathi-style scheduling targets online decode stalls, not closed static batches. Both workload shapes therefore need to be measured.

### Closed batch

This workload submits four identical 75-token prompts together, requests 16 output tokens per prompt, and uses a 16-token Sarathi chunk. Results use 3 warm-ups and 10 measured runs.

| Strategy | Median batch latency | Median TTFT | Aggregate output throughput |
| --- | ---: | ---: | ---: |
| Orca | 0.407724 s | 0.058271 s | 154.20 tokens/s |
| Sarathi | 0.620729 s | 0.072626 s | 102.04 tokens/s |

On this CPU, chunking a closed batch adds overhead and is not beneficial. The [raw closed-batch results](benchmarks/mac_m4_scheduler_strategies.json) contain all 20 measured runs.

### Long-prompt arrival during active decoding

This workload first establishes three active DistilGPT-2 decode requests, then introduces one 526-token prompt. Sarathi uses 128-token chunks. Results use 2 warm-ups and 10 measured runs.

| Strategy | New-prompt TTFT median/p95 | Maximum decode interruption median/p95 |
| --- | ---: | ---: |
| Orca | 108.628 / 115.252 ms | 108.626 / 115.250 ms |
| Sarathi | 180.891 / 273.680 ms | 43.766 / 80.299 ms |

Sarathi reduced the median prefill-induced decode interruption by 59.7%, while increasing the new prompt's median TTFT by 66.5%. An immediate independent repeat measured 105.034 ms versus 41.530 ms median decode interruption and 105.036 ms versus 171.695 ms prompt TTFT, confirming the tradeoff. The [raw stall results](benchmarks/mac_m4_scheduler_stall.json) include every iteration timing.

These CPU figures should not be extrapolated to GPU serving. Sarathi-Serve was designed around GPU compute utilization, and the optimal chunk size depends on the model, hardware, latency target, and arrival pattern.
