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

## NVIDIA Tesla T4 GPU — 18 August 2026

These measurements use one of the two T4s in a Kaggle instance. Restricting every
engine to the same GPU avoids giving vLLM a two-GPU advantage that PagedServe cannot
currently use.

### Environment and method

- Tesla T4, 15,360 MiB; Linux 6.12.90; PyTorch 2.13.0+cu132; CUDA 13.2
- GPT-2 (`openai-community/gpt2`), FP16 weights/activations/KV, no quantization
- 1,024-token model limit, greedy decoding, EOS ignored, prefix caching disabled
- 30 deterministic requests per offered rate: 1, 4, 16 RPS and a simultaneous burst
- Hugging Face: sequential FCFS baseline without continuous batching
- PagedServe: maximum batch 64, 6,144 MiB KV pool; Sarathi chunk size 128
- vLLM 0.27.1: asynchronous continuous batching, maximum 64 sequences, 80% GPU
  memory reservation, CUDA graphs enabled
- All engines completed every request with the requested output length and no failures

No latency SLO was supplied, so SLO-qualified goodput cannot be inferred. Achieved
RPS is measured over the finite trace including drain time; it can be below offered
RPS even when latency remains stable.

### 128 input + 32 output tokens

| Offered | Engine | Achieved RPS | Output tok/s | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | HF sequential | 1.025 | 32.80 | 11.68 / 13.91 ms | 8.11 / 8.40 ms | 263.55 / 273.88 ms |
| 1 | PagedServe Orca | 1.019 | 32.62 | 9.09 / 9.88 ms | 13.72 / 14.55 ms | 434.66 / 460.09 ms |
| 1 | PagedServe Sarathi | 1.020 | 32.63 | 9.15 / 9.74 ms | 13.57 / 14.04 ms | 429.77 / 444.72 ms |
| 1 | vLLM | 1.031 | 32.99 | 16.63 / 18.10 ms | 2.69 / 2.78 ms | 100.00 / 103.17 ms |
| 4 | HF sequential | 3.857 | 123.42 | 95.00 / 265.60 ms | 7.96 / 8.45 ms | 341.42 / 515.65 ms |
| 4 | PagedServe Orca | 3.885 | 124.31 | 25.60 / 31.31 ms | 13.64 / 15.46 ms | 448.80 / 508.51 ms |
| 4 | PagedServe Sarathi | 3.896 | 124.68 | 24.24 / 29.79 ms | 13.72 / 14.34 ms | 448.85 / 465.78 ms |
| 4 | vLLM | 4.082 | 130.62 | 16.29 / 21.23 ms | 2.61 / 2.77 ms | 96.92 / 105.17 ms |
| 16 | HF sequential | 3.816 | 122.10 | 2924.01 / 5484.08 ms | 8.09 / 8.45 ms | 3182.89 / 5734.54 ms |
| 16 | PagedServe Orca | 13.242 | 423.73 | 24.30 / 29.95 ms | 14.87 / 15.37 ms | 484.24 / 503.97 ms |
| 16 | PagedServe Sarathi | 13.220 | 423.03 | 20.78 / 30.78 ms | 14.94 / 15.55 ms | 487.03 / 502.47 ms |
| 16 | vLLM | 15.694 | 502.22 | 21.11 / 22.70 ms | 3.07 / 3.20 ms | 116.26 / 120.59 ms |
| Burst | HF sequential | 3.813 | 122.01 | 3847.63 / 7243.75 ms | 8.13 / 8.32 ms | 4102.16 / 7491.37 ms |
| Burst | PagedServe Orca | 41.296 | 1321.47 | 51.81 / 51.81 ms | 21.76 / 21.76 ms | 726.44 / 726.44 ms |
| Burst | PagedServe Sarathi | 26.960 | 862.73 | 262.84 / 554.20 ms | 20.41 / 20.99 ms | 906.72 / 1093.47 ms |
| Burst | vLLM | 111.598 | 3571.15 | 60.31 / 80.09 ms | 6.61 / 6.61 ms | 265.20 / 268.71 ms |

At 16 offered RPS, Orca produced 3.47 times the output throughput of sequential HF
and reduced median TTFT by 99.2%. vLLM produced 18.5% more output tokens/s than
Orca and reduced median E2E latency by 76.0%. Orca retained slightly lower burst
TTFT, while vLLM completed the burst 2.70 times faster by output throughput.

### 900 input + 64 output tokens

| Offered | Engine | Achieved RPS | Output tok/s | TTFT p50/p95 | TPOT p50/p95 | E2E p50/p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | HF sequential | 1.016 | 65.00 | 23.47 / 26.81 ms | 8.31 / 8.67 ms | 550.00 / 570.78 ms |
| 1 | PagedServe Orca | 1.001 | 64.07 | 17.28 / 18.75 ms | 15.11 / 15.54 ms | 969.42 / 999.78 ms |
| 1 | PagedServe Sarathi | 0.994 | 63.64 | 184.53 / 193.82 ms | 16.14 / 16.57 ms | 1201.16 / 1228.25 ms |
| 1 | vLLM | 1.027 | 65.75 | 37.45 / 38.33 ms | 2.64 / 2.81 ms | 204.68 / 213.73 ms |
| 4 | HF sequential | 1.823 | 116.68 | 4316.40 / 8237.12 ms | 8.36 / 8.60 ms | 4841.04 / 8763.96 ms |
| 4 | PagedServe Orca | 3.591 | 229.84 | 33.43 / 45.96 ms | 20.64 / 20.90 ms | 1337.42 / 1349.07 ms |
| 4 | PagedServe Sarathi | 3.416 | 218.63 | 279.52 / 316.15 ms | 29.83 / 31.76 ms | 2117.48 / 2284.27 ms |
| 4 | vLLM | 4.026 | 257.68 | 36.68 / 37.81 ms | 2.60 / 2.71 ms | 200.53 / 207.45 ms |
| 16 | HF sequential | 1.796 | 114.96 | 7214.92 / 13617.69 ms | 8.50 / 8.68 ms | 7740.68 / 14161.44 ms |
| 16 | PagedServe Orca | 5.931 | 379.59 | 63.08 / 116.20 ms | 55.52 / 58.51 ms | 3574.87 / 3747.88 ms |
| 16 | PagedServe Sarathi | 3.606 | 230.77 | 2500.17 / 5007.78 ms | 31.54 / 31.98 ms | 4514.19 / 6426.69 ms |
| 16 | vLLM | 10.695 | 684.46 | 66.08 / 108.14 ms | 22.13 / 25.85 ms | 1466.49 / 1685.45 ms |
| Burst | HF sequential | 1.811 | 115.90 | 8089.06 / 15244.29 ms | 8.43 / 8.77 ms | 8609.67 / 15771.77 ms |
| Burst | PagedServe Orca | 6.620 | 423.71 | 428.06 / 428.06 ms | 65.13 / 65.13 ms | 4531.42 / 4531.42 ms |
| Burst | PagedServe Sarathi | 3.594 | 230.00 | 3431.86 / 6756.16 ms | 31.50 / 31.98 ms | 5423.19 / 8171.24 ms |
| Burst | vLLM | 11.637 | 744.76 | 566.41 / 1041.01 ms | 31.09 / 36.35 ms | 2524.93 / 2575.94 ms |

At 4 offered RPS, sequential HF was already overloaded. Orca nearly doubled its
output throughput and reduced median TTFT by 99.2%. At 16 offered RPS, vLLM
delivered 80.3% more output tokens/s and 59.0% lower median E2E latency than Orca;
Orca's median TTFT remained 4.5% lower. Sarathi's chunking did not help this
homogeneous workload and was substantially slower for the long prompts.

### GPU activity and memory reservation

The monitor sampled only the active T4 every 200 ms. Burst runs produced only two
to six samples, so their mean utilization is not used for conclusions.

| Workload/load | Engine | GPU kernel mean/p95 | Memory activity mean/p95 | VRAM observed |
| --- | --- | ---: | ---: | ---: |
| 128/32 at 16 RPS | HF sequential | 37.8 / 40.0% | 16.9 / 18.0% | 433 MiB |
| 128/32 at 16 RPS | PagedServe Orca | 41.0 / 47.5% | 17.0 / 20.0% | 6,561 MiB |
| 128/32 at 16 RPS | PagedServe Sarathi | 38.4 / 47.5% | 16.2 / 20.0% | 6,561 MiB |
| 128/32 at 16 RPS | vLLM | 53.8 / 69.5% | 36.1 / 41.0% | 12,323 MiB |
| 900/64 at 16 RPS | HF sequential | 35.8 / 40.0% | 19.5 / 22.0% | 557 MiB |
| 900/64 at 16 RPS | PagedServe Orca | 71.5 / 83.0% | 60.3 / 72.8% | 7,496 MiB mean, 7,865 MiB max |
| 900/64 at 16 RPS | PagedServe Sarathi | 51.0 / 56.0% | 41.9 / 48.0% | 6,593 MiB |
| 900/64 at 16 RPS | vLLM | 94.9 / 100.0% | 26.1 / 32.7% | 12,323 MiB |

vLLM drove the T4 closer to compute saturation and achieved the strongest decode
performance, but it also preallocated 80% of GPU memory. PagedServe used a smaller
6 GiB KV pool, so VRAM figures are allocation policies rather than a like-for-like
measure of memory efficiency. The T4 cannot use FlashAttention 2; vLLM selected its
Triton attention backend. These are single-run, 30-request results and should be
repeated before treating small differences as stable.
