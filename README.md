# PagedServe

A PyTorch LLM inference engine with continuous batching and a paged KV cache.

PagedServe combines new prompts and active generation requests in each model iteration. The KV cache is stored in reusable fixed-size blocks, reducing wasted memory and allowing completed requests to release memory immediately.

## Supported models

| Model | Weights | Tokenizer |
| --- | --- | --- |
| Reference Transformer | Locally initialized | UTF-8 byte tokenizer |
| DistilGPT-2 | `distilbert/distilgpt2` | Hugging Face GPT-2 tokenizer |
| GPT-2 | `openai-community/gpt2` | Hugging Face GPT-2 tokenizer |


## Installation

PagedServe requires Python 3.12.

```bash
git clone https://github.com/jainalphin/pagedserve.git
cd pagedserve
./env.sh
source .venv/bin/activate
```

The first run of each pretrained model downloads and caches its weights from Hugging Face. CUDA is used automatically when available; otherwise, the model runs on CPU.

## Run the web interface

```bash
PYTHONPATH=. python -m streamlit run app.py
```

Select Reference Transformer, DistilGPT-2, or GPT-2 in the interface.

## Run from the command line

Reference Transformer:

```bash
PYTHONPATH=. python main.py
```

GPT-2:

```bash
PYTHONPATH=. python main.py --model gpt2
```

DistilGPT-2:

```bash
PYTHONPATH=. python main.py --model distilgpt2
```

## Scheduling strategies

Orca-style iteration scheduling is the default. A simplified Sarathi-style strategy is also available:

```bash
PYTHONPATH=. python main.py \
  --model distilgpt2 \
  --strategy sarathi \
  --prefill-chunk-size 128
```

The Sarathi strategy admits at most one prompt chunk per iteration and fills the remaining request slots with active decodes. Later chunks attend to earlier chunks through the paged KV cache, and the scheduler emits the first generated token only after the final prompt chunk. This follows the chunked-prefill and decode-maximal batching policy from [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal).

This is a single-device educational implementation, not the complete optimized Sarathi-Serve runtime. Chunking is intended to bound interruptions to active decodes when long prompts arrive; it can increase prompt TTFT and reduce closed-batch throughput, especially on CPU.

## Run in Notebook

Install the project in a notebook cell:

```python
!git clone https://github.com/jainalphin/pagedserve.git
%cd pagedserve
%pip install -r requirements.txt
```

Restart the notebook kernel after installation, then run GPT-2:

```python
from main import GPT2_MODEL, build_scheduler

scheduler = build_scheduler(GPT2_MODEL)
request_id = scheduler.add_request(
    "Once upon a time",
    max_new_tokens=30,
)

results = scheduler.run_until_complete()
print(results[request_id])
```

Multiple requests can be processed with continuous batching:

```python
first = scheduler.add_request("Artificial intelligence is", max_new_tokens=20)
second = scheduler.add_request("The future of computing", max_new_tokens=20)

results = scheduler.run_until_complete()
print(results[first])
print(results[second])
```

## Run benchmarks

Benchmark every supported model:

```bash
PYTHONPATH=. python benchmark.py
```

Benchmark one model with custom settings:

```bash
PYTHONPATH=. python benchmark.py --model gpt2 --batch-size 4 --max-new-tokens 32 --runs 5
```

Use FP32 for the baseline. FP16 is an explicit, separately reported optimization:

```bash
PYTHONPATH=. python benchmark.py \
  --model gpt2 \
  --dtype float16 \
  --max-batch-size 32 \
  --batch-size 32
```

Compare both schedulers, or measure prefill-induced decode stalls:

```bash
PYTHONPATH=. python benchmark.py \
  --model distilgpt2 \
  --strategy orca \
  --strategy sarathi \
  --prefill-chunk-size 128

PYTHONPATH=. python strategy_benchmark.py --model distilgpt2
```

The script reports model and system metadata, loading time, median and p95 generation latency, time to first token, token throughput, exact token counts, and peak CUDA memory. Pass `--json-output PATH` to retain every raw run. New entries in `SUPPORTED_MODELS` are benchmarked automatically.

See [BENCHMARKS.md](BENCHMARKS.md) for measured Apple M4 CPU results for all included models and the exact reproduction command.

## Compare Hugging Face, PagedServe, and vLLM

`comparison_benchmark.py` uses one common open-loop load generator for all three
engines. It creates identical deterministic token-ID prompts and arrival times,
requests an exact output-token count with greedy decoding, and records raw per-request
TTFT, TPOT, ITL, end-to-end latency, achieved RPS, output throughput, failures, GPU
activity, memory, and power. Each engine runs in a separate process.

- `hf`: sequential FCFS Hugging Face baseline without continuous batching.
- `pagedserve`: this repository with Orca or Sarathi scheduling.
- `vllm`: vLLM's asynchronous continuous-batching engine.

The quick profile tests 128-token and 900-token prompts. The extreme profile tests
16, 128, 512, and 900 input tokens at offered rates from 1 to 32 RPS plus an
all-at-once burst. GPT-2 has a 1,024-token context limit, so the 900-input/64-output
case is close to its architectural maximum without exceeding it.

vLLM recommends a compatible fresh environment because its wheel bundles compiled
CUDA/PyTorch components. On a new Kaggle GPU session, install it first and restart
the kernel afterward:

```python
%pip install -q uv
!uv pip install --system vllm --torch-backend=auto
```

Then clone this repository and run the correctness gate before benchmarking:

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider
```

Run vLLM alone first as a smoke test. This uses the same exact prompts, arrival
trace, decoding settings, metrics, and telemetry format as the other engines:

```bash
PYTHONPATH=. python comparison_benchmark.py \
  --engine vllm \
  --dtype float32 \
  --input-length 128 \
  --output-length 32 \
  --num-requests 30 \
  --request-rate 1 \
  --request-rate 4 \
  --request-rate 16 \
  --request-rate inf \
  --max-batch-size 64 \
  --gpu-memory-utilization 0.8 \
  --json-output /tmp/vllm_kaggle_smoke.json
```

The vLLM backend uses `AsyncLLM` rather than the synchronous convenience API, so
requests can arrive while earlier requests are decoding. It disables prefix
caching and uses greedy decoding with EOS ignored to generate exactly the requested
number of tokens. Do not add `--vllm-enforce-eager` for the measured run unless
CUDA graph initialization fails; eager and graph results are different tracks.

Start with the FP32 quick matrix:

```bash
PYTHONPATH=. python run_comparison_matrix.py \
  --profile quick \
  --dtype float32
```

After it completes successfully, run the longer FP32 extreme matrix. Run FP16 as a
separate track rather than combining its results with FP32:

```bash
PYTHONPATH=. python run_comparison_matrix.py \
  --profile extreme \
  --dtype float32

PYTHONPATH=. python run_comparison_matrix.py \
  --profile extreme \
  --dtype float16 \
  --output-dir benchmarks/comparison-fp16
```

The matrix writes temporary raw results and logs under `benchmarks/comparison/`,
which Git ignores. Copy only verified environment details and summary tables into
`BENCHMARKS.md`, the single tracked benchmark record. Do not call one rate
"maximum sustainable RPS" without a latency objective; pass `--ttft-slo-ms`,
`--tpot-slo-ms`, and/or `--e2e-slo-ms` to calculate SLO-qualified goodput.
