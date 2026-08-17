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

Select either **Reference Transformer** or **GPT-2 (pretrained)** in the interface.

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
