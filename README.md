# PagedServe

**A continuous-batching LLM inference engine built with PyTorch.**

PagedServe implements the core systems behind modern language-model serving: iteration-level scheduling, mixed prefill and decode batches, paged KV-cache management, and autoregressive decoding. Its modular runtime is designed to support different decoder-only model configurations while keeping scheduling, memory management, attention, and tokenization as independent components.

## Highlights

- **Continuous batching:** combines newly arrived prompts and active decode requests in the same model iteration.
- **Orca-style scheduling:** schedules work at iteration granularity instead of waiting for an entire static batch to finish.
- **Paged KV cache:** stores keys and values in fixed-size physical blocks and maps them to per-request logical block tables.
- **Memory-aware admission:** reserves cache capacity before admitting requests and releases blocks as soon as generation completes.
- **Unified prefill and decode path:** flattens mixed request phases into a single iteration batch while preserving request boundaries.
- **Configurable model runtime:** supports decoder depth, width, attention heads, context length, vocabulary, and device selection through `TransformerConfig`.
- **End-to-end interface:** includes a Streamlit application for submitting prompts and inspecting generated tokens and latency.
- **Correctness coverage:** tests scheduling, cache allocation, paged attention, prefill/decode parity, and request lifecycle behavior.

## Architecture

```text
Client requests
      │
      ▼
ContinuousBatchScheduler
  ├── waiting queue
  ├── active requests
  └── memory-aware admission
      │
      ▼
IterationBatch
  ├── mixed prefill/decode tokens
  ├── request offsets
  └── position IDs
      │
      ▼
PagedDecoderLM ───────► PagedAttention
      │                       │
      └───────────────────────▼
                       KVCacheManager
                    logical block tables
                    + physical K/V pools
```

The runtime separates request orchestration from model execution. The scheduler only depends on a model engine that exposes configuration and iteration-level inference, making the serving path extensible to additional decoder-only model implementations and tokenizers.

## How an iteration works

1. A request enters the scheduler through `ContinuousBatchScheduler.add_request()`.
2. The tokenizer converts its prompt into token IDs and places it in the waiting queue.
3. The scheduler selects requests in first-come, first-served order, subject to batch size and KV-cache capacity.
4. Prefill tokens from new requests and one decode token from each active request are flattened into a single `IterationBatch`.
5. The model processes the batch once, while attention uses request offsets to preserve independent causal contexts.
6. One next-token result is selected for every request in the iteration.
7. Completed requests immediately release their KV-cache blocks; unfinished requests remain active for the next step.

For example, a decode request and a new prompt can share one iteration:

```text
Request A: decode  [75]
Request B: prefill [10, 20, 30]

flat_input_ids    = [75, 10, 20, 30]
flat_position_ids = [ 5,  0,  1,  2]

A owns offsets [0:1]
B owns offsets [1:4]
```

## Paged KV-cache design

Key and value tensors are allocated from fixed-size pools with the layout:

```text
[layer, physical_block, KV_head, token_offset, head_dimension]
```

Each request maintains a logical block table containing its physical block IDs.

- During **prefill**, causal attention processes the prompt and writes every layer's K/V tensors into allocated cache blocks.
- During **decode**, the manager reserves one token location, each layer writes its K/V state, and the token is committed only after all layer writes succeed.
- When a request finishes, its physical blocks return to the free list for immediate reuse.

## Quick start

```bash
git clone <repository-url> pagedserve
cd pagedserve
./env.sh
source .venv/bin/activate
```

The setup script creates a Python 3.12 virtual environment and installs the dependencies from `requirements.txt`. PagedServe automatically uses CUDA when it is available and otherwise runs on CPU.

## Run the interface

```bash
PYTHONPATH=. python -m streamlit run app.py
```

The interface accepts a prompt and generation length, then displays the request ID, generated token IDs, output text, and end-to-end latency.

## Run the engine directly

```bash
PYTHONPATH=. python main.py
```

## Run the tests

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider testing/test_project.py
```

## Project structure

```text
pagedserve/
├── app.py                         # Streamlit interface
├── main.py                        # Runtime assembly and CLI example
├── src/
│   ├── model/
│   │   ├── iteration.py           # Mixed-phase iteration metadata
│   │   ├── kv_manager.py          # Paged KV-cache allocator
│   │   ├── paged_attention.py     # Cache-aware attention
│   │   ├── paged_decoder.py       # Configurable decoder-only model
│   │   └── tokenizer.py           # Tokenizer interface
│   └── scheduler/
│       └── orca_scheduler.py      # Continuous batch scheduler
└── testing/
    └── test_project.py            # Runtime and correctness tests
```

## Current scope

PagedServe currently ships with a compact, locally initialized decoder-only Transformer that exercises the complete serving path without requiring external model weights. The runtime focuses on inference-system correctness and component boundaries; generated text from the included model is not semantically trained output.

The next engineering milestones are pretrained model adapters, Hugging Face tokenizer integration, configurable sampling, optimized Triton/CUDA attention kernels, asynchronous request handling, and multi-device execution.
