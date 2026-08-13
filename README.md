# LLM Serving Lab

This project is an educational laboratory for experimenting with LLM inference and serving techniques. It currently implements:

- Orca-style iteration-level scheduling
- Selective batching for mixed prefill and decode requests
- A paged key/value cache
- A simple decoder-only Transformer

It is designed to demonstrate inference-system behavior, not to generate useful text. The Transformer uses random weights.

## How one iteration works

1. A user submits a prompt through `ContinuousBatchScheduler.add_request()`.
2. The tokenizer converts the prompt into token IDs and places the request in the waiting queue.
3. `scheduler.step()` selects up to `max_batch_size` requests in FCFS order. The selection may contain both new prompts and active decode requests.
4. The scheduler flattens their current tokens into one tensor.

For example:

```text
Request A: decode [75]
Request B: prefill [10, 20, 30]

flat_input_ids    = [75, 10, 20, 30]
flat_position_ids = [5,   0,  1,  2]

A owns offsets [0:1]
B owns offsets [1:4]
```

5. The model processes the flattened tensor once.

```text
Embedding, LayerNorm, Q/K/V projections, MLP
                    ↓
          run over all flattened tokens
                    ↓
              Attention splits
              by request offsets
                    ↓
       outputs merge into one flat tensor
```

6. The scheduler selects the final-position logits for each request and produces one next token per request.
7. Finished requests release their KV-cache blocks. Other requests remain active for the next iteration.

## KV-cache behavior

The key and value pools use this layout:

```text
[layer, physical_block, KV_head, token_offset, head_dimension]
```

Each request owns a logical block table containing physical block IDs.

### Prefill

- Attention runs causally over the request's prompt slice.
- Every layer's prompt K/V is collected.
- After all layers finish, the K/V tensors are copied into paged blocks.

### Decode

- The KV manager reserves one token location.
- Every layer writes the current token's K/V into that location.
- PagedAttention reads the request's valid physical blocks.
- The token is committed only after every layer has written successfully.

## Create the environment

From the cloned project directory, run the setup script:

```bash
cd llm-serving-lab
./env.sh
source .venv/bin/activate
```

`env.sh` creates a local `.venv` and installs the packages from `requirements.txt`. When opening a new terminal later, only reactivate it:

```bash
cd llm-serving-lab
source .venv/bin/activate
```

## Run the web interface

Start the Streamlit interface:

```bash
PYTHONPATH=. python -m streamlit run app.py
```

The page accepts a prompt and generation length and displays generated text, token IDs, and latency.

## Run the tests

```bash
PYTHONPATH=. python -m pytest -q -p no:cacheprovider testing/test_project.py
```

## Current limitations

- Request-specific Attention uses Python loops as a correctness reference.
- There is no fused Triton or CUDA PagedAttention kernel.
- The included Transformer is not trained.
- The Streamlit demo serializes generation requests with a lock.
- This is a single-process, single-device demonstration.
