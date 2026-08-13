from typing import Dict, Sequence, Tuple

import torch
from src.model.iteration import IterationItem
from src.model.kv_manager import KVCacheManager


class PagedAttention:
    def __init__(self, kv_manager: KVCacheManager):
        self.kv_manager = kv_manager
        self.num_layers = kv_manager.num_layers
        self.num_kv_heads = kv_manager.num_kv_heads
        self.head_dim = kv_manager.head_dim
        self.scale = self.head_dim ** -0.5

    def attention_score(self, request_id, layer_id, query):
        assert query.shape == (self.num_kv_heads, self.head_dim)
        assert 0 <= layer_id < self.num_layers

        # query:     [heads, head_dim]
        # After:     [heads, 1, head_dim]
        query = query.unsqueeze(1)

        request_info = self.kv_manager.requests[request_id]

        if layer_id not in request_info.written_layer_ids:
            raise RuntimeError(f"Layer {layer_id} has not written the reserved token's KV")

        physical_blocks, valid_token_size, context_length = self.kv_manager.get_context_metadata(request_id)

        attention_scores = []
        for physical_block_id, valid_tokens in zip(physical_blocks, valid_token_size):
            key_block = self.kv_manager.key_pool[layer_id, physical_block_id, :, :valid_tokens, :]
            # before: [heads, valid_tokens, head_dim]
            # after:  [heads, head_dim, valid_tokens]
            key_block = key_block.transpose(-2, -1)
            attention_score = torch.matmul(query, key_block)  # [heads, 1, valid_tokens]
            attention_score = attention_score.squeeze(1) * self.scale # [heads, valid_tokens]
            attention_scores.append(attention_score)

        attention_scores = torch.cat(attention_scores, dim=-1) # [num_heads, context_length]
        return attention_scores


    def compute_weighted_value_sum(self, request_id, layer_id, attention_scores):
        # online softmax may be?
        softmax_probabilities = torch.softmax(attention_scores, dim=-1)
        output = torch.zeros(
            self.num_kv_heads,
            self.head_dim,
            dtype=attention_scores.dtype,
            device=attention_scores.device,
        )

        physical_blocks, valid_token_size, context_length = self.kv_manager.get_context_metadata(request_id)

        start = 0
        for physical_block_id, valid_tokens in zip(physical_blocks, valid_token_size):

            value_block = self.kv_manager.value_pool[layer_id, physical_block_id, :, :valid_tokens, :] # [heads, valid_tokens, head_dim]
            block_prob = softmax_probabilities[:, start:start + valid_tokens] # [heads, valid_tokens]
            block_prob = block_prob.unsqueeze(1) # [heads, 1, valid_tokens]
            weighted_sum = torch.matmul(block_prob, value_block).squeeze(dim=1)
            output += weighted_sum
            start += valid_tokens

        return output

    def forward(self, request_id, layer_id, query):
        attention_score = self.attention_score(request_id, layer_id, query)
        return self.compute_weighted_value_sum(request_id, layer_id, attention_score)


    def forward_batch(self, request_ids, layer_id, queries):
        if not request_ids:
            return None

        if len(request_ids) != len(queries):
            raise RuntimeError("Number of request_ids and queries do not match")

        batch_size = len(request_ids)
        assert queries.shape == (batch_size, self.num_kv_heads, self.head_dim)

        outputs = []

        for batch_index, request_id in enumerate(request_ids):
            request_output = self.forward(request_id=request_id, layer_id=layer_id, query=queries[batch_index])
            outputs.append(request_output)

        return torch.stack(outputs, dim=0)

    def causal_prefill(self, queries, keys, values):
        token_count = queries.shape[0]
        queries = queries.transpose(0, 1)
        keys = keys.transpose(0, 1)
        values = values.transpose(0, 1)

        scores = torch.matmul(queries, keys.transpose(-1, -2)) * self.scale
        causal_mask = torch.triu(
            torch.ones(
                token_count,
                token_count,
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, values).transpose(0, 1)

    def forward_iteration(
        self,
        items: Sequence[IterationItem],
        layer_id: int,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[object, Tuple[torch.Tensor, torch.Tensor]]]:
        expected_shape = (queries.shape[0], self.num_kv_heads, self.head_dim)
        if queries.shape != expected_shape:
            raise ValueError("Unexpected flattened query shape")
        if keys.shape != expected_shape or values.shape != expected_shape:
            raise ValueError("Flattened Q/K/V tensors must have matching shapes")

        outputs = torch.empty_like(queries)
        prefill_kv = {}

        decode_items = [item for item in items if item.phase == "decode"]
        if decode_items:
            decode_request_ids = [item.request_id for item in decode_items]
            decode_keys = torch.stack([keys[item.start_offset] for item in decode_items])
            decode_values = torch.stack([values[item.start_offset] for item in decode_items])
            self.kv_manager.write_layer_kv_batch(
                decode_request_ids,
                layer_id,
                decode_keys,
                decode_values,
            )

        for item in items:
            item_slice = slice(item.start_offset, item.end_offset)
            item_queries = queries[item_slice]

            if item.phase == "prefill":
                item_keys = keys[item_slice]
                item_values = values[item_slice]
                outputs[item_slice] = self.causal_prefill(item_queries, item_keys, item_values)
                prefill_kv[item.request_id] = (item_keys.transpose(0, 1), item_values.transpose(0, 1))
            else:
                outputs[item_slice] = self.forward(request_id=item.request_id,
                                                   layer_id=layer_id,
                                                   query=item_queries[0],
                                                   ).unsqueeze(0)

        return outputs, prefill_kv
