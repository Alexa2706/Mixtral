import torch
from transformers.activations import ACT2FN
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
from flax import nnx
from singlechip.flaxconfigmixtral import MixtralConfig
from jax.experimental.shard_map import shard_map
from jax_config import cpu_devices, axis_name, num_devices, device_mesh
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import lax

class MixtralBlockSparseTop2MLP(nnx.Module):
    """MLP module with sparse routing for Mixtral architecture."""

    def __init__(self, config: MixtralConfig, rngs : nnx.Rngs):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim
        self.up_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, rngs = rngs)
        self.gate_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, rngs = rngs)
        self.down_proj = nnx.Linear(inner_dim, embed_dim, use_bias=False, rngs = rngs)
        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states):
        gate_states = self.act_fn(self.up_proj(hidden_states)) * self.gate_proj(hidden_states)
        hidden_states = self.down_proj(gate_states)
        return hidden_states


class MixtralSparseMoeBlock(nnx.Module):
    """Sparse Mixture of Experts block for Mixtral with expert parallelism."""

    def __init__(self, config: MixtralConfig, dtype, rngs: nnx.Rngs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = dtype
        # Router (gate) is replicated across all devices
        self.gate = nnx.Linear(
            config.hidden_size,
            config.num_local_experts,
            use_bias=False,
            dtype=self.dtype,
            rngs=rngs
        )

        self.experts = []
        for i in range(self.num_experts):
            expert = MixtralBlockSparseTop2MLP(config, rngs=rngs)
            self.experts.append(expert)
            print(f"Created expert {i}")

        self.jitter_noise = config.router_jitter_noise


    def __call__(self, hidden_states):
        device_id = jax.lax.axis_index("X")
        batch_size, seq_len, hid_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hid_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = jax.nn.softmax(router_logits, axis = 1)
        routing_weights, selected_experts = lax.top_k(routing_weights, self.top_k)
        routing_weights /= jnp.sum(routing_weights, axis = -1, keepdims = True)

        routing_weights = routing_weights.astype(hidden_states.dtype)
        final_hidden_states = jnp.zeros_like(hidden_states)

        # This loop calculates the combined weights for tokens that are routed to the *current device's expert*.
        # It assumes expert `device_id` is the one "local" or "designated" for this device.
        # This implies self.num_experts should be equal to the number of devices in the "X" mesh.
        
        # Create a mask for tokens that select the current device's designated expert
        # and accumulate their weights.
        # token_mask_for_local_expert: (num_tokens_on_device), true if token is for local expert
        # expert_weights_for_local_expert: (num_tokens_on_device), sum of routing_weights if token is for local expert
        
        token_mask_for_local_expert = jnp.zeros(batch_size * seq_len, dtype=jnp.bool_)
        expert_weights_for_local_expert = jnp.zeros(batch_size * seq_len, dtype=routing_weights.dtype)

        for k in range(self.top_k):
            # is_selected_for_local_expert: (num_tokens_on_device), true if the k-th top expert is the local one
            is_selected_for_local_expert = (selected_experts[:, k] == device_id)
            
            token_mask_for_local_expert = token_mask_for_local_expert | is_selected_for_local_expert
            
            expert_weights_for_local_expert = jnp.where(
                is_selected_for_local_expert,
                expert_weights_for_local_expert + routing_weights[:, k],
                expert_weights_for_local_expert
            )
        
        # # The expert on the current device processes all hidden_states passed to it (local batch slice).
        # # jax.lax.switch is JIT-compatible for selecting the expert.
        # # self.experts should be a list/tuple of callable nnx.Modules.
        # # device_id (tracer) selects which expert callable to use.
        # # This assumes len(self.experts) is at least num_devices. If num_experts == num_devices, this is fine.
        current_expert_output = jax.lax.switch(
            device_id,
            self.experts, # Sequence of callable experts
            hidden_states # Operand for the selected expert
        )
    
    # # Apply mask and weights: only contribute if token was routed to this device's expert.
    # # weighted_expert_output has shape (num_tokens_on_device, hid_dim)
        weighted_expert_output = jnp.where(
            token_mask_for_local_expert[:, None], # Broadcast mask to (num_tokens, 1)
            current_expert_output * expert_weights_for_local_expert[:, None], # Broadcast weights
            jnp.zeros_like(current_expert_output)
        )
    
    # # Reshape back to (batch_slice_size, seq_len, hid_dim)
        weighted_expert_output_reshaped = weighted_expert_output.reshape(batch_size, seq_len, hid_dim)
    
    # # All-reduce across devices to sum expert outputs.
    # # Each device contributes the processed output for tokens routed to its expert.
    # # The sum over "X" combines these contributions for all tokens.
        final_output = jax.lax.psum(weighted_expert_output_reshaped, axis_name="X")
        return final_output

config = MixtralConfig(num_hidden_layers=1)
prng_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)
batch_size = 8
seq_len = 10
tokens = 5
hidden_size = 4096
max_len = seq_len + tokens
input_data = jax.random.normal(key = prng_key, shape = (batch_size, seq_len, hidden_size))
attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

def runner(input_data, attention_mask, max_len):
    print("Creating sharded model...")
    model = MixtralSparseMoeBlock(config, dtype = jnp.float16, rngs = rngs)
    print("Sharded Model created")
    batch_size, seq_len, hidden_size = input_data.shape
    
    # Handle padding - this is fine during compilation since it's based on shape
    inputs_spec = P("X")  
    pad_size = 0
    if batch_size % num_devices != 0:
        pad_size = num_devices - (batch_size % num_devices)
        padding = jnp.zeros((pad_size, seq_len), dtype=input_data.dtype)
        input_data = jnp.concatenate([input_data, padding], axis=0)
        
        mask_padding = jnp.zeros((pad_size, seq_len), dtype=attention_mask.dtype)
        attention_mask = jnp.concatenate([attention_mask, mask_padding], axis=0)
        
        batch_size += pad_size

    # Same sharding setup
    sharded_input = jax.device_put(input_data, NamedSharding(device_mesh, inputs_spec))
    out_spec = P()
    
    # Set up attention mask
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype="i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    sharded_mask = jax.device_put(extended_attention_mask, NamedSharding(device_mesh, P("X", None)))

    # KV Cache setup - same as before
    past_key_values = {}
    for i in range(config.num_hidden_layers):
        layer_key = f'layer_{i}'
        past_key_values[layer_key] = {
            'cached_key': jnp.zeros((batch_size, max_len, 8, 128), dtype=jnp.float32),
            'cached_value': jnp.zeros((batch_size, max_len, 8, 128), dtype=jnp.float32),
            'cache_index': jnp.array(0, dtype=jnp.int32)
        }
        
    sharded_cache = {}
    cache_specs = {}
    
    for layer_key, layer_cache in past_key_values.items():
        sharded_cache[layer_key] = {}
        sharded_cache[layer_key]['cached_key'] = jax.device_put(
            layer_cache['cached_key'],
            NamedSharding(device_mesh, P("X", None, None, None))
        )
        sharded_cache[layer_key]['cached_value'] = jax.device_put(
            layer_cache['cached_value'],
            NamedSharding(device_mesh, P("X", None, None, None))
        )
        sharded_cache[layer_key]['cache_index'] = jax.device_put(
            layer_cache['cache_index'],
            NamedSharding(device_mesh, P())
        )
        
    for i in range(config.num_hidden_layers):
        layer_key = f'layer_{i}'
        cache_specs[layer_key] = {
            'cached_key': P("X", None, None, None),
            'cached_value': P("X", None, None, None),
            'cache_index': P()
        }
        
    position_ids = jnp.cumsum(extended_attention_mask, axis=-1) - 1
    sharded_position_ids = jax.device_put(position_ids, NamedSharding(device_mesh, P("X", None)))
    
    def forward_pass(x):
        # Single forward pass through the model
        outputs = model(
            hidden_states=x,
            # attention_mask=mask,
            # past_key_values=cache,
            # position_ids=pos,
            # init_cache=True,
        )
        return outputs
    
    # Create sharded and JIT-compiled forward function
    sharded_forward = shard_map(
        forward_pass,
        device_mesh,
        in_specs=(inputs_spec), 
        out_specs=(P(None, None, None)),  # logits and updated cache
        check_rep=False,
    )
    
    # JIT compile the sharded forward pass
    jit_forward = jax.jit(sharded_forward)
    
    # Manual generation loop (outside JIT)
    # all_token_ids = jnp.zeros((batch_size, max_len), dtype=input_data.dtype)
    # all_token_ids = all_token_ids.at[:, :seq_len].set(sharded_input)
    
    # Process initial prompt
    logits = jit_forward(
        sharded_input, 
        # sharded_mask, 
        # sharded_cache, 
        # sharded_position_ids
    )
    return logits
    # Get next token
    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    # all_token_ids = all_token_ids.at[:, seq_len].set(next_token)
    
    # Continue generation for remaining tokens
    cur_len = seq_len + 1
    # for i in range(1, max_len - seq_len):
    #     # Prepare next input (only the last generated token)
    #     next_token_input = next_token[:, None]
    #     next_token_input = jax.device_put(
    #         next_token_input, 
    #         NamedSharding(device_mesh, P("X", None))
    #     )
        
    #     # Update position ids for the new token
    #     new_position_ids = sharded_position_ids[:, cur_len-1:cur_len]
        
    #     # Forward pass with JIT
    #     logits, sharded_cache = jit_forward(
    #         next_token_input,
    #         sharded_mask,
    #         sharded_cache,
    #         new_position_ids
    #     )
        
    #     # Get next token
    #     next_token_logits = logits[:, -1, :]
    #     next_token = jnp.argmax(next_token_logits, axis=-1)
    #     all_token_ids = all_token_ids.at[:, cur_len].set(next_token)
        
    #     cur_len += 1
    
    # Remove padding if necessary
    if pad_size:
        original_batch_size = input_data.shape[0] - pad_size
        all_token_ids = all_token_ids[:original_batch_size]
    
    return all_token_ids[:, :cur_len]

runner(input_data, attention_mask, max_len)