import os
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax import lax
import numpy as np
from functools import partial

from singlechip.flaxmixtral import FlaxMixtralForCausalLM as NotShardedModel
from multichip.multichipmixtral import FlaxMixtralForCausalLM as ShardedModel
from singlechip.flaxconfigmixtral import MixtralConfig
from jax_config import cpu_devices, axis_name, num_devices, device_mesh

config = MixtralConfig(num_hidden_layers=2)
prng_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)


def run_single_chip(input_data, attention_mask, max_len):
    print("Creating Regular model...")
    model = NotShardedModel(config)
    batch_size, seq_len = input_data.shape
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    for i in range(config.num_hidden_layers):
        model.model.layers[i].attn.cached_key = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        model.model.layers[i].attn.cached_value = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        model.model.layers[i].attn.cache_index = jnp.array(0, dtype = jnp.int32)

    out = model.generate(
        input_ids = input_data,
        attention_mask = extended_attention_mask,
        max_new_tokens = max_len - seq_len 
    )
    return out


@partial(jax.jit, static_argnames=['max_len'])
def run_multi_chip(input_data, attention_mask, max_len):
    # Same initial setup as before
    model = ShardedModel(config)
    batch_size, seq_len = input_data.shape
    
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
    out_spec = P("X")
    
    # Set up attention mask
    batch_size, seq_len = input_data.shape
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
    
    # Modified to handle both token ids and valid_length return values
    def model_fn(x, mask, cache, pos):
        tokens = model.generate(
            input_ids=x,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=pos,
            max_new_tokens=max_len - seq_len
        )
        return tokens
    
    unapplied_function = shard_map(
        model_fn,
        device_mesh,
        in_specs=(inputs_spec, P("X", None), cache_specs, P("X", None)), 
        out_specs=out_spec,  # Now we have two outputs with their specs
        check_rep=False,
    )
    
    # Call the function and get both outputs
    token_results = unapplied_function(
        sharded_input, sharded_mask, sharded_cache, sharded_position_ids
    )

    if pad_size:
        # Remove padding from both results
        original_batch_size = input_data.shape[0] - pad_size
        token_results = token_results[:original_batch_size]

    return token_results
  
if __name__ == '__main__':
    batch_size = 8
    seq_len = 10
    tokens = 5
    max_len = seq_len + tokens
    input_data = jax.random.randint(key = prng_key, shape = (batch_size, seq_len), minval = 0, maxval = config.vocab_size)
    attention_mask = jnp.ones_like(input_data)
    single_chip = run_single_chip(input_data, attention_mask, max_len)
    multi_chip = run_multi_chip(input_data, attention_mask, max_len)
    print(single_chip) #this runs
    print()
    print(multi_chip.shape) #this runs
    # print(multi_chip) #this doesnt
    if np.array_equal(np.array(single_chip), np.array(multi_chip)):
        print('Test successful!')
    else:
        print('Something doesnt work :(')