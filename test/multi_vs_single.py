import os
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax import lax

from singlechip.flaxmixtral import FlaxMixtralForCausalLM as NotShardedModel
from multichip.multichipmixtral import FlaxMixtralForCausalLM as ShardedModel
from singlechip.flaxconfigmixtral import MixtralConfig
from jax_config import cpu_devices, axis_name, num_devices, device_mesh

config = MixtralConfig(num_hidden_layers=2)
prng_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)


def run_single_chip(input_data, attention_mask, max_len):
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


def run_multi_chip(input_data, attention_mask, max_len):    
    print("Creating Mixtral MoE model...")
    model = ShardedModel(config)
    
    inputs_spec = P()  #repeated across devices
    replicated_input = jax.device_put(input_data, NamedSharding(device_mesh, inputs_spec))
    
    out_spec = P() #also repeated acorss devices

    batch_size, seq_len = input_data.shape
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    past_key_values = {}
    for i in range(config.num_hidden_layers):
        past_key_values[f'layer_{i}'] = {}
        past_key_values[f'layer_{i}']['cached_key'] = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        past_key_values[f'layer_{i}']['cached_value'] = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        past_key_values[f'layer_{i}']['cache_index'] = jnp.array(0, dtype = jnp.int32)
        
    print("Compiling model application...")
    unapplied_function = shard_map(
        lambda x: model.generate(
            input_ids=x,
            attention_mask=extended_attention_mask,
            past_key_values = past_key_values,
            max_new_tokens=max_len - seq_len
        ),
        device_mesh,
        in_specs=(inputs_spec,),
        out_specs=out_spec,
        check_rep=False,
    )
    
    # Run the model
    print("Running model...")
    results = unapplied_function(input_data)
    
    # Print results info
    print(f"Results shape: {results.shape}")
    print(f"Results dtype: {results.dtype}")
    
    return results
  
if __name__ == '__main__':
      batch_size = 8
      seq_len = 10
      tokens = 5
      max_len = seq_len + tokens
      input_data = jax.random.randint(key = prng_key, shape = (batch_size, seq_len), minval = 0, maxval = config.vocab_size)
      attention_mask = jnp.ones_like(input_data)
      print(input_data)
    #   run_single_chip(input_data, attention_mask, max_len)
      run_multi_chip(input_data, attention_mask, max_len)