import os
from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax import lax

NUM_VIRTUAL_DEVICES = 8  # Adjustable
jax.config.update("jax_num_cpu_devices", NUM_VIRTUAL_DEVICES)


NUM_VIRTUAL_DEVICES = 8  
jax.config.update("jax_num_cpu_devices", NUM_VIRTUAL_DEVICES)
cpu_devices = jax.devices("cpu")
num_devices = len(cpu_devices)
print(f"Using {num_devices} virtual CPU devices: {cpu_devices}")
assert num_devices == 8, f"Need exactly 8 devices for 8 experts, but got {num_devices}"
axis_name = "X"
device_mesh = jax.make_mesh((num_devices,), (axis_name), devices=cpu_devices)
print(f"Created device mesh with axis '{axis_name}' of size {num_devices}")



def run_mixtral_on_virtual_cpus():
    print(f"JAX version: {jax.__version__}")
    rngs = nnx.Rngs(0)
    
    cpu_devices = jax.devices("cpu")
    num_devices = len(cpu_devices)
    print(f"Using {num_devices} virtual CPU devices: {cpu_devices}")
    
    # Ensure we have exactly 8 devices for 8 experts
    assert num_devices == 8, f"Need exactly 8 devices for 8 experts, but got {num_devices}"
    
    axis_name = "X"
    device_mesh = jax.make_mesh((num_devices,), (axis_name), devices=cpu_devices)
    print(f"Created device mesh with axis '{axis_name}' of size {num_devices}")
    
    # Generate example input
    batch_size = 4
    seq_len = 128
    hidden_dim = 1024
    
    prng_key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(prng_key)
    
    input_data = jnp.ones((batch_size, seq_len, hidden_dim))
    
    print("Creating Mixtral MoE model...")
    model = MixtralSparseMoeBlock(
        config=MixtralConfig(),
        dtype=jnp.float32,
        rngs=rngs,
        axis_name=axis_name
    )
    
    inputs_spec = P()  
    replicated_input = jax.device_put(input_data, NamedSharding(device_mesh, inputs_spec))
    
    out_spec = P()
    
    print("Compiling model application...")
    compiled_apply = jax.jit(
        lambda x: shard_map(
            lambda x: model(x)[0],  # Only return the hidden states, not router logits
            device_mesh,
            in_specs=(inputs_spec,),
            out_specs=out_spec,
            check_rep=False,
        )(x),
        out_shardings=NamedSharding(device_mesh, out_spec),
    )
    
    # Run the model
    print("Running model...")
    results = compiled_apply(replicated_input)
    
    # Print results info
    print(f"Results shape: {results.shape}")
    print(f"Results dtype: {results.dtype}")
    
    return results
  
if __name__ == '__main__':
      batch_size = 8
      seq_len = 10
      run_mixtral_on_virtual_cpus()