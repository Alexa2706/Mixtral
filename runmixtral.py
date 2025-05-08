import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import os
from flaxconfigmixtral import MixtralConfig
from flaxmixtral import MixtralForCausalLM
from hf import 
config = MixtralConfig(num_hidden_layers=1)
# Assuming you want to test with tensor parallelism
# These settings can be adjusted based on your available devices
# Setting up global mesh for testing
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_platform_name', 'cpu')

def initialize_global_mesh(shape=(1, 8), axis_names=('tp',)):
    """Initialize the global mesh for tensor parallelism"""
    return jax.sharding.Mesh(jax.devices(), axis_names)

def test_mixtral_forward_pass():
    batch_size = 1
    seq_length = 5
    input_ids = jnp.ones((batch_size, seq_length), dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    
    print(f"Created test input with shape {input_ids.shape}")
    rngs = nnx.Rngs(0)
    # Initialize model using NNX
    print("Creating model...")
        # Create the model directly using its constructor
    model = MixtralForCausalLM(
        config=config,
        dtype=jnp.float32
    )
    
    print("Model created successfully!")
    
    # Run a forward pass
    print("Running forward pass...")
    input_ids = np.load('hf_input_ids.npy')  # Your tokenized prompt
    attention_mask = np.load('hf_attention_mask.npy')
    print(input_ids)
    print(attention_mask)
    def extract_mixtral_parameters(model):
        """Extract all parameter values from the Mixtral model."""
        params = {}
        
        # Embedding
        params["embed_tokens"] = model.model.embed_tokens.embedding.value
        
        # For each layer (in this case just 1)
        for i, layer in enumerate(model.model.layers):
            # Attention
            params[f"layers.{i}.attn.q_proj"] = layer.attn.q_proj.kernel.value
            params[f"layers.{i}.attn.k_proj"] = layer.attn.k_proj.kernel.value
            params[f"layers.{i}.attn.v_proj"] = layer.attn.v_proj.kernel.value
            params[f"layers.{i}.attn.o_proj"] = layer.attn.o_proj.kernel.value
            
            # MoE gate
            params[f"layers.{i}.block_sparse_moe.gate"] = layer.block_sparse_moe.gate.kernel.value
            
            # Experts
            for e in range(8):  # 8 experts
                expert = getattr(layer.block_sparse_moe, f"experts_{e}")
                params[f"layers.{i}.block_sparse_moe.experts.{e}.up_proj"] = expert.up_proj.kernel.value
                params[f"layers.{i}.block_sparse_moe.experts.{e}.gate_proj"] = expert.gate_proj.kernel.value
                params[f"layers.{i}.block_sparse_moe.experts.{e}.down_proj"] = expert.down_proj.kernel.value
        
        # LM head
        params["lm_head"] = model.lm_head.kernel.value
        
        return params

      # Usage
    #mixtral_params = extract_mixtral_parameters(model)

    # Print shapes for verification
    # for key, value in mixtral_params.items():
    #     print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        # outputs, cache = model(
        #       input_ids=input_ids,
        #       attention_mask=attention_mask,
        #       use_cache=True  # Enable caching
        # )
        # # Generate text
        # generated_ids = model.generate(
        #       input_ids=input_ids,
        #       attention_mask=attention_mask,
        #       max_new_tokens=19,        # Generate 20 new tokensens
        #       cache = cache,
        #       key=jax.random.PRNGKey(42)  # RNG key for sampling
        # )
        # print(generated_ids)

if __name__ == "__main__":
    success = test_mixtral_forward_pass()