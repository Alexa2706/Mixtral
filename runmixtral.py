import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import os
from flaxconfigmixtral import MixtralConfig
from flaxmixtral import MixtralForCausalLM

# Set up configuration with single layer


# Environment setup for tensor parallelism
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
jax.config.update('jax_platform_name', 'cpu')

def initialize_global_mesh(shape=(1, 8), axis_names=('tp',)):
    """Initialize the global mesh for tensor parallelism"""
    return jax.sharding.Mesh(jax.devices(), axis_names)

def load_numpy_weights(np_dir="mixtral_numpy_weights"):
    """Load all the NumPy weights from the directory"""
    weights = {}
    
    # Load embeddings
    weights["embeddings"] = np.load(f"{np_dir}/embeddings.npy")
    
    # Load attention weights
    weights["attn_q_proj"] = np.load(f"{np_dir}/attn_q_proj.npy")
    weights["attn_k_proj"] = np.load(f"{np_dir}/attn_k_proj.npy")
    weights["attn_v_proj"] = np.load(f"{np_dir}/attn_v_proj.npy")
    weights["attn_o_proj"] = np.load(f"{np_dir}/attn_o_proj.npy")
    
    # Load expert weights
    weights["experts"] = []
    for i in range(8):  # Mixtral has 8 experts
        expert = {
            "w1": np.load(f"{np_dir}/expert_{i}_w1.npy"),  # gate_proj in your model
            "w2": np.load(f"{np_dir}/expert_{i}_w2.npy"),  # up_proj in your model
            "w3": np.load(f"{np_dir}/expert_{i}_w3.npy"),  # down_proj in your model
        }
        weights["experts"].append(expert)
    
    # Load gate/router
    weights["moe_gate"] = np.load(f"{np_dir}/moe_gate.npy")
    
    # Load layer norms
    weights["input_layernorm"] = np.load(f"{np_dir}/input_layernorm.npy")
    weights["post_attention_layernorm"] = np.load(f"{np_dir}/post_attention_layernorm.npy")
    weights["final_norm"] = np.load(f"{np_dir}/final_norm.npy")
    
    # Load LM head
    weights["lm_head"] = np.load(f"{np_dir}/lm_head.npy")
    
    return weights

def initialize_model_with_numpy_weights():
    """Initialize the model and load the NumPy weights with shape awareness"""
    # Create model
    config = MixtralConfig(num_hidden_layers=1)
    print("Creating model...")
    model = MixtralForCausalLM(
        config=config,
        dtype=jnp.float32
    )
    
    # Load NumPy weights
    print("Loading weights...")
    np_weights = load_numpy_weights()
    
    # Helper function to match shapes
    def match_shape(model_param, np_param):
        """Check if shapes match directly or when transposed, return appropriately shaped array"""
        model_shape = model_param.shape
        np_shape = np_param.shape
        
        print(f"Comparing shapes: model={model_shape}, numpy={np_shape}")
        
        if model_shape == np_shape:
            return jnp.array(np_param)  # Shapes match directly
        elif model_shape == np_shape[::-1]:
            print(f"  Transposing from {np_shape} to {model_shape}")
            return jnp.array(np_param.T)  # Shapes match when transposed
        else:
            # Shapes don't match even with transposition
            print(f"  WARNING: Shapes incompatible: model={model_shape}, numpy={np_shape}")
            # Return original shape, but log warning
            return jnp.array(np_param)
    
    # Update model parameters with loaded weights
    print("Updating model parameters with loaded weights...")
    
    # Embeddings - these typically don't need transposition
    model.model.embed_tokens.embedding.value = jnp.array(np_weights["embeddings"])
    
    # LM head - may need transposition
    model.lm_head.kernel.value = match_shape(
        model.lm_head.kernel.value, 
        np_weights["lm_head"]
    )
    
    # Layer 0 parameters (since we're using a 1-layer model)
    layer = model.model.layers[0]
    
    # Attention - these typically need transposition
    layer.attn.q_proj.kernel.value = jnp.array(np_weights["attn_q_proj"].T)
    layer.attn.k_proj.kernel.value = match_shape(
        layer.attn.k_proj.kernel.value, 
        np_weights["attn_k_proj"]
    )
    layer.attn.v_proj.kernel.value = match_shape(
        layer.attn.v_proj.kernel.value, 
        np_weights["attn_v_proj"]
    )
    layer.attn.o_proj.kernel.value = jnp.array(np_weights["attn_o_proj"].T)
    
    # MoE gate - may need transposition
    layer.block_sparse_moe.gate.kernel.value = match_shape(
        layer.block_sparse_moe.gate.kernel.value, 
        np_weights["moe_gate"]
    )
    
    # Experts - these typically need transposition
    for e in range(8):
        expert = getattr(layer.block_sparse_moe, f"experts_{e}")
        # Map w1->gate_proj, w2->up_proj, w3->down_proj with shape matching
        expert.gate_proj.kernel.value = match_shape(
            expert.gate_proj.kernel.value, 
            np_weights["experts"][e]["w1"]
        )
        expert.up_proj.kernel.value = match_shape(
            expert.up_proj.kernel.value, 
            np_weights["experts"][e]["w2"]
        )
        expert.down_proj.kernel.value = match_shape(
            expert.down_proj.kernel.value, 
            np_weights["experts"][e]["w3"]
        )
    
    print("Model parameters updated successfully!")
    return model

def test_mixtral_forward_pass():
    # Load input data
    input_ids = jnp.array(np.load('hf_input_ids.npy'))
    attention_mask = jnp.array(np.load('hf_attention_mask.npy'))
    
    print(f"Created test input with shape {input_ids.shape}")
    
    # Initialize the model with NumPy weights
    model = initialize_model_with_numpy_weights()
    # model = MixtralForCausalLM(config)
    print("Model initialized with NumPy weights successfully!")
    
    # Run forward pass
    print("Running forward pass...")
    rngs = nnx.Rngs(0)
    print(input_ids.shape, attention_mask.shape)
    outputs, cache = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True
    )
    
    
    # Optional: Generate text
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=5,
        cache=cache,
        key=rngs
    )
    
    print(f"Generated IDs shape: {generated_ids.shape}")
    print(generated_ids)
    return True
"""
Should get:
 1,   415,  5565,  2990,   302,  7035,   349, 28705,  3685,  3685,
        27096, 25931, 30990
Got:
1   415  5565  2990   302  7035   349 28705   520   520 21827  5455
  10012
"""
if __name__ == "__main__":
    success = test_mixtral_forward_pass()
    if success:
        print("Test completed successfully!")