import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
from flaxconfigmixtral import MixtralConfig
from runmixtral import initialize_model_with_numpy_weights 
from flax import nnx

def pcc(x, y):
    if hasattr(x, 'numpy'):
        x = x.numpy()  # PyTorch tensor
    if hasattr(y, 'numpy'):
        y = y.numpy()  # PyTorch tensor
    
    # Convert JAX arrays to numpy if needed
    if isinstance(x, jnp.ndarray):
        x = np.array(x)
    if isinstance(y, jnp.ndarray):
        y = np.array(y)
    
    # Flatten tensors
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate correlation
    correlation = np.corrcoef(x_flat, y_flat)[0, 1]
    
    return correlation

np_dir = "mixtral_numpy_weights"
os.makedirs(np_dir, exist_ok=True)
hf_token = None
with open('tokens.txt') as f:
    hf_token = f.read()

# Log in to Hugging Face Hub with your token
login(token=hf_token)  # Replace with your actual token


# Print status
print("Loading model - this may take several minutes...")
model_id = "mistralai/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 1
# Load the model with quantization

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    torch_dtype=torch.float16
)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

config_dict = config.to_dict()
with open(f"{np_dir}/config.json", "w") as f:
    import json
    json.dump(config_dict, f, indent=2)
print(f"Saved configuration to {np_dir}/config.json")

# 4. First, let's examine the model structure to verify the component names
print("Model structure verification:")
layer0 = model.model.layers[0]

# Print the keys of the first layer's state dict to understand the structure
layer0_keys = layer0.state_dict().keys()
print("\nLayer 0 component keys:")
for key in sorted(layer0_keys):
    print(f"  - {key}")

# Now let's correctly save the weights
print("\nSaving weights as NumPy files...")

# Save embeddings
embeddings = model.model.embed_tokens.weight.detach().cpu().numpy()
np.save(f"{np_dir}/embeddings.npy", embeddings)
print(f"Saved embeddings with shape {embeddings.shape}")

# Save LM head
lm_head = model.lm_head.weight.detach().cpu().numpy()
np.save(f"{np_dir}/lm_head.npy", lm_head)
print(f"Saved lm_head with shape {lm_head.shape}")

# Save attention components
attn_q = layer0.self_attn.q_proj.weight.detach().cpu().numpy()
attn_k = layer0.self_attn.k_proj.weight.detach().cpu().numpy()
attn_v = layer0.self_attn.v_proj.weight.detach().cpu().numpy()
attn_o = layer0.self_attn.o_proj.weight.detach().cpu().numpy()

np.save(f"{np_dir}/attn_q_proj.npy", attn_q)
np.save(f"{np_dir}/attn_k_proj.npy", attn_k)
np.save(f"{np_dir}/attn_v_proj.npy", attn_v)
np.save(f"{np_dir}/attn_o_proj.npy", attn_o)
print(f"Saved attention components")

# Save MoE experts correctly - inspect the structure first
moe = layer0.block_sparse_moe
num_experts = config.num_local_experts

# Get all experts' weights
for i in range(num_experts):
    # Correct structure for Mixtral's experts
    w1 = moe.experts[i].w1.weight.detach().cpu().numpy()
    w2 = moe.experts[i].w2.weight.detach().cpu().numpy()
    w3 = moe.experts[i].w3.weight.detach().cpu().numpy()
    
    np.save(f"{np_dir}/expert_{i}_w1.npy", w1)
    np.save(f"{np_dir}/expert_{i}_w2.npy", w2)
    np.save(f"{np_dir}/expert_{i}_w3.npy", w3)
    print(f"Saved expert {i} weights")

# Save router/gate weights
gate = moe.gate.weight.detach().cpu().numpy()
np.save(f"{np_dir}/moe_gate.npy", gate)
print(f"Saved MoE gate with shape {gate.shape}")

# Save layer norms
input_layernorm = layer0.input_layernorm.weight.detach().cpu().numpy()
post_attention_layernorm = layer0.post_attention_layernorm.weight.detach().cpu().numpy()
final_norm = model.model.norm.weight.detach().cpu().numpy()

np.save(f"{np_dir}/input_layernorm.npy", input_layernorm)
np.save(f"{np_dir}/post_attention_layernorm.npy", post_attention_layernorm)
np.save(f"{np_dir}/final_norm.npy", final_norm)
print(f"Saved all normalization layers")

# Create a metadata file
metadata = {
    "model": model_id,
    "version": "single-layer",
    "vocab_size": config.vocab_size,
    "hidden_size": config.hidden_size,
    "intermediate_size": config.intermediate_size,
    "num_attention_heads": config.num_attention_heads,
    "num_key_value_heads": config.num_key_value_heads,
    "num_experts": config.num_local_experts,
    "num_experts_per_tok": config.num_experts_per_tok,
    "files": [f for f in os.listdir(np_dir) if f.endswith('.npy')],
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "component_shapes": {
        "embeddings": embeddings.shape,
        "lm_head": lm_head.shape,
        "attn_q_proj": attn_q.shape,
        "attn_k_proj": attn_k.shape,
        "attn_v_proj": attn_v.shape,
        "attn_o_proj": attn_o.shape,
        "expert_w1": w1.shape,  # Same for all experts
        "expert_w2": w2.shape,
        "expert_w3": w3.shape,
        "moe_gate": gate.shape,
        "input_layernorm": input_layernorm.shape,
        "post_attention_layernorm": post_attention_layernorm.shape,
        "final_norm": final_norm.shape,
    }
}

with open(f"{np_dir}/metadata.json", "w") as f:
    import json
    json.dump(metadata, f, indent=2)

print(f"\nSuccessfully saved all weights as NumPy files to {np_dir}/")
print(f"Total parameter count: {metadata['parameter_count']:,}")

# Example prompt
prompt = "The capital city of USA is "
inputs = tokenizer(prompt, return_tensors="pt")
print("Model created successfully!")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
np.save("hf_input_ids.npy", input_ids.numpy())
np.save("hf_attention_mask.npy", attention_mask.numpy())

jax_weights = {}

# # Function to convert PyTorch tensor to JAX array
# for name, param in model.named_parameters():
#     # Convert to numpy array
#     param_numpy = param.detach().cpu().numpy()
    
#     # Create directory structure if needed
#     parts = name.split('.')
#     directory = os.path.join("mixtral_numpy_params", *parts[:-1])
#     os.makedirs(directory, exist_ok=True)
    
#     # Save as .npy file (efficient binary NumPy format)
#     filename = os.path.join("mixtral_numpy_params", *parts) + ".npy"
#     np.save(filename, param_numpy)
    
    # print(f"Saved {name} with shape {param_numpy.shape} to {filename}")

# with open("mixtral_numpy_params/manifest.txt", "w") as f:
#     for name, param in model.named_parameters():
#         f.write(f"{name},{','.join(str(dim) for dim in param.shape)}\n")

# print("All parameters saved!")
print("Generating respose...")

def compare_attention_with_manual_input():
    """Compare attention using manually created inputs"""
    print("Testing attention with manual inputs...")
    
    # --- Create manual inputs ---
    # Set dimensions based on Mixtral's configuration
    batch_size = 1
    seq_len = 10
    hidden_size = 4096  # Mixtral's hidden size
    
    # Create a simple random input tensor that will be identical for both models
    np.random.seed(42)  # Set seed for reproducibility
    manual_input_np = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
    
    # Create attention mask (all 1's for simplicity)
    attention_mask_np = np.ones((batch_size, seq_len), dtype=np.float32)
    
    print(f"Created manual input with shape: {manual_input_np.shape}")
    
    # --- Load Flax model ---
    print("Loading Flax model...")
    from runmixtral import initialize_model_with_numpy_weights
    from flaxconfigmixtral import MixtralConfig
    
    flax_config = MixtralConfig(num_hidden_layers=1)
    flax_model = initialize_model_with_numpy_weights()  # Your loading function

    # --- Convert inputs to appropriate formats ---
    # For PyTorch
    manual_input_torch = torch.tensor(manual_input_np, dtype=torch.float16).to(model.device)
    attention_mask_torch = torch.tensor(attention_mask_np).to(model.device)
    
    # For JAX
    manual_input_jax = jnp.array(manual_input_np)
    attention_mask_jax = jnp.array(attention_mask_np)

    # --- Compare Q, K, V projections ---
    print("\nComparing Q, K, V projections...")
    
    hf_layer = model.model.layers[0]
    flax_layer = flax_model.model.layers[0]
    position_ids = np.arange(seq_len)[None, :]  # shape: [1, 5]
    hf_position_ids = torch.tensor(position_ids).to(model.device)
    flax_position_ids = jnp.array(position_ids)

    # Get rope parameters from config
    rope_theta = getattr(config, "rope_theta", 1000000.0)
    head_dim = config.hidden_size // config.num_attention_heads
    
    # Create PyTorch version
    position_ids_float = hf_position_ids.float()
    half_dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half_dim, device=model.device).float() / half_dim))
    
    # Compute sin/cos for PyTorch
    freqs = torch.einsum("i,j->ij", position_ids_float.reshape(-1), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    hf_cos = torch.cos(emb).view(1, seq_len, head_dim).to(dtype=torch.float16)
    hf_sin = torch.sin(emb).view(1, seq_len, head_dim).to(dtype=torch.float16)

    # Position embeddings in the format expected by MixtralAttention
    hf_position_embeddings = (hf_cos, hf_sin)
    
    # Create JAX/Flax version
    pos_ids_float = flax_position_ids.astype(jnp.float32)
    jax_half_dim = head_dim // 2
    jax_inv_freq = 1.0 / (rope_theta ** (jnp.arange(0, jax_half_dim) / jax_half_dim))
    
    # Compute sin/cos for JAX
    jax_freqs = jnp.einsum("i,j->ij", pos_ids_float.reshape(-1), jax_inv_freq)
    jax_emb = jnp.concatenate([jax_freqs, jax_freqs], axis=-1)
    flax_cos = jnp.cos(jax_emb).reshape(1, seq_len, head_dim)
    flax_sin = jnp.sin(jax_emb).reshape(1, seq_len, head_dim)
    
    # Position embeddings in the format expected by Flax implementation
    flax_position_embeddings = (flax_cos, flax_sin)
    
    print("Successfully created position embeddings for both implementations")


    with torch.no_grad():
        # HF projections
        hf_q = hf_layer.self_attn.q_proj(manual_input_torch).cpu().numpy()
        hf_k = hf_layer.self_attn.k_proj(manual_input_torch).cpu().numpy()
        hf_v = hf_layer.self_attn.v_proj(manual_input_torch).cpu().numpy()
        hf_o = hf_layer.self_attn.o_proj(manual_input_torch).cpu().numpy()
    # Flax projections
    flax_q= np.array(flax_model.model.layers[0].attn.q_proj(manual_input_jax))
    flax_k = np.array(flax_model.model.layers[0].attn.k_proj(manual_input_jax))
    flax_v = np.array(flax_model.model.layers[0].attn.v_proj(manual_input_jax))
    flax_o = np.array(flax_model.model.layers[0].attn.o_proj(manual_input_jax))
    
    # Compare projections with PCC
    q_corr = pcc(hf_q, flax_q)
    k_corr = pcc(hf_k, flax_k)
    v_corr = pcc(hf_v, flax_v)
    o_corr = pcc(hf_o, flax_o)
    
    print(f"Q projection correlation: {q_corr}")
    print(f"K projection correlation: {k_corr}")
    print(f"V projection correlation: {v_corr}")
    print(f"O projection correlation: {o_corr}")
    
    print("\nProjection shapes:")
    print(f"HF Q: {hf_q.shape}, Flax Q: {flax_q.shape}")
    print(f"HF K: {hf_k.shape}, Flax K: {flax_k.shape}")
    print(f"HF V: {hf_v.shape}, Flax V: {flax_v.shape}")
    
    print("\nSample values (first token, first 5 values):")
    print(f"HF Q: {hf_q[0, 0, :5]}")
    print(f"Flax Q: {flax_q[0, 0, :5]}")
    
    # --- Compare complete attention output ---
    print("\nComparing full attention outputs...")
    try:
        # HF attention output
        attention_mask_torch = attention_mask_torch.view(1, 1, batch_size, seq_len)
        with torch.no_grad():
            hf_attn_output = hf_layer.self_attn(
                hidden_states=manual_input_torch,
                attention_mask=attention_mask_torch,
                position_embeddings = hf_position_embeddings,
                output_attentions=True
            )[1]
        # Flax attention output
        flax_attn_output = np.array(flax_layer.attn(
            hidden_states = manual_input_jax,
            attention_mask=attention_mask_jax,
            position_ids = flax_position_embeddings
        )[1])
        
        print(hf_attn_output.shape)
        print(flax_attn_output.shape)
        # Compare with PCC
        attn_corr = pcc(hf_attn_output, flax_attn_output)
        print(f"Full attention output correlation: {attn_corr}")
        
        # Calculate differences
        attn_diff = np.abs(hf_attn_output - flax_attn_output)
        max_diff = np.max(attn_diff)
        mean_diff = np.mean(attn_diff)
        
        print(f"Max absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        
        # Sample values
        print("\nSample attention output values (first token, first 5 values):")
        print(f"HF:   {hf_attn_output[0, 0, :5]}")
        print(f"Flax: {flax_attn_output[0, 0, :5]}")
        
    except Exception as e:
        print(f"Error running attention comparison: {e}")
        import traceback
        traceback.print_exc()
        
        # Try running with different signatures or without attention mask
        print("\nTrying alternative attention call patterns...")
    
    return {
        "q_corr": q_corr,
        "k_corr": k_corr,
        "v_corr": v_corr,
        "hf_q": hf_q,
        "flax_q": flax_q,
        "hf_k": hf_k,
        "flax_k": flax_k,
        "hf_v": hf_v,
        "flax_v": flax_v
    }
results = compare_attention_with_manual_input()

# print('Token ids:')
# print(output[0])
# # # Decode the output
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print("\nRESPONSE:")
# print(generated_text)
# np.save("output.npy", output[0].numpy())