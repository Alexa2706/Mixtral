import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os

with open('tokens.txt') as f:
    hf_token = f.read()

# Log in to Hugging Face Hub with your token
login(token=hf_token)  # Replace with your actual token


# Print status
print("Loading model - this may take several minutes...")

# Load the model with quantization
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",  # Will use CPU
#     quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token # Set pad token to eos token
# Example prompt
prompt = "The capital city of USA is "
inputs = tokenizer(prompt, return_tensors="pt")
print("Model created successfully!")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
np.save("hf_input_ids.npy", input_ids.numpy())
np.save("hf_attention_mask.npy", attention_mask.numpy())

jax_weights = {}

# Function to convert PyTorch tensor to JAX array
for name, param in model.named_parameters():
    # Convert to numpy array
    param_numpy = param.detach().cpu().numpy()
    
    # Create directory structure if needed
    parts = name.split('.')
    directory = os.path.join("mixtral_numpy_params", *parts[:-1])
    os.makedirs(directory, exist_ok=True)
    
    # Save as .npy file (efficient binary NumPy format)
    filename = os.path.join("mixtral_numpy_params", *parts) + ".npy"
    np.save(filename, param_numpy)
    
    # print(f"Saved {name} with shape {param_numpy.shape} to {filename}")

with open("mixtral_numpy_params/manifest.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name},{','.join(str(dim) for dim in param.shape)}\n")

print("All parameters saved!")
print("Generating respose...")
# Generate output (this will be slow on CPU)
with torch.no_grad():
    output = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens = 5,
    )

print('Token ids:')
print(output[0])
# # Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nRESPONSE:")
print(generated_text)
np.save("output.npy", output[0].numpy())