import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import numpy as np


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
    device_map="auto",  # Will use CPU
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

print("Model loaded successfully! Generating response...")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
np.save("hf_input_ids.npy", input_ids.numpy())
np.save("hf_attention_mask.npy", attention_mask.numpy())
# Generate output (this will be slow on CPU)
with torch.no_grad():
    output = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens = 5,
    )

# Decode the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\nRESPONSE:")
print(generated_text)
