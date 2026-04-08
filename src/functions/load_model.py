"""
Modal Function to download and cache Qwen 2.5 7B model from HuggingFace.

This demonstrates Modal Volumes - persistent storage that survives across runs.
The model is downloaded once and cached in a Volume, so subsequent runs are instant.
"""

import modal

# Create a Modal app
app = modal.App("vertical-ai-load-model")

# Create a Volume to persist the model weights
# Volumes are Modal's distributed file system - perfect for large ML models
model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

# Define the container image with all ML dependencies
# This image is cached by Modal, so builds are fast after the first time
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.46.0",
        "torch==2.5.1",
        "accelerate==1.1.0",
        "huggingface_hub==0.26.0",
    )
)


@app.function(
    image=image,
    volumes={"/models": model_volume},  # Mount the volume at /models
    timeout=3600,  # Model download can take time on first run
    secrets=[modal.Secret.from_name("huggingface-secret")],  # Optional: for gated models
)
def download_model():
    """
    Download Qwen 2.5 7B Instruct model and tokenizer to the Volume.

    This function runs on Modal's infrastructure. The first run downloads
    ~14GB of model weights. Subsequent runs skip the download since the
    Volume persists the data.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os

    model_name = "Qwen/Qwen2.5-7B-Instruct"
    cache_dir = "/models/qwen-2.5-7b-instruct"

    print(f"🔍 Checking if model exists in Volume at {cache_dir}...")

    # Check if model is already cached
    if os.path.exists(cache_dir) and os.listdir(cache_dir):
        print(f"✅ Model already cached! Skipping download.")
        print(f"📁 Cached files: {len(os.listdir(cache_dir))} items")
        return {"status": "cached", "cache_dir": cache_dir, "model_name": model_name}

    print(f"📥 Downloading {model_name} from HuggingFace...")
    print("⏱️  This will take a few minutes on first run...")

    # Download tokenizer
    print("📝 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(cache_dir)

    # Download model weights
    # Using device_map="cpu" since we're just downloading, not running inference yet
    print("🧠 Downloading model weights (~14GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype="auto",
    )
    model.save_pretrained(cache_dir)

    # Commit changes to the Volume
    # This is crucial! Without commit(), changes are lost when the container exits
    print("💾 Committing model to Volume...")
    model_volume.commit()

    print(f"✅ Model successfully cached to Volume!")
    print(f"📊 Total size: {sum(os.path.getsize(os.path.join(cache_dir, f)) for f in os.listdir(cache_dir) if os.path.isfile(os.path.join(cache_dir, f))) / 1e9:.2f} GB")

    return {
        "status": "downloaded",
        "cache_dir": cache_dir,
        "model_name": model_name,
        "files_cached": len(os.listdir(cache_dir)),
    }


@app.function(
    image=image,
    gpu="T4",  # Use a GPU for faster inference
    volumes={"/models": model_volume},
    timeout=600,
)
def test_inference(prompt: str = "Hello! How can I help you today?"):
    """
    Test inference with the cached model to verify it works.

    This demonstrates loading a model from a Volume and running inference.
    The model loads quickly since it's already in the Volume.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    cache_dir = "/models/qwen-2.5-7b-instruct"

    print(f"📂 Loading model from Volume at {cache_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    print(f"🎯 Running test inference with prompt: '{prompt}'")

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(f"🤖 Model response: {response}")

    return {"prompt": prompt, "response": response}


@app.local_entrypoint()
def main():
    """
    Local entrypoint - run this with: modal run src/functions/load_model.py
    """
    print("🚀 Starting model download...")
    result = download_model.remote()
    print(f"\n📋 Result: {result}")

    print("\n🧪 Testing inference...")
    test_result = test_inference.remote("What's the weather like today?")
    print(f"\n📋 Inference test: {test_result}")
