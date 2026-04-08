"""
Modal Function for serving the fine-tuned model and comparing with base model.

This is the PAYOFF MOMENT of the demo - we show the before/after comparison:
- Base model (generic) vs Fine-tuned model (specialized for customer service)

This demonstrates Modal Functions for inference serving:
- Fast cold starts (model loaded from Volume)
- GPU-accelerated inference
- Web endpoint for production use
- Easy A/B testing between model versions
"""

import modal

app = modal.App("vertical-ai-serve")

model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.46.0",
        "torch==2.5.1",
        "accelerate==1.1.0",
        "fastapi==0.115.0",
        "pydantic==2.9.0",
    )
)


@app.function(
    image=image,
    gpu="T4",  # T4 is perfect for inference (cheaper than A100, still fast)
    volumes={"/models": model_volume},
    timeout=600,
    keep_warm=1,  # Keep 1 instance warm for fast responses
)
def generate_base(prompt: str, max_tokens: int = 150) -> str:
    """
    Generate response using the BASE (non-fine-tuned) model.

    This is the "before" in our before/after comparison.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    cache_dir = "/models/qwen-2.5-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    return response.strip()


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=600,
    keep_warm=1,
)
def generate_finetuned(prompt: str, max_tokens: int = 150) -> str:
    """
    Generate response using the FINE-TUNED model.

    This is the "after" - specialized for customer service.
    Should be more helpful, specific, and aligned with support best practices.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    cache_dir = "/models/qwen-2.5-7b-instruct-finetuned"

    tokenizer = AutoTokenizer.from_pretrained(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )

    return response.strip()


@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_volume},
    timeout=600,
)
def compare_models(prompt: str, max_tokens: int = 150) -> dict:
    """
    Run the same prompt through both models and return side-by-side comparison.

    This is the key demo moment - showing how fine-tuning improves the model
    for the specific task.
    """
    print(f"🔬 Comparing models on prompt: '{prompt}'")

    # Generate with both models
    print("\n📊 Running base model...")
    base_response = generate_base.local(prompt, max_tokens)

    print("📊 Running fine-tuned model...")
    finetuned_response = generate_finetuned.local(prompt, max_tokens)

    return {
        "prompt": prompt,
        "base_model": {
            "name": "Qwen 2.5 7B Instruct (base)",
            "response": base_response,
        },
        "finetuned_model": {
            "name": "Qwen 2.5 7B Instruct (fine-tuned on customer service)",
            "response": finetuned_response,
        },
    }


# Web endpoint for production serving
@app.function(
    image=image,
    gpu="T4",
    volumes={"/models": model_volume},
)
@modal.web_endpoint(method="POST")
def inference_endpoint(prompt: str, use_finetuned: bool = True, max_tokens: int = 150):
    """
    Production web endpoint for serving the model.

    Usage:
        curl -X POST https://your-modal-url/inference_endpoint \
          -H "Content-Type: application/json" \
          -d '{"prompt": "How do I reset my password?", "use_finetuned": true}'
    """
    if use_finetuned:
        response = generate_finetuned.local(prompt, max_tokens)
        model_version = "fine-tuned"
    else:
        response = generate_base.local(prompt, max_tokens)
        model_version = "base"

    return {
        "model": model_version,
        "prompt": prompt,
        "response": response,
    }


@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing.
    Run with: modal run src/functions/serve.py
    """
    test_prompts = [
        "How do I reset my password?",
        "Can I upgrade my plan?",
        "What's your refund policy?",
    ]

    print("🎯 Before/After Comparison: Base vs Fine-tuned Model\n")
    print("=" * 80)

    for prompt in test_prompts:
        print(f"\n🔍 Prompt: {prompt}\n")

        result = compare_models.remote(prompt)

        print("📊 BASE MODEL:")
        print(f"   {result['base_model']['response']}\n")

        print("✨ FINE-TUNED MODEL:")
        print(f"   {result['finetuned_model']['response']}\n")

        print("-" * 80)

    print("\n✅ Comparison complete!")
    print("\n💡 The fine-tuned model should:")
    print("   • Give more specific, actionable answers")
    print("   • Follow support best practices")
    print("   • Mention specific features from training data")
    print("   • Sound more like a professional support agent\n")
