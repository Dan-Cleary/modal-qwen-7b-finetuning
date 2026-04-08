"""
Modal Function for fine-tuning Qwen 2.5 7B on customer service data.

This demonstrates Modal Functions with GPU - we're using an A100 (or H100)
to fine-tune the model on our synthetic dataset + RL rollout results.

The fine-tuned weights are saved back to the same Modal Volume, so the
serving function can load them instantly.
"""

import modal
import json

app = modal.App("vertical-ai-finetune")

# Reuse the same volume where we stored the base model
model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

# Image with fine-tuning dependencies
# Adding peft for LoRA (parameter-efficient fine-tuning) and datasets for data loading
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers==4.46.0",
        "torch==2.5.1",
        "accelerate==1.1.0",
        "peft==0.13.0",  # For LoRA fine-tuning
        "datasets==3.0.0",
        "huggingface_hub==0.26.0",
        "bitsandbytes==0.44.0",  # For efficient quantization
    )
)


@app.function(
    image=image,
    gpu="A100",  # Using A100 for fast fine-tuning (A10G or T4 also work, just slower)
    volumes={"/models": model_volume},
    timeout=7200,  # 2 hours for fine-tuning
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def finetune_model(
    dataset_jsonl: str,
    rollout_results_jsonl: str = None,
    output_name: str = "qwen-2.5-7b-instruct-finetuned",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune Qwen 2.5 7B using LoRA on customer service data.

    Args:
        dataset_jsonl: JSONL string containing training data
        rollout_results_jsonl: Optional JSONL string with RL rollout results
        output_name: Name for the fine-tuned model in the Volume
        num_epochs: Number of training epochs
        learning_rate: Learning rate for fine-tuning

    Returns:
        Dict with training metrics and model location
    """
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    import torch
    import os

    print("🚀 Starting fine-tuning process...")

    # Load base model from Volume
    base_model_dir = "/models/qwen-2.5-7b-instruct"
    output_dir = f"/models/{output_name}"

    print(f"📂 Loading base model from {base_model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False,  # Disable cache for training
    )

    print(f"✅ Model loaded: {model.num_parameters():,} parameters")

    # Prepare training data
    print("\n📚 Preparing training data...")

    # Parse dataset
    dataset_lines = [json.loads(line) for line in dataset_jsonl.strip().split("\n")]

    # Optionally add RL rollout results
    if rollout_results_jsonl:
        rollout_lines = [
            json.loads(line) for line in rollout_results_jsonl.strip().split("\n")
        ]
        # Add high-scoring rollouts to training data
        for rollout in rollout_lines:
            if rollout.get("score", 0) > 0.5:  # Only use good rollouts
                dataset_lines.append({
                    "question": rollout["question"],
                    "answer": rollout["modelResponse"],
                })

    print(f"📊 Training samples: {len(dataset_lines)}")

    # Format data for instruction fine-tuning
    def format_instruction(example):
        """Format each example as a chat conversation."""
        messages = [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return {"text": text}

    # Create dataset
    dataset = Dataset.from_list(dataset_lines)
    dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

    # Tokenize
    def tokenize_function(examples):
        outputs = tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        outputs["labels"] = outputs["input_ids"].copy()
        return outputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    print(f"✅ Data prepared: {len(tokenized_dataset)} samples")

    # Set up LoRA configuration
    # LoRA is parameter-efficient: we only train ~1% of parameters
    print("\n⚙️  Configuring LoRA for parameter-efficient fine-tuning...")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"🎯 Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Training arguments
    training_args = TrainingArguments(
        output_dir="/tmp/training_output",  # Temporary dir for checkpoints
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=True,  # Use mixed precision for speed
        logging_steps=10,
        save_strategy="epoch",
        warmup_steps=100,
        report_to="none",  # Disable wandb etc
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Initialize trainer
    print("\n🏋️ Initializing trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train!
    print("\n🔥 Starting training...")
    print(f"   Epochs: {num_epochs}")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   GPU: A100\n")

    train_result = trainer.train()

    print("\n✅ Training complete!")
    print(f"📊 Final loss: {train_result.training_loss:.4f}")

    # Save the fine-tuned model to the Volume
    print(f"\n💾 Saving fine-tuned model to {output_dir}...")

    # Merge LoRA weights with base model for faster inference
    model = model.merge_and_unload()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Commit to Volume
    print("💾 Committing to Volume...")
    model_volume.commit()

    print("✅ Fine-tuning complete and model saved!")

    return {
        "status": "success",
        "output_dir": output_dir,
        "final_loss": float(train_result.training_loss),
        "epochs": num_epochs,
        "training_samples": len(tokenized_dataset),
        "trainable_params": trainable_params,
    }


@app.local_entrypoint()
def main():
    """
    Local entrypoint for testing.
    Run with: modal run src/functions/finetune.py
    """
    # Load dataset for testing
    import os

    dataset_path = os.path.join(os.getcwd(), "data", "dataset.jsonl")
    with open(dataset_path) as f:
        dataset_jsonl = f.read()

    print("🚀 Starting fine-tuning job on Modal...")
    result = finetune_model.remote(
        dataset_jsonl=dataset_jsonl,
        num_epochs=1,  # Quick test with 1 epoch
    )

    print(f"\n📋 Result: {json.dumps(result, indent=2)}")
