/**
 * Step 3: Fine-tune the Model
 *
 * This script orchestrates fine-tuning by calling the Python Modal Function.
 * It demonstrates Modal Functions with GPU - running on an A100 for fast training.
 *
 * Modal Functions explained:
 * - Defined in Python (for GPU/ML workloads)
 * - Called from any language (we're using TypeScript)
 * - Automatically scaled, no infrastructure management
 * - Access to high-end GPUs (A100, H100) on demand
 * - Functions can run for hours (we set 2 hour timeout for fine-tuning)
 *
 * What this does:
 * 1. Loads the customer service dataset
 * 2. Optionally loads RL rollout results from step 2
 * 3. Calls the Python fine-tuning function on Modal
 * 4. The function runs on an A100 GPU
 * 5. Uses LoRA for parameter-efficient fine-tuning (~1% of params)
 * 6. Saves fine-tuned model back to the Volume
 *
 * The before/after comparison happens in step 4 (serve.ts)
 */

import { ModalClient } from "modal";
import { readFileSync, existsSync } from "fs";
import { join } from "path";

async function finetune() {
  console.log("🔥 Starting Model Fine-tuning on Modal\n");

  // Load training data
  console.log("📚 Loading training data...");
  const datasetPath = join(process.cwd(), "data", "dataset.jsonl");
  const datasetContent = readFileSync(datasetPath, "utf-8");
  console.log(`✅ Loaded ${datasetContent.split("\n").filter((l) => l.trim()).length} training examples`);

  // Optionally load RL rollout results
  let rolloutContent: string | undefined;
  const rolloutPath = join(process.cwd(), "data", "rollout_results.jsonl");

  if (existsSync(rolloutPath)) {
    console.log("📊 Loading RL rollout results...");
    rolloutContent = readFileSync(rolloutPath, "utf-8");
    const rolloutCount = rolloutContent.split("\n").filter((l) => l.trim()).length;
    console.log(`✅ Loaded ${rolloutCount} rollout results`);
    console.log("💡 High-scoring rollouts will be added to training data\n");
  } else {
    console.log("ℹ️  No rollout results found - fine-tuning on dataset only\n");
  }

  // Initialize Modal client
  console.log("🔌 Connecting to Modal...");
  const modal = new ModalClient();

  try {
    // Get the fine-tuning function
    console.log("🔍 Finding finetune_model function...\n");
    const finetuneFn = await modal.functions.fromName(
      "vertical-ai-finetune",
      "finetune_model"
    );

    console.log("🚀 Starting fine-tuning on Modal A100 GPU...");
    console.log("⏱️  This will take ~20-40 minutes with enhanced training");
    console.log("💡 Modal automatically provisions the GPU and scales down when done\n");

    console.log("Training configuration:");
    console.log("  • GPU: A100 (40GB)");
    console.log("  • Method: LoRA (parameter-efficient fine-tuning)");
    console.log("  • Trainable params: ~1% of total (much faster!)");
    console.log("  • Epochs: 10 (increased for better results)");
    console.log("  • Learning rate: 2e-5");
    console.log("  • Batch size: 4 (with gradient accumulation)\n");

    const startTime = Date.now();

    // Invoke the fine-tuning function
    // This runs remotely on Modal's infrastructure with an A100 GPU
    const result = await finetuneFn.remote(
      [datasetContent, rolloutContent, "qwen-2.5-7b-instruct-finetuned", 10, 2e-5]
    );

    const duration = (Date.now() - startTime) / 1000;

    console.log("\n✅ Fine-tuning complete!");
    console.log(`⏱️  Total time: ${(duration / 60).toFixed(1)} minutes`);
    console.log("\n📊 Training Results:");
    console.log(`   Final loss: ${result.final_loss.toFixed(4)}`);
    console.log(`   Epochs: ${result.epochs}`);
    console.log(`   Training samples: ${result.training_samples}`);
    console.log(`   Trainable parameters: ${result.trainable_params.toLocaleString()}`);
    console.log(`   Model saved to: ${result.output_dir}`);

    console.log("\n💾 Fine-tuned model saved to Modal Volume!");
    console.log("✨ Ready for inference serving (run npm run serve)\n");

    console.log("🎯 What happened:");
    console.log("   1. Base model loaded from Volume");
    console.log("   2. LoRA adapters trained on customer service data");
    console.log("   3. Model fine-tuned on A100 GPU");
    console.log("   4. Fine-tuned weights saved back to Volume");
    console.log("   5. GPU automatically released\n");

    console.log("💡 Modal automatically:");
    console.log("   • Provisioned an A100 GPU");
    console.log("   • Loaded the 14GB model");
    console.log("   • Ran training for 10 epochs");
    console.log("   • Saved results to persistent storage");
    console.log("   • Released resources when done");
    console.log("   • You only pay for actual GPU time used!\n");

  } catch (error) {
    console.error("\n❌ Error during fine-tuning:", error);
    console.error("\n💡 Troubleshooting:");
    console.error("   1. Make sure step 1 (load model) completed successfully");
    console.error("   2. Check that the Python function is deployed:");
    console.error("      modal deploy src/functions/finetune.py");
    console.error("   3. Verify your Modal account has GPU access");
    throw error;
  }
}

// Run the script
finetune().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
