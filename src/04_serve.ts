/**
 * Step 4: Serve and Compare Models
 *
 * This is the PAYOFF MOMENT - we compare the base model vs fine-tuned model
 * side-by-side on customer service queries.
 *
 * This demonstrates:
 * - Modal Functions for production inference serving
 * - Fast model loading from Volumes
 * - GPU-accelerated inference
 * - Easy A/B testing between model versions
 *
 * The before/after comparison shows the value of the entire pipeline:
 * Load → RL Rollouts → Fine-tune → Serve
 *
 * This is what companies like Intercom and Cursor do - build vertical AI models
 * specialized for their specific use case, all running on Modal's infrastructure.
 */

import { ModalClient } from "modal";

// Test queries that will show the difference between base and fine-tuned
const TEST_QUERIES = [
  "How do I reset my password?",
  "Can I upgrade my plan mid-cycle?",
  "What's your refund policy?",
];

async function serveAndCompare() {
  console.log("🎯 Model Comparison: Base vs Fine-tuned\n");
  console.log("This is the payoff moment of the vertical AI pipeline!");
  console.log("We'll compare how the generic model vs specialized model handle");
  console.log("customer service queries.\n");

  // Initialize Modal client
  console.log("🔌 Connecting to Modal...");
  const modal = new ModalClient();

  try {
    // Get both inference functions
    console.log("🔍 Finding inference functions...\n");
    const baseModelFn = await modal.functions.fromName(
      "vertical-ai-serve",
      "generate_base"
    );
    const finetunedModelFn = await modal.functions.fromName(
      "vertical-ai-serve",
      "generate_finetuned"
    );

    console.log("=".repeat(80));
    console.log("🔬 BEFORE/AFTER COMPARISON");
    console.log("=".repeat(80));

    // Run comparisons for each test query
    for (let i = 0; i < TEST_QUERIES.length; i++) {
      const query = TEST_QUERIES[i];

      console.log(`\n[${i + 1}/${TEST_QUERIES.length}] Query: "${query}"\n`);

      try {
        // Call base model
        console.log("📊 Running base model...");
        const baseResult = await baseModelFn.remote([query, 150]);

        // Call fine-tuned model
        console.log("✨ Running fine-tuned model...");
        const finetunedResult = await finetunedModelFn.remote([query, 150]);

        console.log("\n📊 BASE MODEL (Generic Qwen 2.5 7B):");
        console.log(`   ${baseResult}\n`);

        console.log("✨ FINE-TUNED MODEL (Specialized for Customer Service):");
        console.log(`   ${finetunedResult}\n`);

        console.log("-".repeat(80));
      } catch (error) {
        console.error(`\n❌ Error on query ${i + 1}:`, (error as Error).message);
        console.log("Continuing with next query...\n");
        continue;
      }

      // Small delay between comparisons for readability
      await new Promise((resolve) => setTimeout(resolve, 2000));
    }

    console.log("\n✅ Comparison complete!\n");

    console.log("🎯 What you should notice:");
    console.log("   1. Fine-tuned model gives more SPECIFIC answers");
    console.log("   2. Mentions actual features from the training data");
    console.log("   3. Follows support best practices (clear steps, helpful tone)");
    console.log("   4. More actionable and aligned with the product\n");

    console.log("💡 The vertical AI loop:");
    console.log("   ✓ Step 1: Loaded Qwen 2.5 7B to Modal Volume");
    console.log("   ✓ Step 2: Ran RL rollouts with parallel inference");
    console.log("   ✓ Step 3: Fine-tuned on A100 GPU with customer service data");
    console.log("   ✓ Step 4: Served both models for comparison on T4 GPU\n");

    console.log("🚀 All three Modal primitives in action:");
    console.log("   • VOLUMES: Persistent model storage (~14GB)");
    console.log("   • FUNCTIONS: GPU-powered ML operations (auto-scaling)");
    console.log("   • Full pipeline running on one platform with one SDK\n");

    console.log("✨ This entire infrastructure loop runs on Modal,");
    console.log("   and you only pay for actual compute time used.\n");

    console.log("🌐 Production Deployment:");
    console.log("   Your fine-tuned model is deployed at:");
    console.log("   https://dancleary54--vertical-ai-serve-inference-endpoint.modal.run");
    console.log("   It's ready for production use with auto-scaling!\n");

  } catch (error) {
    console.error("\n❌ Error during comparison:", error);
    console.error("\n💡 Troubleshooting:");
    console.error("   1. Make sure all previous steps completed:");
    console.error("      npm run load && npm run rollouts && npm run finetune");
    console.error("   2. Check that the Python function is deployed:");
    console.error("      modal deploy src/functions/serve.py");
    console.error("   3. Verify both base and fine-tuned models are in the Volume");
    throw error;
  }
}

// Run the script
serveAndCompare().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
