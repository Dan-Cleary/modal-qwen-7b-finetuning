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
 * - The "keep_warm" feature to reduce cold starts
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
  "How do I enable two-factor authentication?",
  "Do you offer student discounts?",
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
    // Look up the serving app
    console.log("📦 Looking up Modal app: vertical-ai-serve");
    const app = await modal.apps.fromName("vertical-ai-serve");

    // Get the comparison function
    console.log("🔍 Finding compare_models function...\n");
    const compareFn = await modal.functions.lookup(app.appId, "compare_models");

    console.log("=" .repeat(80));
    console.log("🔬 BEFORE/AFTER COMPARISON");
    console.log("=" .repeat(80));

    // Run comparisons for each test query
    for (let i = 0; i < TEST_QUERIES.length; i++) {
      const query = TEST_QUERIES[i];

      console.log(`\n[${i + 1}/${TEST_QUERIES.length}] Query: "${query}"\n`);

      // Call Modal function to compare both models
      const result = await compareFn.callRemote(query, 150);

      console.log("📊 BASE MODEL (Generic Qwen 2.5 7B):");
      console.log(`   ${result.base_model.response}\n`);

      console.log("✨ FINE-TUNED MODEL (Specialized for Customer Service):");
      console.log(`   ${result.finetuned_model.response}\n`);

      console.log("-".repeat(80));

      // Small delay between comparisons for readability
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }

    console.log("\n✅ Comparison complete!\n");

    console.log("🎯 What you should notice:");
    console.log("   1. Fine-tuned model gives more SPECIFIC answers");
    console.log("   2. Mentions actual features from the training data");
    console.log("   3. Follows support best practices (clear steps, helpful tone)");
    console.log("   4. More actionable and aligned with the product\n");

    console.log("💡 The vertical AI loop:");
    console.log("   ✓ Step 1: Loaded Qwen 2.5 7B to Modal Volume");
    console.log("   ✓ Step 2: Ran RL rollouts in Modal Sandboxes (massively parallel)");
    console.log("   ✓ Step 3: Fine-tuned on A100 GPU with customer service data");
    console.log("   ✓ Step 4: Served both models for comparison on T4 GPU\n");

    console.log("🚀 All three Modal primitives in action:");
    console.log("   • VOLUMES: Persistent model storage");
    console.log("   • SANDBOXES: Isolated RL rollout environments (scale to 100k+)");
    console.log("   • FUNCTIONS: GPU-powered ML operations (load, train, serve)\n");

    console.log("✨ This entire infrastructure loop runs on one platform,");
    console.log("   with one SDK, and you only pay for actual compute time used.\n");

    // Show the web endpoint info
    console.log("🌐 Production Deployment:");
    console.log("   Your fine-tuned model is now served via Modal Functions!");
    console.log("   It has a web endpoint ready for production use.");
    console.log("   Modal handles scaling, GPUs, and infrastructure automatically.\n");

    console.log("📚 Next steps:");
    console.log("   • Deploy to production: modal deploy src/functions/serve.py");
    console.log("   • Get web URL: modal app show vertical-ai-serve");
    console.log("   • Call the API from your application");
    console.log("   • Iterate: collect feedback → more rollouts → fine-tune again\n");

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
