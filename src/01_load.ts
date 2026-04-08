/**
 * Step 1: Load Model to Modal Volume
 *
 * This script orchestrates the model loading process by invoking the Python Modal Function.
 * It demonstrates how TypeScript can call Python Modal Functions seamlessly.
 *
 * What happens:
 * 1. Connects to Modal using the TypeScript SDK
 * 2. Looks up the Python Modal Function we defined in load_model.py
 * 3. Invokes the function remotely on Modal's infrastructure
 * 4. The model gets downloaded and cached in a Modal Volume
 *
 * Modal Volumes explained:
 * - Persistent distributed file system
 * - Survives across function runs
 * - Perfect for large ML model weights (~14GB for Qwen 2.5 7B)
 * - First run downloads the model, subsequent runs use the cached version
 * - Can scale to 2.5 GB/s bandwidth
 */

import { ModalClient } from "modal";

async function loadModel() {
  console.log("🔌 Connecting to Modal...\n");

  // Initialize the Modal client
  // This uses your MODAL_TOKEN_ID and MODAL_TOKEN_SECRET from environment
  const modal = new ModalClient();

  try {
    // Look up the Python Modal app we defined
    console.log("📦 Looking up Modal app: vertical-ai-load-model");
    const app = await modal.apps.fromName("vertical-ai-load-model");

    // Get a reference to the download_model function
    console.log("🔍 Finding download_model function...\n");
    const downloadFn = await modal.functions.lookup(
      app.appId,
      "download_model"
    );

    console.log("🚀 Invoking download_model on Modal infrastructure...");
    console.log("⏱️  First run: ~5-10 minutes to download 14GB model");
    console.log("⏱️  Subsequent runs: instant (model cached in Volume)\n");

    // Invoke the function remotely
    // This runs on Modal's infrastructure with the GPU and dependencies we specified
    const result = await downloadFn.callRemote();

    console.log("\n✅ Model loading complete!");
    console.log("📊 Result:", JSON.stringify(result, null, 2));

    if (result.status === "cached") {
      console.log("\n💡 Model was already in the Volume - no download needed!");
    } else {
      console.log("\n💡 Model downloaded and committed to Volume!");
      console.log(`📁 Files cached: ${result.files_cached}`);
    }

    console.log(`\n📂 Model location: ${result.cache_dir}`);
    console.log(`🤖 Model name: ${result.model_name}`);

    // Now let's test inference to make sure the model works
    console.log("\n🧪 Testing inference...");
    const testFn = await modal.functions.lookup(app.appId, "test_inference");
    const testResult = await testFn.callRemote(
      "How do I reset my password?"
    );

    console.log("\n🤖 Test inference result:");
    console.log(`Prompt: ${testResult.prompt}`);
    console.log(`Response: ${testResult.response}`);

    console.log("\n✨ All done! Model is ready for RL rollouts and fine-tuning.");
  } catch (error) {
    console.error("\n❌ Error:", error);
    console.error("\n💡 Make sure you have:");
    console.error("   1. Set up Modal credentials (modal token new)");
    console.error("   2. Deployed the Python function (modal deploy src/functions/load_model.py)");
    console.error("   3. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env");
    throw error;
  }
}

// Run the script
loadModel().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
