/**
 * Step 2: RL Rollouts with Modal Sandboxes
 *
 * This demonstrates the CORE differentiator of Modal: massively parallel sandboxed execution.
 * We spin up multiple isolated environments where the model attempts customer service queries
 * and gets scored based on response quality.
 *
 * Modal Sandboxes explained:
 * - Isolated execution environments (secure containers)
 * - Can scale to 100,000+ concurrent sandboxes
 * - Perfect for RL rollouts, AI agent execution, code evaluation
 * - Each sandbox has its own filesystem, process space, network
 * - This is what companies like Cursor use for their AI infrastructure
 *
 * What this script does:
 * 1. Loads our customer service dataset
 * 2. Samples queries to test the model against
 * 3. Spins up parallel sandboxes (10 concurrent by default, but Modal scales to 100k+)
 * 4. Each sandbox runs the model against a query and evaluates the response
 * 5. Collects results and scores for fine-tuning
 *
 * This is a simplified RL setup - production systems would have:
 * - More sophisticated reward models
 * - PPO/RLHF training loops
 * - Human feedback integration
 * But this shows the infrastructure that makes it possible.
 */

import { ModalClient } from "modal";
import { readFileSync, writeFileSync } from "fs";
import { join } from "path";

interface DatasetEntry {
  question: string;
  answer: string;
}

interface RolloutResult {
  question: string;
  modelResponse: string;
  expectedAnswer: string;
  score: number;
  sandboxId: string;
  executionTime: number;
}

async function runRLRollouts() {
  console.log("🎯 Starting RL Rollouts with Modal Sandboxes\n");

  // Load the dataset
  console.log("📚 Loading customer service dataset...");
  const datasetPath = join(process.cwd(), "data", "dataset.jsonl");
  const datasetContent = readFileSync(datasetPath, "utf-8");
  const dataset: DatasetEntry[] = datasetContent
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));

  console.log(`✅ Loaded ${dataset.length} Q&A pairs\n`);

  // Sample queries for rollouts (using first 10 for demo, but Modal can handle 100k+)
  const numRollouts = 10;
  const sampledQueries = dataset.slice(0, numRollouts);
  console.log(`🎲 Running ${numRollouts} parallel rollouts...`);
  console.log(`💡 Modal can scale to 100,000+ concurrent sandboxes!\n`);

  // Initialize Modal client
  const modal = new ModalClient();

  try {
    // Create/get the app
    console.log("📦 Setting up Modal app...");
    const app = await modal.apps.fromName("vertical-ai-rl-rollouts", {
      createIfMissing: true,
    });

    // Get the model volume (created in step 1)
    console.log("📂 Mounting model volume...");
    const modelVolume = await modal.volumes.fromName("qwen-model-cache");

    // Create a Python image with ML dependencies
    // Even though we're in TypeScript, the sandbox needs Python to run the model
    const image = modal.images.fromRegistry("python:3.11-slim").run(
      "pip",
      "install",
      "transformers==4.46.0",
      "torch==2.5.1",
      "accelerate==1.1.0"
    );

    console.log("🚀 Launching sandboxes in parallel...\n");

    const startTime = Date.now();

    // Launch all sandboxes in parallel
    // This is the power of Modal - each sandbox is isolated and can run concurrently
    const rolloutPromises = sampledQueries.map(async (query, idx) => {
      const sandboxStart = Date.now();

      // Create a sandbox for this rollout
      const sandbox = await modal.sandboxes.create(app, image, {
        volumes: { "/models": modelVolume.readOnly() }, // Mount model as read-only
        timeout: 300, // 5 minute timeout
        cpu: 2,
        memory: 8192, // 8GB RAM
      });

      console.log(
        `[Sandbox ${idx + 1}/${numRollouts}] Created: ${sandbox.sandboxId}`
      );

      try {
        // Create a Python script to run inference
        // In a real system, you'd have a more sophisticated evaluation setup
        const inferenceScript = `
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/models/qwen-2.5-7b-instruct"
question = "${query.question.replace(/"/g, '\\"')}"

# Load model
tokenizer = AutoTokenizer.from_pretrained(cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    cache_dir,
    device_map="cpu",  # Using CPU for demo, production would use GPU
    torch_dtype=torch.float32,
)

# Run inference
messages = [{"role": "user", "content": question}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response.strip())
`;

        // Write the script to the sandbox
        await sandbox.exec(["mkdir", "-p", "/tmp"]);
        await sandbox.exec([
          "sh",
          "-c",
          `cat > /tmp/infer.py << 'EOF'\n${inferenceScript}\nEOF`,
        ]);

        // Run inference
        console.log(`[Sandbox ${idx + 1}] Running inference...`);
        const inferProc = await sandbox.exec(["python", "/tmp/infer.py"]);
        const modelResponse = (await inferProc.stdout.readText()).trim();
        await inferProc.wait();

        // Simple scoring: check if key terms from expected answer appear in model response
        // Production systems would use more sophisticated reward models
        const score = calculateScore(modelResponse, query.answer);

        const executionTime = Date.now() - sandboxStart;

        console.log(
          `[Sandbox ${idx + 1}] ✅ Complete (${(executionTime / 1000).toFixed(1)}s, score: ${score.toFixed(2)})`
        );

        // Clean up
        await sandbox.terminate();

        return {
          question: query.question,
          modelResponse,
          expectedAnswer: query.answer,
          score,
          sandboxId: sandbox.sandboxId,
          executionTime,
        } as RolloutResult;
      } catch (error) {
        console.error(`[Sandbox ${idx + 1}] ❌ Error:`, error);
        await sandbox.terminate();
        throw error;
      }
    });

    // Wait for all rollouts to complete
    const results = await Promise.all(rolloutPromises);

    const totalTime = Date.now() - startTime;

    console.log(`\n✨ All rollouts complete in ${(totalTime / 1000).toFixed(1)}s!`);
    console.log(`⚡ Average time per rollout: ${(totalTime / numRollouts / 1000).toFixed(1)}s`);

    // Calculate aggregate stats
    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    console.log(`\n📊 Aggregate Statistics:`);
    console.log(`   Average score: ${avgScore.toFixed(2)}`);
    console.log(`   Total rollouts: ${results.length}`);
    console.log(`   Successful: ${results.filter((r) => r.score > 0.5).length}`);

    // Save results for fine-tuning
    const outputPath = join(process.cwd(), "data", "rollout_results.jsonl");
    const outputContent = results.map((r) => JSON.stringify(r)).join("\n");
    writeFileSync(outputPath, outputContent);

    console.log(`\n💾 Results saved to: ${outputPath}`);
    console.log("\n🎯 Top 3 responses:");
    results
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .forEach((r, idx) => {
        console.log(`\n${idx + 1}. Question: ${r.question}`);
        console.log(`   Model: ${r.modelResponse.substring(0, 100)}...`);
        console.log(`   Score: ${r.score.toFixed(2)}`);
      });

    console.log("\n✅ RL rollouts complete! Ready for fine-tuning.");
  } catch (error) {
    console.error("\n❌ Error during rollouts:", error);
    throw error;
  }
}

/**
 * Simple scoring function based on keyword overlap.
 * Production systems would use:
 * - BERT-based semantic similarity
 * - Human feedback (RLHF)
 * - Multi-dimensional reward models
 * - Task-specific metrics
 */
function calculateScore(modelResponse: string, expectedAnswer: string): number {
  const modelWords = new Set(
    modelResponse.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
  );
  const expectedWords = new Set(
    expectedAnswer.toLowerCase().split(/\s+/).filter((w) => w.length > 3)
  );

  // Calculate Jaccard similarity
  const intersection = new Set([...modelWords].filter((w) => expectedWords.has(w)));
  const union = new Set([...modelWords, ...expectedWords]);

  return intersection.size / union.size;
}

// Run the script
runRLRollouts().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
