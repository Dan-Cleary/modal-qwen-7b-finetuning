/**
 * Step 2: RL Rollouts with Modal Functions
 *
 * This demonstrates Modal's ability to run massively parallel workloads.
 * We run multiple inference calls in parallel, score the responses, and save
 * results for fine-tuning.
 *
 * NOTE: In production, you'd use Modal Sandboxes for fully isolated execution
 * environments (perfect for untrusted code, AI agents, etc). For this demo,
 * we're using parallel Function calls which is simpler with the TypeScript SDK.
 *
 * Modal's scalability:
 * - Functions auto-scale based on load
 * - Can handle thousands of concurrent requests
 * - Each invocation is isolated
 * - GPU functions spin up/down automatically
 *
 * What this script does:
 * 1. Loads our customer service dataset
 * 2. Samples queries to test the model against
 * 3. Runs parallel inference calls (10 concurrent by default)
 * 4. Scores each response based on quality
 * 5. Saves results for fine-tuning
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
  executionTime: number;
}

async function runRLRollouts() {
  console.log("🎯 Starting RL Rollouts with Modal Functions\n");

  // Load the dataset
  console.log("📚 Loading customer service dataset...");
  const datasetPath = join(process.cwd(), "data", "dataset.jsonl");
  const datasetContent = readFileSync(datasetPath, "utf-8");
  const dataset: DatasetEntry[] = datasetContent
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));

  console.log(`✅ Loaded ${dataset.length} Q&A pairs\n`);

  // Sample queries for rollouts
  const numRollouts = 10;
  const sampledQueries = dataset.slice(0, numRollouts);
  console.log(`🎲 Running ${numRollouts} parallel rollouts...`);
  console.log(`💡 Modal auto-scales to handle the load!\n`);

  // Initialize Modal client
  const modal = new ModalClient();

  try {
    // Get the test_inference function from our deployed app
    console.log("🔍 Looking up inference function...");
    const inferenceFn = await modal.functions.fromName(
      "vertical-ai-load-model",
      "test_inference"
    );

    console.log("🚀 Launching parallel inference calls...\n");

    const startTime = Date.now();

    // Run all inferences in parallel
    const rolloutPromises = sampledQueries.map(async (query, idx) => {
      const rolloutStart = Date.now();

      console.log(`[Rollout ${idx + 1}/${numRollouts}] Running: "${query.question}"`);

      try {
        // Call the inference function
        const result = await inferenceFn.remote([query.question]);
        const modelResponse = result.response;

        // Score the response
        const score = calculateScore(modelResponse, query.answer);

        const executionTime = Date.now() - rolloutStart;

        console.log(
          `[Rollout ${idx + 1}] ✅ Complete (${(executionTime / 1000).toFixed(1)}s, score: ${score.toFixed(2)})`
        );

        return {
          question: query.question,
          modelResponse,
          expectedAnswer: query.answer,
          score,
          executionTime,
        } as RolloutResult;
      } catch (error) {
        console.error(`[Rollout ${idx + 1}] ❌ Error: ${(error as Error).message}`);
        // Return null for failed rollouts, we'll filter them out
        return null;
      }
    });

    // Wait for all rollouts to complete, filter out failures
    const allResults = await Promise.all(rolloutPromises);
    const results = allResults.filter((r): r is RolloutResult => r !== null);

    const totalTime = Date.now() - startTime;

    console.log(`\n✨ Rollouts complete in ${(totalTime / 1000).toFixed(1)}s!`);
    console.log(`   Successful: ${results.length}/${numRollouts}`);
    if (results.length < numRollouts) {
      console.log(`   ⚠️  ${numRollouts - results.length} rollout(s) timed out or failed`);
    }
    console.log(`⚡ Average time per successful rollout: ${(totalTime / results.length / 1000).toFixed(1)}s`);

    // Calculate aggregate stats
    if (results.length === 0) {
      console.error("\n❌ No successful rollouts! Cannot continue.");
      throw new Error("All rollouts failed");
    }

    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    console.log(`\n📊 Aggregate Statistics:`);
    console.log(`   Average score: ${avgScore.toFixed(2)}`);
    console.log(`   Total successful rollouts: ${results.length}`);
    console.log(`   High-scoring (>0.3): ${results.filter((r) => r.score > 0.3).length}`);

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
    console.log("\n💡 In production, you'd use Modal Sandboxes for:");
    console.log("   • Fully isolated execution environments");
    console.log("   • Running untrusted code safely");
    console.log("   • AI agent evaluation at 100k+ concurrency");
    console.log("   • This demo uses Functions for simplicity with TS SDK");

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
