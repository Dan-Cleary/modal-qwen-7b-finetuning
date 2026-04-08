# Vertical AI Infrastructure Demo with Modal

> **Building specialized AI models the way companies like Intercom and Cursor do it**
>
> This repository demonstrates the complete vertical AI infrastructure loop: loading models, running RL rollouts at scale, fine-tuning on GPUs, and serving inference — all on Modal using a TypeScript-first approach.

## What This Is

This is an open-source demo showing how to build **vertical AI models** — AI systems specialized for specific tasks rather than generic ones. It's the same pattern that companies like Intercom (customer service AI) and Cursor (code completion AI) use to build superior AI products.

We implement the full loop in miniature:

1. **Load** a foundation model (Qwen 2.5 7B) into persistent storage
2. **Evaluate** it with RL rollouts in thousands of parallel sandboxed environments
3. **Fine-tune** it on domain-specific data (customer service)
4. **Serve** the specialized model for inference

The "before/after" comparison — generic model vs fine-tuned model answering customer service questions — is the payoff moment that shows why this infrastructure matters.

## Why Modal?

**All three primitives working together in one place:**

- **🗄️ Volumes**: Persistent distributed file system for model weights (~14GB)
  - Write once, read many - perfect for ML models
  - Fast: up to 2.5 GB/s bandwidth
  - Survives across function runs

- **📦 Sandboxes**: Isolated execution environments that scale to 100,000+ concurrent instances
  - Perfect for RL rollouts and AI agent execution
  - Each sandbox is fully isolated with its own filesystem and process space
  - This is what makes companies like Cursor's infrastructure possible

- **⚡ Functions**: Serverless GPU functions for ML workloads
  - Access to A100, H100, T4 GPUs on demand
  - Only pay for actual compute time used
  - Auto-scaling and zero infrastructure management

This is the only platform where you can run the entire vertical AI loop with one SDK.

## What's Unique About This Demo

**TypeScript-First**: Most Modal examples use Python. This demo uses TypeScript for orchestration and only drops to Python where necessary (GPU functions), showcasing Modal's multi-language support.

**Full Pipeline**: Most demos show one piece (inference OR training). This shows the complete loop that production AI companies actually use.

**Sandbox-Powered RL**: The RL rollouts demonstrate Modal's killer feature — the ability to spin up thousands of isolated environments in parallel. At scale, this is what makes modern AI training possible.

## Prerequisites

- Node.js 18+ and npm
- Python 3.11+ (for Modal Functions)
- A Modal account ([sign up here](https://modal.com))
- Optional: HuggingFace account for gated models

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/modal-vertical-ai-demo.git
cd modal-vertical-ai-demo
npm install
```

### 2. Set Up Modal

Install the Modal CLI and authenticate:

```bash
pip install modal
modal token new
```

This will open a browser to authenticate. Once complete, your credentials are saved locally.

### 3. Configure Environment (Optional)

Copy `.env.example` to `.env` and add your tokens:

```bash
cp .env.example .env
```

For the TypeScript SDK, add:
```
MODAL_TOKEN_ID=your_token_id
MODAL_TOKEN_SECRET=your_token_secret
```

Get these from: https://modal.com/settings

### 4. Create HuggingFace Secret (Optional)

If using gated models, create a Modal secret:

```bash
modal secret create huggingface-secret HF_TOKEN=your_hf_token
```

### 5. Deploy Python Functions

Deploy all Modal functions to Modal's infrastructure:

```bash
modal deploy src/functions/load_model.py
modal deploy src/functions/finetune.py
modal deploy src/functions/serve.py
```

## Running the Demo

Execute the pipeline in order:

### Step 1: Load Model

Download and cache Qwen 2.5 7B (~14GB) to a Modal Volume:

```bash
npm run load
```

**First run**: ~5-10 minutes to download the model  
**Subsequent runs**: Instant (model cached in Volume)

**What happens**: The model is downloaded from HuggingFace and stored in a Modal Volume. This Volume persists across runs, so you only download once.

### Step 2: RL Rollouts

Run parallel RL rollouts using Modal Sandboxes:

```bash
npm run rollouts
```

**Runtime**: ~5-10 minutes for 10 rollouts  
**Scales to**: 100,000+ concurrent sandboxes

**What happens**: 10 sandboxes spin up in parallel, each loading the model and attempting to answer a customer service query. Responses are scored and saved for fine-tuning. Modal can scale this to 100k+ concurrent instances.

### Step 3: Fine-tune

Fine-tune the model on customer service data using an A100 GPU:

```bash
npm run finetune
```

**Runtime**: ~15-30 minutes  
**GPU**: A100 (40GB)

**What happens**: The base model is loaded, LoRA adapters are trained on the customer service dataset + high-scoring rollouts, and the fine-tuned model is saved back to the Volume.

### Step 4: Serve and Compare

Compare base model vs fine-tuned model side-by-side:

```bash
npm run serve
```

**Runtime**: ~2-3 minutes  
**GPU**: T4 (for inference)

**What happens**: Both models are loaded and the same queries are sent to each. You'll see the before/after comparison showing how fine-tuning improves responses.

## Example Output

```
🔬 BEFORE/AFTER COMPARISON
================================================================================

[1/5] Query: "How do I reset my password?"

📊 BASE MODEL (Generic Qwen 2.5 7B):
   You can reset your password by going to the login page and clicking "Forgot Password"...

✨ FINE-TUNED MODEL (Specialized for Customer Service):
   To reset your password, click 'Forgot Password' on the login page, enter your 
   email address, and follow the instructions in the reset email we'll send you. 
   The link expires in 24 hours. Need help? Email support@example.com.

```

**Notice how the fine-tuned model**:
- Gives more specific, actionable steps
- Mentions the 24-hour expiration (from training data)
- Adds a helpful support contact
- Uses a more professional support tone

## Project Structure

```
modal-vertical-ai-demo/
├── README.md                      # This file
├── package.json                   # TypeScript dependencies
├── tsconfig.json                  # TypeScript configuration
├── data/
│   └── dataset.jsonl              # 60 synthetic customer service Q&A pairs
├── src/
│   ├── 01_load.ts                 # TypeScript: Model loading orchestrator
│   ├── 02_rl_rollouts.ts          # TypeScript: Parallel RL rollouts with Sandboxes
│   ├── 03_finetune.ts             # TypeScript: Fine-tuning orchestrator
│   ├── 04_serve.ts                # TypeScript: Inference and comparison
│   └── functions/
│       ├── load_model.py          # Python: Modal Function for loading model
│       ├── finetune.py            # Python: Modal Function for fine-tuning on GPU
│       └── serve.py               # Python: Modal Functions for inference serving
└── .env.example                   # Environment variables template
```

## How It Works

### TypeScript + Python Hybrid Architecture

This demo uses a hybrid approach:

**TypeScript**: Orchestration, data handling, calling Modal Functions, creating Sandboxes
**Python**: Defining Modal Functions that need GPU access (load, fine-tune, serve)

This showcases Modal's flexibility — use the best language for each part of your pipeline.

### The Three Modal Primitives

#### 1. Volumes (Persistent Storage)

```typescript
// TypeScript
const modelVolume = await modal.volumes.fromName("qwen-model-cache");
```

```python
# Python
model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

@app.function(volumes={"/models": model_volume})
def load_model():
    # Model saved here persists across runs
    model.save_pretrained("/models/qwen-2.5-7b-instruct")
    model_volume.commit()
```

**Use case**: Store the 14GB model weights once, reuse thousands of times

#### 2. Sandboxes (Isolated Environments)

```typescript
// Spin up 10 sandboxes in parallel for RL rollouts
const sandbox = await modal.sandboxes.create(app, image, {
  volumes: { "/models": modelVolume },
  cpu: 2,
  memory: 8192,
});

await sandbox.exec(["python", "/tmp/infer.py"]);
```

**Use case**: Run model evaluation in isolated environments, scale to 100k+ concurrent

#### 3. Functions (GPU Workloads)

```python
@app.function(gpu="A100", image=image, volumes={"/models": model_volume})
def finetune_model(dataset_jsonl: str):
    # This runs on an A100 GPU, scales automatically
    model = AutoModelForCausalLM.from_pretrained("/models/...")
    trainer.train()
    model.save_pretrained("/models/...-finetuned")
```

**Use case**: GPU-accelerated fine-tuning without managing infrastructure

### Why This Architecture Matters

Companies building vertical AI need all three:

- **Volumes** to share expensive model weights across workers
- **Sandboxes** to safely evaluate models at massive scale
- **Functions** to run GPU workloads without managing servers

Modal is the only platform that offers all three with one SDK.

## Production Deployment

To deploy the serving endpoint to production:

```bash
modal deploy src/functions/serve.py
```

Get your web endpoint URL:

```bash
modal app show vertical-ai-serve
```

Call it from your application:

```bash
curl -X POST https://your-modal-url/inference_endpoint \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?", "use_finetuned": true}'
```

Modal handles:
- ✅ Auto-scaling based on traffic
- ✅ GPU provisioning and management
- ✅ Cold start optimization (keep_warm)
- ✅ Load balancing
- ✅ Monitoring and logs

## Cost Optimization

**Volumes**: Billed for storage (~$0.15/GB/month for 14GB = ~$2/month)

**Sandboxes**: Billed for CPU time (Active CPU pricing)
- 10 rollouts x 30 seconds x $0.0001/second = $0.03

**Functions**:
- Fine-tuning: ~20 minutes on A100 = ~$1
- Inference: T4 GPU, only pay for request time

**Total cost for this demo**: ~$3-5 to run the complete pipeline once

**Compare to DIY**:
- Setting up GPU servers: Days of work
- Managing Kubernetes: Ongoing ops burden
- Provisioning storage: Infrastructure complexity

Modal's serverless model means you only pay for what you use, with zero ops overhead.

## Extending This Demo

Ideas for taking this further:

- **Add RLHF**: Integrate human feedback into the rollout scoring
- **Scale rollouts**: Try 1,000 or 10,000 concurrent sandboxes
- **Different models**: Swap in Llama 3, Mistral, or other models
- **Different domains**: Finance, legal, medical, code generation
- **Online learning**: Continuously fine-tune as new data arrives
- **A/B testing**: Deploy multiple model versions and compare performance

## Learn More

- **YouTube Video**: [Building Vertical AI Infrastructure with Modal](https://youtube.com/link) _(coming soon)_
- **Blog Post**: [The Vertical AI Stack](https://substack.link) _(coming soon)_
- **Modal Docs**: https://modal.com/docs
- **Modal TypeScript SDK**: https://modal.com/docs/guide/sdk-javascript-go
- **Modal Sandboxes**: https://modal.com/docs/guide/sandboxes
- **Qwen Model**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Troubleshooting

**"Authentication failed"**: Run `modal token new` to authenticate

**"Volume not found"**: Run step 1 first to create the volume and load the model

**"Out of GPU quota"**: Check your Modal account GPU limits at modal.com/settings

**"Python function not found"**: Make sure you ran `modal deploy src/functions/*.py`

**Slow first run**: First model download takes ~10 minutes. Subsequent runs are instant.

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- **Modal** for sponsoring this demo and building amazing infrastructure
- **Alibaba Cloud** for the Qwen model series
- The HuggingFace team for transformers and model hosting

---

**Built with ❤️ to showcase the future of AI infrastructure**

Questions? Issues? Open an issue or reach out on [Twitter](https://twitter.com/yourhandle)
