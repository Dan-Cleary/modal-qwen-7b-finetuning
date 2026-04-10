# Fine-Tuning Qwen 2.5 7B on Modal

> **Build an open-source demo that shows how to fine-tune an open-source LLM and serve it, all on Modal.**
>
> Take Qwen 2.5 7B, run RL rollouts, fine-tune it on a small customer service dataset on Modal GPU, then serve it and show a before/after comparison — same question, base model vs fine-tuned model.

## What This Demo Does

This is a complete vertical AI pipeline built on Modal infrastructure:

1. **Load** Qwen 2.5 7B (~14GB) to Modal Volume
2. **Evaluate** with parallel RL rollouts 
3. **Fine-tune** on 206 customer service Q&A pairs using LoRA
4. **Serve** both models and show side-by-side comparison

**The payoff**: See how a $0.52 fine-tuning run transforms generic responses into product-specific answers.

## Why This Matters

This is the same infrastructure pattern that companies like Cursor and Intercom use to build specialized AI models. The complete pipeline—from model loading to production serving—runs on Modal with:

- **Modal Volumes**: Persistent storage for 14GB model weights
- **Modal Functions**: Serverless GPU compute (A100 for training, A10G for inference)
- **TypeScript SDK**: Orchestrate the entire pipeline from TypeScript
- **Auto-scaling**: Pay only for actual compute time used

## Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- A Modal account ([sign up here](https://modal.com))
- Modal CLI installed: `pip install modal`

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/Dan-Cleary/modal-qwen-7b-finetuning.git
cd modal-qwen-7b-finetuning
npm install
```

### 2. Set Up Modal

Authenticate with Modal:

```bash
modal token new
```

This opens a browser to authenticate. Your credentials are saved locally.

For the TypeScript SDK, get your token from https://modal.com/settings and add to `.env`:

```bash
cp .env.example .env
# Add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to .env
```

### 3. Deploy Python Functions

Deploy all Modal functions:

```bash
modal deploy src/functions/load_model.py
modal deploy src/functions/finetune.py
modal deploy src/functions/serve.py
```

### 4. Run the Pipeline

Execute in order:

```bash
# Step 1: Load model to Volume (~10 min first time, instant after)
npm run load

# Step 2: RL rollouts (~5-10 min)
npm run rollouts

# Step 3: Fine-tune on A100 GPU (~10 min, costs $0.52)
npm run finetune

# Step 4: Compare base vs fine-tuned models
npm run serve
```

### 5. View the Results

Open `demo.html` in your browser to see the before/after comparison with costs and pipeline details.

## What You'll See

**Before (Base Model)**:
```
Whether you can upgrade your plan mid-cycle often depends on the specific 
service provider or platform you're using. Many providers offer the 
flexibility to upgrade your plan at any time, but there might be certain 
conditions or restrictions...
```

**After (Fine-Tuned Model)**:
```
Yes, you can upgrade at any time. Billing starts immediately and prorates 
based on the remaining days. Downgrades also work this way. No surprises 
or hidden fees.
```

The fine-tuned model learned from 206 customer service examples to give direct, product-specific answers.

## Pipeline Details

### Step 1: Load Model ($1.16)

Downloads Qwen 2.5 7B Instruct (~14GB) from HuggingFace to Modal Volume. First run takes ~10 minutes. Subsequent runs are instant because the model is cached in the Volume.

**Key code** (`src/01_load.ts`):
```typescript
const loadFn = await modal.functions.fromName("vertical-ai-load-model", "download_model");
await loadFn.remote([]);
```

### Step 2: RL Rollouts

Runs 10 parallel inference calls to evaluate base model performance. Demonstrates Modal's auto-scaling: spin up multiple GPU instances, run evaluations, scale back to zero.

**Results**: 8/10 successful completions. High-scoring rollouts added to training data.

### Step 3: Fine-Tune ($0.52)

LoRA fine-tuning on A100 GPU (40GB memory):
- **Training data**: 206 customer service Q&A pairs
- **Epochs**: 10
- **Trainable params**: ~10M (1% of 7B total)
- **Time**: 9.3 minutes
- **Final loss**: 2.78
- **Cost**: $0.52

**Key tech**: Uses LoRA (Low-Rank Adaptation) which only trains 1% of parameters, making it much faster and cheaper than full fine-tuning.

### Step 4: Serve

Deploys both models on A10G GPUs with auto-scaling. Runs the same queries through both models to show the difference.

**Pricing model**: You pay for GPU time, not tokens. Modal charges per second of GPU usage (~$0.018/min on A10G) regardless of token count.

## Project Structure

```
modal-qwen-7b-finetuning/
├── README.md                      # This file
├── package.json                   # TypeScript dependencies
├── demo.html                      # Visual demo of results
├── data/
│   └── dataset.jsonl              # 206 customer service Q&A pairs
├── src/
│   ├── 01_load.ts                 # Load model to Volume
│   ├── 02_rl_rollouts.ts          # Parallel RL rollouts
│   ├── 03_finetune.ts             # Fine-tuning orchestrator
│   ├── 04_serve.ts                # Serve and compare models
│   └── functions/
│       ├── load_model.py          # Modal Function: load model
│       ├── finetune.py            # Modal Function: fine-tune on GPU
│       └── serve.py               # Modal Function: inference serving
└── .env.example                   # Environment template
```

## TypeScript + Python Architecture

This demo uses a hybrid approach:

**TypeScript**: Orchestration, data handling, calling Modal Functions  
**Python**: Modal Functions that need GPU access (load, fine-tune, serve)

Example:
```typescript
// TypeScript orchestration
const finetuneFn = await modal.functions.fromName("vertical-ai-finetune", "finetune_model");
const result = await finetuneFn.remote([datasetContent, rolloutContent, "qwen-2.5-7b-instruct-finetuned", 10, 2e-5]);
```

```python
# Python GPU function
@app.function(gpu="A100", volumes={"/models": model_volume}, timeout=7200)
def finetune_model(dataset_jsonl: str, rollout_results_jsonl: str, output_name: str, epochs: int, lr: float):
    # Runs on A100 GPU, Modal handles provisioning
    model = get_peft_model(model, lora_config)
    trainer.train()
    model.save_pretrained(f"/models/{output_name}")
```

## Production Deployment

The fine-tuned model is already deployed as a web endpoint:

```bash
curl -X POST https://dancleary54--vertical-ai-serve-inference-endpoint.modal.run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "How do I reset my password?", "use_finetuned": true}'
```

Modal handles auto-scaling, GPU provisioning, and monitoring. With the current setup (no warm containers), the service scales to zero when idle and only costs money during actual inference calls.

## Cost Breakdown

**One complete pipeline run**:
- Model loading: $1.16
- RL rollouts: ~$0.10
- Fine-tuning (A100, 9.3 min): $0.52
- **Total**: ~$1.78

**Inference serving**:
- Pay per GPU second (~$0.018/min on A10G)
- Each call: 60-90 seconds (includes cold start)
- Cost per call: ~$0.02-0.03

**Optional: Keep warm**:
- $1.10/hour per GPU to eliminate cold starts
- Makes calls faster but adds continuous cost

## Extending This Demo

**Increase training data**: Add more customer service examples to improve results

**Try different models**: Swap in Llama 3, Mistral, Phi-3

**Scale rollouts**: Run 1,000+ parallel evaluations

**Use vLLM for production**: Swap transformers for vLLM to get 20-50x throughput per GPU

**Add quantization**: Use INT8/INT4 to serve smaller, faster models

**Different domains**: Finance, legal, medical, code generation

## Modal Primitives Used

### 1. Modal Volumes
Persistent distributed storage shared across all functions:
```python
model_volume = modal.Volume.from_name("qwen-model-cache", create_if_missing=True)

@app.function(volumes={"/models": model_volume})
def load_model():
    model.save_pretrained("/models/qwen-2.5-7b-instruct")
    model_volume.commit()  # Persist changes
```

**Why it matters**: Download the 14GB model once, reuse it across every pipeline step. No repeated downloads.

### 2. Modal Functions
Serverless GPU functions that auto-scale:
```python
@app.function(gpu="A100", image=image, volumes={"/models": model_volume})
def finetune_model(dataset_jsonl: str):
    # Modal provisions A100, loads model from Volume, runs training
    trainer.train()
```

**Why it matters**: Get A100/A10G GPUs on demand. No infrastructure management. Pay only for actual compute time.

### 3. Auto-scaling
Modal automatically scales based on load:
- 1 request → 1 GPU spins up
- 10 concurrent requests → 10 GPUs spin up
- No requests → scales to zero

**Why it matters**: Handle traffic spikes without provisioning capacity. Only pay when actually processing requests.

## Learn More

- **Modal Docs**: https://modal.com/docs
- **Modal TypeScript SDK**: https://modal.com/docs/guide/sdk-javascript-go
- **Qwen 2.5 7B Model**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- **LoRA Fine-tuning**: https://arxiv.org/abs/2106.09685

## Troubleshooting

**"Authentication failed"**: Run `modal token new` to authenticate

**"Volume not found"**: Run `npm run load` first to create the volume

**"Function not found"**: Deploy functions with `modal deploy src/functions/*.py`

**Timeouts on serve step**: Normal for first cold start. Model loading takes 30-60 seconds.

**Out of GPU quota**: Check your Modal plan at modal.com/settings

## Code Philosophy

This demo follows one principle: **Clarity over cleverness.**

- Clean, well-commented code
- TypeScript where possible for type safety
- Python only for GPU operations
- Easy to clone and run
- No magic, no abstractions

The code is meant to be read, understood, and modified.

## License

MIT License - see [LICENSE](LICENSE) for details

## Acknowledgments

- **Modal** for sponsoring this demo and building infrastructure that makes vertical AI accessible
- **Alibaba Cloud** for the Qwen model series
- **HuggingFace** for transformers and model hosting

---

**Questions?** Open an issue or check out the [demo page](demo.html)

**Sponsored by Modal** → [modal.com/dancleary](https://modal.com/dancleary)
