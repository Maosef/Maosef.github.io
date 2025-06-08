---
layout: single
title: "An Overview of the LLM Training Pipeline"
date: 2024-04-22
categories: AI
tags: [LLM, Architecture, Training, AI]
classes: wide
---

<style>
.wide {
  max-width: 90%;
  margin: 0 auto;
}
pre {
  font-size: 0.9em;
  line-height: 1.4;
  max-width: 100%;
  overflow-x: auto;
}
table {
  font-size: 0.9em;
  width: 100%;
  margin: 1em 0;
}
</style>

<!-- # Notes on LLMs: Architecture and Training Process -->

Large Language Models (LLMs) are transforming the modern world, in some ways exciting and unsettling. I'm writing a series of posts about them for learning, and as an experiment to explore the productivity boost from using AI. In this post, I'll map out the process of training and deploying LLMs. I'll be using diagrams and code to assist the learning process. I'll start high level and go into depth.

<!-- ## Learning Resources

- Andrej Karpathy has many tutorials.
- Substack: Sebastian 
- The research papers: Llama -->

## High Level LLM Training and Deployment Process

<div class="mermaid">
flowchart LR
    data["Raw Data\nCollection & Processing"]
    pretrain["Foundation\nPre‑training"]
    sft["Supervised\nFine‑Tuning"]
    rlhf["RLHF / Alignment"]
    deploy["Inference\nService"]
    monitor["Monitoring &\nUser Feedback"]

    data --> pretrain --> sft --> rlhf --> deploy --> monitor
    monitor -- signals / new labels --> sft


</div>

The training of LLMs involves several main stages:

| Stage                                   | What happens                                                                                                                                                                         | Key outputs / checkpoints                                                |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| **1. Raw Data Collection & Processing** | Gather large‑scale text, code, and multimodal data ⇒ clean, deduplicate, filter toxic or private content, tokenize, and shard into training files.                                   | Curated, versioned dataset + token statistics                            |
| **2. Foundation Pre‑training**          | Train a base transformer on the full corpus with next‑token prediction (or masked modeling) across thousands of GPU hours; periodically checkpoint and validate perplexity.          | Foundational model checkpoints (billions of parameters)                  |
| **3. Supervised Fine‑Tuning (SFT)**     | Further train the base model on high‑quality, human‑written prompt‑response pairs to teach task formats, instruction following, and chain‑of‑thought style.                          | Instruction‑tuned weights + alignment eval scores                        |
| **4. RLHF / Alignment**                 | Collect preference rankings or comparisons, train a reward model, then optimize the policy with PPO, DPO, or RLAIF to reduce harmful or unhelpful responses and improve UX.          | Aligned model weights; reward‑model checkpoints                          |
| **5. Inference Service**                | Package the final model behind an efficient runtime (vLLM, TGI, TensorRT‑LLM), add batching & KV‑cache, expose streaming endpoints, autoscale in Kubernetes.                         | Production API endpoints, latency/throughput SLOs                        |
| **6. Monitoring & User Feedback**       | Log prompts, completions, costs, safety verdicts, and real‑time metrics; collect thumbs‑up/down, harvest new preference data; trigger rollback or retraining when drift is detected. | Telemetry dashboards, new labels feeding back into SFT / alignment loops |


## Training in depth

A more detailed diagram of the LLM training process:

<div class="mermaid">

flowchart TD
    %% ---------------- Data layer ----------------
    subgraph Data_Collection_and_Processing
        direction TB
        DIngest["Data Ingestion\n(web crawl, docs, code, etc.)"]
        PreProc["Pre‑processing & Tokenisation"]
        DSets["Versioned Dataset Storage"]
        HF["Human Review UI"]
        DIngest --> PreProc --> DSets
        HF -->|filter / label| DSets
    end

    %% ------------- Foundation pre‑training -------------
    subgraph Foundation_Model_Training
        direction TB
        Pretrain["Pre‑training\n(GPU cluster)"]
        Ckpt["Checkpoint Store"]
        Eval0["Validation / Eval"]
        DSets --> Pretrain --> Ckpt
        Pretrain --> Eval0
    end

    %% ------------- Supervised fine‑tuning -------------
    subgraph Supervised_Fine_Tuning
        direction TB
        SFTData["SFT Dataset\n(prompt‑response pairs)"]
        SFT["SFT (GPU cluster)"]
        SFTData --> SFT
        Ckpt --> SFT -->|updated weights| Ckpt
    end

    %% ------------- Reward model & RLHF -------------
    subgraph RLHF_Stage
        direction TB
        RewardData["Preference / ranking data"]
        RM["Reward Model Training"]
        RLHF["RLHF (PPO / DPO / RLAIF)"]
        Ckpt --> RM
        RewardData --> RM --> RLHF
        Ckpt --> RLHF -->|final policy| Ckpt
    end

    %% ------------- Styling -------------
    classDef data fill:#f9f871,stroke:#333,stroke-width:1px,color:#000;
    classDef train fill:#f7b500,stroke:#333,stroke-width:1px,color:#000;
    classDef eval fill:#b5e8ff,stroke:#333,stroke-width:1px,color:#000;
    classDef deploy fill:#caffbf,stroke:#333,stroke-width:1px,color:#000;

    class DIngest,PreProc,DSets,SFTData,RewardData data;
    class Pretrain,SFT,RM,RLHF train;
    class Eval0,Eval1 eval;
    class Deploy,Monitor deploy;

</div>

### 1. Pre-training

| Concept                             | Why it matters                                                                                                                | Typical choices / tips                                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Pre‑training objective**          | The model learns a general‑purpose prior by predicting the next token (causal LM) or filling masks (MLM) across huge corpora. | 99 % of large LLMs today use *causal autoregressive* loss with byte‑pair or sentencepiece tokens.                               |
| **Tokenizer & sequence packing**    | Converts raw text → IDs and assembles fixed‑length training examples without wasting context windows.                         | Train a **BPE**/Unigram tokenizer on the same corpus; use *dynamic sequence packing* so batches are \~99 % full.                |
| **Model backbone**                  | Defines parameter count, attention layout, positional encoding, etc.                                                          | GPT‑style decoder‑only transformer with **FlashAttention 2**, **RoPE** or **ALiBi** positions; optional SwiGLU activations.     |
| **Optimizer & schedule**            | Handles huge batches (>4 M tokens) and learning‑rate stability.                                                               | **AdamW** or **Lion** with β₂ ≈ 0.95, **grad‑clip 1.0**, **linear warm‑up → cosine decay**; BF16 or FP16 + QLoRA for memory.    |


### Basic Training Loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb
from tqdm import tqdm

def train_loop(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    warmup_steps: int = 100,
    max_grad_norm: float = 1.0,
    device: str = "cuda"
):
    """Basic training loop with gradient accumulation and mixed precision."""
    
    # Setup
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Optimizer step with gradient scaling
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Update learning rate
            if step < warmup_steps:
                lr_scale = min(1.0, float(step + 1) / float(warmup_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * learning_rate
            else:
                scheduler.step()
            
            # Log metrics
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            

# Usage example:
if __name__ == "__main__":
    # Initialize model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    # Initialize wandb
    wandb.init(project="llm-training", name="basic-training-loop")
    
    # Create dataloaders (using the chunk_generator from previous example)
    train_ds = chunk_generator(ds, seq_len=4096)
    train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(train_ds, batch_size=1)  # In practice, use a separate validation set
    
    # Train
    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=3,
        learning_rate=2e-4,
        warmup_steps=100
    )
```

<!-- This training loop includes several important features:

1. **Mixed Precision Training**: Uses `torch.cuda.amp` for FP16/BF16 training
2. **Gradient Accumulation**: Built into the loop structure
3. **Learning Rate Scheduling**: Cosine annealing with warmup
4. **Gradient Clipping**: Prevents exploding gradients
5. **Checkpointing**: Saves model state periodically
6. **Metrics Logging**: Uses Weights & Biases for experiment tracking
7. **Progress Tracking**: Uses tqdm for progress bars -->

### More Advanced Training

| Concept                             | Why it matters                                                                                                                | Typical choices / tips                                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Distributed parallelism**         | Spreads the model and data across 100–10 000 GPUs.                                                                            | **Data Parallel + ZeRO Stage 3 (DeepSpeed)** for most cases; add **Tensor & Pipeline Parallel** (Megatron‑LM) for >30 B params. |
| **Gradient accumulation**           | Virtual large batches without exceeding GPU RAM.                                                                              | Accumulate 8–64 micro‑batches before an optimizer step; sync grads only at the step boundary.                                   |
| **Mixed precision & kernel fusion** | Doubles throughput and halves memory.                                                                                         | BF16 + **FlashAttention**, fused RMSNorm, rotary cache priming.                                                                 |
| **Evaluation / early warning**      | Tracks quality and detects divergence.                                                                                        | Perplexity on held‑out shards every N steps; log with WandB / TensorBoard.                                                      |
| **Checkpointing & resumption**      | Protects days of GPU time from crashes; enables later SFT or RLHF.                                                            | Save model+optimizer+LR sched every 500–2 000 steps to S3/GCS; keep last 2 + every power‑of‑2 for time‑travel debugging.        |





<!-- - **Data Collection and Processing**
- **Tokenization**
- Choices for model architecture, optimizer, learning rate scheduling, other training details.
- **Distributed Training**: Training across multiple GPUs/TPUs; data parallelism, model parallelism, pipeline parallelism, ZeRO, etc.
- **Optimizer**: Adam, LR scheduling, gradient checkpointing, etc.
- **Mixed Precision Training**: Using lower precision to speed up training
- See DeepSpeed docs for more details.

### 2. Fine-tuning
- **Supervised Fine-tuning**: includes **Instruction Tuning**
- **Human Preference Learning**: Optimizing model outputs based on human feedback (RLHF, DPO)
- **Reinforcement Learning** for long-context reasoning -->




## Supervised Fine-tuning

| Concept                | Why it matters                                                     | Practical notes                                                                                          |
| ---------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| **Instruction tuning** | Teaches the foundation model to follow tasks and formats.          | Mix narrow task data (e.g. SQL‑gen) with broad instruction sets; keep < 5 % of tokens but strong effect. |
| **LoRA / QLoRA**       | Adapter layers let you fine‑tune multi‑B‑param models on 1–4 GPUs. | Rank = 8–32, α ≈ 16; use 4‑bit GPTQ weights → 16× memory savings.                                        |
| **Data curriculum**    | Over‑fitting to synthetic instructions hurts creativity.           | Interleave human‑written (e.g. ShareGPT) with synthetic (Self‑Instruct) using Temp −1 sampling.          |
| **Loss weighting**     | Certain tasks (e.g. JSON tools) deserve higher weight.             | Group by "source", apply sample‑level weights in the `collate_fn`.                                       |

## RLHF/Alignment

| Concept                          | Role in pipeline                                           | Implementation hints                                                            |
| -------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Reward model (RM)**            | Approximates human preferences from ranked pairs.          | Same backbone as policy; freeze limb norms; use pairwise log‑softmax loss.      |
| **Policy optimisation**          | Improves helpfulness while controlling deviation from SFT. | *PPO* (OpenAI), *DPO* (Kim et al.), *RLAIF* (no RM).                            |
| **KL‑penalty / reference model** | Keeps policy near SFT to avoid mode collapse.              | Calculate KL(p‖p\_ref) token‑wise; β ≈ 0.1–0.3.                                 |
| **Safety tuning**                | Extra pass with refusal data, heuristics, jailbreak tests. | Can be applied as a reward shaping term or small SFT on refusal demonstrations. |


## Serving/Inference

<!-- | Pillar                 | Key idea                                                   | Tools / best practice                                                                      |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Runtime engine**     | Kernels optimised for KV‑cache & batch stitching.          | `vLLM` (Pytorch‑FlashAttn, continuous batching); `TensorRT‑LLM` (CUDA Graphs); `TGI` (HF). |
| **Quantisation & MoE** | Cut memory & cost with minimal quality loss.               | GPTQ, AWQ, SmoothQuant for 4‑bit; vLLM now streams 4‑bit right off disk.                   |
| **Autoscaling**        | Align GPU count with QPS; bursty traffic.                  | KEDA + Prometheus custom metric (`tokens_generated_total`).                                |
| **Security / auth**    | Throttle malicious prompts, enforce rate limits.           | API Gateway (Kong/Envoy) + JWT; prompt‑shield; T\&E Guardrail.                             |
| **Observability**      | Structured logs = queries, latencies, token counts, costs. | OpenTelemetry traces; Loki + Grafana dashboards.                                           | -->

| Pillar                 | Key idea                                                   | Tools / best practice                                                                      |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| **Runtime engine**     | Kernels optimised for KV‑cache & batch stitching.          | `vLLM` (Pytorch‑FlashAttn, continuous batching); `TensorRT‑LLM` (CUDA Graphs); `TGI` (HF). |
| **Quantisation & MoE** | Cut memory & cost with minimal quality loss.               | GPTQ, AWQ, SmoothQuant for 4‑bit; vLLM now streams 4‑bit right off disk.                   |
| **Autoscaling**        | Align GPU count with QPS; bursty traffic.                  | KEDA + Prometheus custom metric (`tokens_generated_total`).                                |
| **Observability**      | Structured logs = queries, latencies, token counts, costs. | OpenTelemetry traces; Loki + Grafana dashboards.                                           |


## Deep Dive into LLM Training

### 1 Data curation at trillion‑token scale

Volume & mixture. Meta's Llama 3[^1] pre‑trained on ≈15.6 trillion text tokens—an order‑of‑magnitude jump over Llama 2 (1.8 T) and similar to other 2025 frontier runs.

Filtering & deduplication. Frontier teams now apply multi‑stage "quality cascades": aggressive near‑duplicate removal, per‑domain quality classifiers, heuristics for adult/hate content, and language balancing to avoid Anglo‑centric bias.

Mixture‑of‑sources. A typical recipe is ≈ 50‑60 % web crawl (CommonCrawl variants), 15–20 % curated corpora (books, papers, code), 10–15 % synthetic model‑generated text, and task‑specialised "gold" data (<1 %) used later for supervised fine‑tuning (SFT).

Packing & prefixing. Token‑level sequence packing (to minimise padding) and metadata prefixing (domain, language, license) are now standard to raise effective throughput by 15‑25 %.

[^1]: https://ar5iv.labs.arxiv.org/html/2407.21783v1

## 3 Model architecture choices



| Generation  | Parameterisation                                      | Context     | Core design                                         | Why it matters                                                                       |
| ----------- | ----------------------------------------------------- | ----------- | --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| **Llama 3** | 8 B / 70 B / 405 B **dense**                          | up to 128 k | classic Transformer + minor rotary‑embedding tweaks | Dense is simpler to train & debug at 16 k GPU scale ([ar5iv][1])                     |
| **Llama 4** | 109 B (SCOUT) / 400 B (MAVERICK) **MoE** (16 experts) | 1 mil       | router + shared expert + SwiGLU blocks              | Activates ≈10 % of params per token—better FLOP ↔ quality trade‑off ([TechTalks][2]) |

[1]: https://ar5iv.org/abs/2407.21783v1 "[2407.21783] The Llama 3 Herd of Models"
[2]: https://bdtechtalks.com/2025/04/06/meta-llama-4/?utm_source=chatgpt.com "What to know about Meta's Llama 4 model family - TechTalks"


## Optimiser & precision recipe
AdamW β1 = 0.9, β2 = 0.95, ε = 1e‑8 remains the default for stability.

LR schedule: 2 % warm‑up → cosine decay to 10 % of peak.

Mixed precision: BF16 for activations & gradients, FP8 (E4M3) for certain matmuls using FlashAttention‑3 kernels, giving 1.3–1.4× speed‑ups on Hopper GPUs .

Gradient clipping at 1.0; weight decay 0.1; dropout only in embeddings for long‑context models.


## Post‑training alignment pipeline
Supervised fine‑tuning (SFT) on curated instruction‑response sets (1–5 M examples).

Rejection sampling to prune low‑quality generations.

Direct Preference Optimisation (DPO)—a KL‑regularised, pairwise‑ranking objective that's simpler and more stable than PPO yet matches RLHF quality
ar5iv
.

Safety adapters like Llama Guard 3 or "red‑team" classifiers are attached as routing layers or post‑decoders.

8 Evaluation & safety gates
Automated evals (MMLU, GSM‑8K, GPQA, CodeEval) every 1–2 B training tokens.

Human preference eval on 2 k–4 k prompts to monitor helpfulness/harmlessness.



## Architecture Overview

There are many existing great posts that explain LLM model architectures. I include some key components here, with a future goal of going into depth:

### 1. Transformer Architecture
- **Self-Attention Mechanism**: Allows the model to weigh the importance of different words in a sequence
- **Multi-Head Attention**: Enables the model to focus on different parts of the sequence simultaneously
- **Feed-Forward Networks**: Process the attended information
- **Layer Normalization**: Helps stabilize training
- **Residual Connections**: Facilitate gradient flow during training

### 2. Model Components
- **Embedding Layer**: Converts input tokens into dense vectors
- **Positional Encoding**: Provides information about the position of tokens in the sequence
- **Decoder/Encoder Blocks**: Process the input through multiple layers of attention and feed-forward networks


## Challenges and Considerations

1. **Computational Resources**
   - Large models require significant computational power
   - Training can take weeks or months on specialized hardware

2. **Data Quality**
   - The quality of training data significantly impacts model performance
   - Careful filtering and preprocessing are essential

3. **Ethical Considerations**
   - Bias in training data
   - Potential for misuse
   - Environmental impact of training large models

## Future Directions

1. **Efficiency Improvements**
   - Model compression techniques
   - More efficient architectures
   - Better training algorithms

2. **Multimodal Capabilities**
   - Integration with vision and audio
   - Cross-modal understanding

3. **Specialized Applications**
   - Domain-specific fine-tuning
   - Customized solutions for specific industries

## Conclusion

Understanding the architecture and training process of LLMs is crucial for both researchers and practitioners in the field of AI. As these models continue to evolve, they present both exciting opportunities and important challenges that need to be addressed.

---

*This post provides a high-level overview of LLM architecture and training. For more detailed information, please refer to the original research papers and technical documentation.* 