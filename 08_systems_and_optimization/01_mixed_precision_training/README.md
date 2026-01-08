# ⚡ Mixed Precision Training

<div align="center">

![Mixed Precision Training](./images/mixed_precision_hero.svg)

*2× speed, 50% memory with FP16/BF16 — no quality loss*

[![Difficulty](https://img.shields.io/badge/Difficulty-Intermediate-yellow?style=for-the-badge)](#)
[![Impact](https://img.shields.io/badge/Impact-High-red?style=for-the-badge)](#)

</div>

---

## 🎯 Where & Why: Real-World Applications

### Where is Mixed Precision Used?

| Use Case | Impact | Who Uses It |
|----------|--------|-------------|
| 🖼️ **Diffusion Model Training** | Train Stable Diffusion 2× faster | Stability AI, Midjourney |
| 🤖 **LLM Training** | Essential for training 70B+ models | OpenAI, Anthropic, Meta |
| 🎮 **Game AI** | Real-time inference in games | NVIDIA DLSS, game studios |
| 🔬 **Research** | Faster iteration on experiments | Every ML lab |
| ☁️ **Cloud Training** | 50% cost reduction | All cloud ML platforms |

### Why Learn Mixed Precision?

> 💡 **The Reality**: If you're training any modern deep learning model without mixed precision, you're wasting money and time.

**Benefits at a glance:**
- ⚡ **1.5-2× training speed** on tensor core GPUs
- 💾 **50% memory reduction** for activations and gradients
- 🎯 **Zero quality loss** with proper implementation
- 💰 **Direct cost savings** on cloud compute

---

## 📖 Understanding Precision Formats

### Floating Point Formats Compared

| Format | Total Bits | Sign | Exponent | Mantissa | Range | Precision |
|--------|-----------|------|----------|----------|-------|-----------|
| **FP32** | 32 | 1 | 8 | 23 | ±3.4×10³⁸ | ~7 decimal digits |
| **FP16** | 16 | 1 | 5 | 10 | ±65504 | ~3 decimal digits |
| **BF16** | 16 | 1 | 8 | 7 | ±3.4×10³⁸ | ~2 decimal digits |

### Visual Comparison

```
FP32: [1 bit sign][8 bits exponent][23 bits mantissa]
FP16: [1 bit sign][5 bits exponent][10 bits mantissa]
BF16: [1 bit sign][8 bits exponent][7 bits mantissa]
```

### Which to Choose?

| Scenario | Recommendation | Why |
|----------|---------------|-----|
| **NVIDIA Ampere+ (A100, RTX 30/40)** | BF16 | Same range as FP32, no loss scaling needed |
| **NVIDIA Volta/Turing (V100, RTX 20)** | FP16 + Loss Scaling | Hardware optimized for FP16 |
| **TPUs** | BF16 | Native BF16 support |
| **Inference only** | FP16 or INT8 | Maximum speed |

---

## 🧮 The Mathematics

### Why FP16 Can Fail

FP16's limited range causes two problems:

**1. Overflow:** Values > 65504 become `inf`
$$\text{If } |x| > 65504: x_{\text{fp16}} = \pm\infty$$

**2. Underflow:** Small gradients become 0
$$\text{If } |x| < 2^{-24} \approx 6 \times 10^{-8}: x_{\text{fp16}} = 0$$

### Loss Scaling Solution

Scale up the loss before backward pass, scale down gradients after:

$$\mathcal{L}_{\text{scaled}} = s \cdot \mathcal{L}$$

$$\nabla\theta = \frac{1}{s} \cdot \nabla\theta_{\text{scaled}}$$

where $s$ is typically 1024-65536, chosen dynamically.

---

## 🔧 Implementation

### PyTorch AMP (Recommended)

```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for FP16 (not needed for BF16)
scaler = GradScaler()

model = Model().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in mixed precision
    with autocast(dtype=torch.float16):
        output = model(batch)
        loss = criterion(output)
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    
    # Unscale before gradient clipping
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update weights
    scaler.step(optimizer)
    scaler.update()
```

### BF16 (Simpler, No Scaling Needed)

```python
# BF16 on Ampere+ GPUs - no GradScaler needed!
for batch in dataloader:
    optimizer.zero_grad()
    
    with autocast(dtype=torch.bfloat16):
        output = model(batch)
        loss = criterion(output)
    
    loss.backward()
    optimizer.step()
```

### What Stays in FP32?

Certain operations must remain in FP32 for numerical stability:

| Operation | Why FP32? |
|-----------|-----------|
| **Master weights** | Accumulate small gradient updates |
| **Loss computation** | Precision in reduction operations |
| **Softmax** | Exponentials can overflow |
| **LayerNorm/BatchNorm** | Statistics accumulation |
| **Optimizer states** | Adam momentum/variance |

---

## 🎲 Diffusion-Specific Considerations

### Training Loop for Diffusion

```python
class MixedPrecisionDiffusionTrainer:
    def __init__(self, unet, vae, noise_scheduler):
        self.unet = unet
        self.vae = vae  # Often kept in FP32 for quality
        self.scheduler = noise_scheduler
        self.scaler = GradScaler()
    
    def training_step(self, images, conditioning):
        # Encode to latent (can be FP32 for quality)
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * 0.18215
        
        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],))
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # U-Net forward in FP16/BF16
        with autocast(dtype=torch.float16):
            noise_pred = self.unet(noisy_latents, timesteps, conditioning)
            loss = F.mse_loss(noise_pred, noise)
        
        # Backward with scaling
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### VAE Precision Considerations

The VAE encoder/decoder is sensitive to precision:

```python
# Option 1: Keep VAE in FP32
vae = vae.float()

# Option 2: Use FP16 with careful handling
with autocast(dtype=torch.float16):
    latents = vae.encode(images).latent_dist.sample()
# Scale factor helps with FP16 range
latents = latents * 0.18215
```

---

## 📊 Memory Savings Breakdown

For a 1B parameter model:

| Component | FP32 | FP16/BF16 | Savings |
|-----------|------|-----------|---------|
| **Model weights** | 4 GB | 2 GB | 50% |
| **Gradients** | 4 GB | 2 GB | 50% |
| **Activations** | ~8 GB | ~4 GB | 50% |
| **Optimizer (Adam)** | 8 GB | 8 GB* | 0%* |
| **Total** | ~24 GB | ~16 GB | ~33% |

*Optimizer states typically kept in FP32

---

## ⚠️ Common Pitfalls

### 1. Gradient Overflow/Underflow

**Symptom:** Loss becomes NaN or training diverges

**Solution:**
```python
# Use dynamic loss scaling
scaler = GradScaler(
    init_scale=65536.0,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)
```

### 2. Inconsistent Precision

**Symptom:** Slow training despite autocast

**Solution:**
```python
# Ensure inputs are on GPU and correct dtype
images = images.cuda()
# Don't manually cast - let autocast handle it
with autocast(dtype=torch.float16):
    output = model(images)  # Autocast handles casting
```

### 3. Custom Operations

**Symptom:** Custom CUDA kernels fail

**Solution:**
```python
@torch.cuda.amp.custom_fwd(cast_inputs=torch.float16)
def custom_forward(x):
    # Your custom operation
    return result

@torch.cuda.amp.custom_bwd
def custom_backward(grad):
    # Gradient computation
    return grad_input
```

---

## 📈 Benchmarks

### Training Speed (A100 80GB)

| Model | FP32 | FP16 | BF16 | Speedup |
|-------|------|------|------|---------|
| ResNet-50 | 1.0× | 1.8× | 1.7× | ~1.8× |
| ViT-L | 1.0× | 2.1× | 2.0× | ~2× |
| Stable Diffusion U-Net | 1.0× | 1.9× | 1.85× | ~1.9× |
| GPT-2 (1.5B) | 1.0× | 1.7× | 1.65× | ~1.7× |

---

## 📚 Key Equations Summary

| Concept | Formula |
|---------|---------|
| **Loss Scaling** | $\mathcal{L}_{scaled} = s \cdot \mathcal{L}$ |
| **Gradient Unscaling** | $\nabla\theta = \frac{1}{s} \nabla\theta_{scaled}$ |
| **FP16 Range** | $[-65504, 65504]$ |
| **BF16 Range** | $[-3.4 \times 10^{38}, 3.4 \times 10^{38}]$ |

---

## 📖 References

1. **Micikevicius, P., et al.** (2018). "Mixed Precision Training." *ICLR*. [arXiv:1710.03740](https://arxiv.org/abs/1710.03740)

2. **NVIDIA** (2023). "Automatic Mixed Precision." [Documentation](https://pytorch.org/docs/stable/amp.html)

3. **Kalamkar, D., et al.** (2019). "A Study of BFLOAT16 for Deep Learning Training." [arXiv:1905.12322](https://arxiv.org/abs/1905.12322)

---

## ✏️ Exercises

<details>
<summary><b>Exercise 1:</b> Benchmark FP32 vs FP16 vs BF16</summary>

**Task:** Train a small CNN on CIFAR-10 with all three precisions.

**Measure:** Training time, memory usage, final accuracy.

**Expected:** FP16/BF16 should be ~1.5× faster with identical accuracy.
</details>

<details>
<summary><b>Exercise 2:</b> Debug gradient overflow</summary>

**Task:** Create a scenario that causes gradient overflow in FP16.

**Hint:** Very large loss values or learning rates.

**Solution:** Implement dynamic loss scaling.
</details>

---

<div align="center">

**[← Back to Systems & Optimization](../)** | **[Next: Gradient Checkpointing →](../02_gradient_checkpointing/)**

</div>
