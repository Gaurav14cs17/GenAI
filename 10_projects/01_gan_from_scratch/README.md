# 🛠️ Project: GAN from Scratch

<div align="center">

![GAN from Scratch](./images/gan_project_hero.svg)

*Build your first adversarial training pipeline*

[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner-green?style=for-the-badge)](#)
[![Time](https://img.shields.io/badge/Time-1--2%20Weeks-blue?style=for-the-badge)](#)

</div>

---

## 🎯 Where & Why: Learning Objectives

### Why Build a GAN from Scratch?

| What You'll Learn | Why It Matters |
|-------------------|----------------|
| 🎲 **Adversarial training** | Foundation for understanding all GAN variants |
| ⚖️ **Training stability** | Core skill for training any generative model |
| 📊 **Evaluation metrics** | FID, IS — used everywhere in generative AI |
| 🏗️ **Architecture design** | Generator/discriminator patterns reused in diffusion |
| 🐛 **Debugging skills** | Mode collapse, gradient issues — common problems |

### Real-World Skills Gained

After this project, you'll be able to:
- ✅ Train adversarial models from scratch
- ✅ Diagnose and fix training instabilities
- ✅ Evaluate generative model quality
- ✅ Implement modern architectural patterns

---

## 📖 Project Overview

### Goal

Build a complete GAN training pipeline that generates recognizable images on MNIST and CIFAR-10.

### Milestones

| # | Milestone | Description | Success Criteria |
|---|-----------|-------------|------------------|
| 1 | **Basic GAN** | Simple FC generator/discriminator | Loss converges |
| 2 | **DCGAN** | Convolutional architecture | Blurry but recognizable digits |
| 3 | **Training Tricks** | Labels, learning rates, architecture | Stable training curves |
| 4 | **CIFAR-10** | Scale to color images | Recognizable objects |
| 5 | **Evaluation** | FID and Inception Score | FID < 50 |

---

## 🧮 Mathematical Background

### The GAN Objective

**Minimax Game:**
$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

**Intuition:**
- $D$ wants to classify real as 1, fake as 0 → maximize $V$
- $G$ wants $D$ to classify fake as real → minimize $V$

### Alternative Losses

**Non-Saturating Generator Loss:**
$$\mathcal{L}_G = -\mathbb{E}_{z}[\log D(G(z))]$$

**Heuristic:** Instead of minimizing $\log(1 - D(G(z)))$, maximize $\log D(G(z))$. Provides stronger gradients early in training.

### At Nash Equilibrium

When training converges:
- $p_G = p_{data}$ (generator matches data distribution)
- $D(x) = 0.5$ for all $x$ (discriminator can't distinguish)

---

## 🏗️ Architecture Guide

### Milestone 1: Basic GAN (MLP)

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
```

### Milestone 2: DCGAN (Convolutional)

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # latent_dim -> features*8 x 4 x 4
            nn.ConvTranspose2d(latent_dim, features*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),
            
            # -> features*4 x 8 x 8
            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),
            
            # -> features*2 x 16 x 16
            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),
            
            # -> features x 32 x 32
            nn.ConvTranspose2d(features*2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            
            # -> channels x 64 x 64
            nn.ConvTranspose2d(features, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z.view(-1, z.size(1), 1, 1))

class DCGANDiscriminator(nn.Module):
    def __init__(self, channels=1, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # channels x 64 x 64 -> features x 32 x 32
            nn.Conv2d(channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> features*2 x 16 x 16
            nn.Conv2d(features, features*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> features*4 x 8 x 8
            nn.Conv2d(features*2, features*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> features*8 x 4 x 4
            nn.Conv2d(features*4, features*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> 1 x 1 x 1
            nn.Conv2d(features*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1)
```

---

## 🔧 Training Loop

```python
def train_step(G, D, real_images, opt_G, opt_D, criterion):
    batch_size = real_images.size(0)
    device = real_images.device
    
    # Labels
    real_labels = torch.ones(batch_size, 1, device=device)
    fake_labels = torch.zeros(batch_size, 1, device=device)
    
    # ============ Train Discriminator ============
    opt_D.zero_grad()
    
    # Real images
    real_pred = D(real_images)
    loss_real = criterion(real_pred, real_labels)
    
    # Fake images
    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = G(z)
    fake_pred = D(fake_images.detach())  # Detach to avoid training G
    loss_fake = criterion(fake_pred, fake_labels)
    
    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    opt_D.step()
    
    # ============ Train Generator ============
    opt_G.zero_grad()
    
    # Generate new fakes (or reuse, but need fresh D output)
    fake_pred = D(fake_images)
    loss_G = criterion(fake_pred, real_labels)  # Want D to think fake is real
    
    loss_G.backward()
    opt_G.step()
    
    return loss_D.item(), loss_G.item()
```

---

## 💡 Training Tips (Milestone 3)

### 1. Label Smoothing

```python
# Instead of 1.0 and 0.0
real_labels = torch.ones(batch_size, 1) * 0.9  # Smooth to 0.9
fake_labels = torch.zeros(batch_size, 1) + 0.1  # Smooth to 0.1
```

### 2. Learning Rate Balance

```python
# D often learns faster, so use lower LR
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
```

### 3. Train D More Often (if needed)

```python
for i, (real_images, _) in enumerate(dataloader):
    # Train D every step
    loss_D = train_discriminator(...)
    
    # Train G every n_critic steps
    if i % n_critic == 0:
        loss_G = train_generator(...)
```

### 4. Spectral Normalization

```python
from torch.nn.utils import spectral_norm

# Apply to discriminator layers
nn.Conv2d(...) → spectral_norm(nn.Conv2d(...))
```

---

## 📊 Evaluation

### FID Score (Fréchet Inception Distance)

```python
from pytorch_fid import fid_score

def compute_fid(real_path, fake_path):
    """Lower is better. Good GAN: FID < 50"""
    return fid_score.calculate_fid_given_paths(
        [real_path, fake_path],
        batch_size=50,
        device='cuda',
        dims=2048
    )
```

### Inception Score

```python
from torchmetrics.image.inception import InceptionScore

is_metric = InceptionScore()
is_metric.update(generated_images)
is_mean, is_std = is_metric.compute()
# Higher is better. Good: IS > 5 on CIFAR-10
```

---

## 🐛 Common Problems & Solutions

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Mode collapse** | All outputs look same | Minibatch discrimination, unrolled GAN |
| **D too strong** | G loss doesn't decrease | Lower D learning rate, label smoothing |
| **Training oscillates** | Losses fluctuate wildly | Reduce learning rates, add noise to D inputs |
| **Checkerboard artifacts** | Grid patterns in output | Use resize-conv instead of transpose conv |

---

## 📁 Project Structure

```
gan_from_scratch/
├── models/
│   ├── __init__.py
│   ├── generator.py
│   ├── discriminator.py
│   └── dcgan.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── losses.py
├── evaluation/
│   ├── __init__.py
│   ├── fid.py
│   └── visualize.py
├── configs/
│   └── config.yaml
├── notebooks/
│   └── exploration.ipynb
├── train.py
├── generate.py
└── requirements.txt
```

---

## ✅ Submission Checklist

- [ ] Generator produces recognizable images
- [ ] Training is stable (no mode collapse)
- [ ] FID score improves over training
- [ ] Loss curves logged and visualized
- [ ] Sample grids saved at checkpoints
- [ ] Code is clean and documented
- [ ] README explains how to run

---

## 🚀 Extensions (Optional)

1. **WGAN-GP**: Replace BCE loss with Wasserstein loss + gradient penalty
2. **Conditional GAN**: Add class conditioning
3. **Progressive Growing**: Start small, add layers during training
4. **Self-Attention GAN**: Add attention layers for global coherence

---

## 📖 References

1. **Goodfellow, I., et al.** (2014). "Generative Adversarial Networks." *NeurIPS*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

2. **Radford, A., et al.** (2016). "Unsupervised Representation Learning with DCGANs." *ICLR*. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

3. **Salimans, T., et al.** (2016). "Improved Techniques for Training GANs." *NeurIPS*. [arXiv:1606.03498](https://arxiv.org/abs/1606.03498)

---

<div align="center">

**[← Back to Projects](../)** | **[Next: Diffusion from Scratch →](../02_diffusion_from_scratch/)**

</div>
