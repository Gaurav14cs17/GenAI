# 🚀 SoFlow: One-Step Image Generation

<div align="center">

![SoFlow Overview](./overview.svg)

**Solution Flow Models for One-Step Generative Modeling**

[![Paper](https://img.shields.io/badge/arXiv-2512.15657-b31b1b.svg)](https://arxiv.org/pdf/2512.15657)
[![GitHub](https://img.shields.io/badge/GitHub-Gaurav14cs17%2FGenAI-black.svg)](https://github.com/Gaurav14cs17/GenAI/tree/main/01_soflow)

*Generate high-quality images in just ONE step — no more waiting for 1000 denoising iterations!*

</div>

---

## 🎯 TL;DR

> **SoFlow** revolutionizes image generation by learning the **solution function** instead of velocity, enabling **one-step generation** with state-of-the-art quality — achieving **FID 2.96** on ImageNet 256×256 with a single forward pass!

---

## 📚 Documentation Chapters

| Chapter | Topic | What You'll Learn |
|:-------:|-------|-------------------|
| [📖 Chapter 1](./01-introduction/README.md) | **Introduction** | Why one-step generation matters, the efficiency problem |
| [📖 Chapter 2](./02-flow-matching/README.md) | **Flow Matching** | Velocity fields, ODEs, the foundation of SoFlow |
| [📖 Chapter 3](./03-solution-function/README.md) | **Solution Function** | The key innovation: f(x_t, t, s) = x_s |
| [📖 Chapter 4](./04-training/README.md) | **Training** | Two losses: Flow Matching + Consistency |
| [📖 Chapter 5](./05-proofs/README.md) | **Math Proofs** | Theoretical guarantees and error bounds |
| [📖 Chapter 6](./06-cfg/README.md) | **CFG** | Classifier-Free Guidance integration |
| [📖 Chapter 7](./07-architecture/README.md) | **Architecture** | DiT model with solution function |
| [📖 Chapter 8](./08-comparison/README.md) | **Comparisons** | vs Consistency Models, MeanFlow |
| [📖 Chapter 9](./09-diffusion/README.md) | **vs Diffusion** | Step-by-step comparison with DDPM |

---

## 💡 The Core Idea

### ❌ The Old Way (Slow)
```
Diffusion/Flow Matching: 
  Learn velocity v(x,t) → Solve ODE with 50-1000 steps → Image
  
  [Noise] → step → step → step → ... → step → [Image]
                    (50-1000 iterations!)
```

### ✅ The SoFlow Way (Fast)
```
SoFlow:
  Learn solution f(x,t,s) → Direct mapping → Image
  
  [Noise] ─────────── ONE STEP ───────────→ [Image]
```

---

## 📊 Results at a Glance

### ImageNet 256×256 (1-NFE)

| Model | MeanFlow | SoFlow | Improvement |
|:-----:|:--------:|:------:|:-----------:|
| **B/2** | 6.17 | **4.85** | 🔻 21% |
| **M/2** | 5.01 | **3.73** | 🔻 26% |
| **L/2** | 3.84 | **3.20** | 🔻 17% |
| **XL/2** | 3.43 | **2.96** | 🔻 14% |

> 💡 **Lower FID = Better Quality** — SoFlow beats all one-step methods!

---

## 🔑 Key Innovations

<table>
<tr>
<td width="33%" align="center">

### 🎯 Solution Function
Learn `f(x_t, t, s) = x_s` directly instead of velocity

</td>
<td width="33%" align="center">

### ⚡ No JVP Needed
Unlike competitors, no expensive Jacobian-Vector Products

</td>
<td width="33%" align="center">

### 🎨 Native CFG
Classifier-Free Guidance works naturally

</td>
</tr>
</table>

---

## 🧮 The Math (Simplified)

### Training Objective
```
L_total = L_FM + λ · L_cons
```

| Loss | Formula | Purpose |
|------|---------|---------|
| **L_FM** | `‖f_θ(x_t, t, 0) − x₀‖²` | Learn to denoise |
| **L_cons** | `‖f_θ(x_t, t, s) − sg[f_θ(x_l, l, s)]‖²` | Self-consistency |

### One-Step Generation
```python
# That's it! Just one line:
image = model(noise, t=1, s=0)
```

---

## 🏗️ Project Structure

```
01_soflow/
├── 📚 docs/
│   ├── 📄 README.md              ← You are here!
│   ├── 🖼️ overview.svg
│   │
│   ├── 📁 01-introduction/       ← Start here!
│   ├── 📁 02-flow-matching/
│   ├── 📁 03-solution-function/
│   ├── 📁 04-training/
│   ├── 📁 05-proofs/
│   ├── 📁 06-cfg/
│   ├── 📁 07-architecture/
│   ├── 📁 08-comparison/
│   └── 📁 09-diffusion/
├── 📦 soflow/                    # Main package
├── 🔧 scripts/                   # Training & sampling scripts
├── 📓 notebooks/                 # Colab notebooks
└── ⚙️ configs/                   # Configuration files
```

---

## 🚀 Quick Start Reading

1. **New to generative models?** → Start with [Chapter 1: Introduction](./01-introduction/README.md)
2. **Know diffusion but want speed?** → Jump to [Chapter 9: vs Diffusion](./09-diffusion/README.md)
3. **Want the math?** → See [Chapter 5: Proofs](./05-proofs/README.md)
4. **Ready to implement?** → Check [Chapter 7: Architecture](./07-architecture/README.md)

---

## 📓 Google Colab Notebooks

Run SoFlow directly in your browser — no setup required!

| Notebook | Description | Link |
|:--------:|-------------|:----:|
| **Training** | Train SoFlow on CIFAR-10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/GenAI/blob/main/01_soflow/notebooks/SoFlow_Training.ipynb) |
| **Inference** | Generate images with trained model | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/GenAI/blob/main/01_soflow/notebooks/SoFlow_Inference.ipynb) |

---

## 📖 Citation

```bibtex
@article{luo2024soflow,
  title={SoFlow: Solution Flow Models for One-Step Generative Modeling},
  author={Luo, Tianze and Yuan, Haotian and Liu, Zhuang},
  journal={arXiv preprint arXiv:2512.15657},
  year={2024}
}
```

---

<div align="center">

**[📖 Start Reading →](./01-introduction/README.md)**

*Made with ❤️ by [Gaurav](https://github.com/Gaurav14cs17) for the ML community*

</div>
