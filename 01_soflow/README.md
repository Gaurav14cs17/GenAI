# 🚀 SoFlow: Solution Flow Models for One-Step Generative Modeling

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-2512.15657-b31b1b.svg)](https://arxiv.org/abs/2512.15657)
[![GitHub](https://img.shields.io/badge/GitHub-Gaurav14cs17%2FGenAI-black.svg)](https://github.com/Gaurav14cs17/GenAI)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**One-Step Image Generation | No JVP | State-of-the-Art FID**

</div>

---

## 📓 Quick Start with Google Colab

Run SoFlow instantly in your browser — no setup required!

| Notebook | Description | Open |
|:--------:|-------------|:----:|
| **🎓 Training** | Train SoFlow on CIFAR-10 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/GenAI/blob/main/notebooks/SoFlow_Training.ipynb) |
| **🎨 Inference** | Generate images (one-step!) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/GenAI/blob/main/notebooks/SoFlow_Inference.ipynb) |

---

## 📖 Overview

PyTorch implementation of **Solution Flow Models (SoFlow)** from the paper:

> **SoFlow: Solution Flow Models for One-Step Generative Modeling**  
> Tianze Luo, Haotian Yuan, Zhuang Liu  
> Princeton University  
> [arXiv:2512.15657](https://arxiv.org/abs/2512.15657)

### ✨ Key Features

- ⚡ **One-step generation**: Generate high-quality samples in a single forward pass (1-NFE)
- 🚫 **No JVP computation**: Unlike recent works, our consistency loss doesn't require Jacobian-vector products
- 🎨 **CFG support**: Natural integration of Classifier-Free Guidance during training
- 🏆 **State-of-the-art**: Achieves better FID-50K scores than MeanFlow on ImageNet 256×256

---

## 🎯 Method

The core idea is to learn a **solution function** `f_θ(x_t, t, s)` that maps a state `x_t` at time `t` to its evolved state `x_s` at time `s`.

```
┌─────────────────────────────────────────────────────────┐
│  Standard Flow Matching: Many steps                     │
│  [Noise] → step → step → ... → step → [Image]          │
│                    (50-1000 iterations)                 │
├─────────────────────────────────────────────────────────┤
│  SoFlow: ONE step!                                      │
│  [Noise] ───────── f(x, 1, 0) ─────────→ [Image]       │
└─────────────────────────────────────────────────────────┘
```

### Training Objectives

1. **Flow Matching Loss**: Enables the model to provide estimated velocity fields for CFG
2. **Solution Consistency Loss**: Ensures the model learns a valid solution function without expensive JVP calculations

---

## 🛠️ Installation

```bash
git clone https://github.com/Gaurav14cs17/GenAI.git
cd GenAI
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
GenAI/
├── 📓 notebooks/              # Colab notebooks
│   ├── SoFlow_Training.ipynb
│   └── SoFlow_Inference.ipynb
├── 📚 docs/                   # Documentation
│   ├── README.md              # Documentation index
│   ├── 01-introduction/
│   ├── 02-flow-matching/
│   ├── ... (9 chapters)
│   └── 09-diffusion/
├── ⚙️ configs/                # Hydra configuration files
│   ├── config.yaml
│   ├── model/
│   └── training/
├── 📦 soflow/                 # Main package
│   ├── models/
│   │   ├── dit.py             # Diffusion Transformer
│   │   ├── soflow.py          # SoFlow wrapper
│   │   └── layers.py          # Custom layers
│   ├── losses/
│   │   ├── flow_matching.py
│   │   ├── consistency.py
│   │   └── combined.py
│   ├── utils/
│   │   ├── scheduler.py
│   │   ├── ema.py
│   │   └── visualization.py
│   └── data/
│       ├── cifar10.py
│       └── imagenet.py
├── 🔧 scripts/
│   ├── train.py
│   ├── sample.py
│   └── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🎓 Training

### CIFAR-10 (Quick Demo)

```bash
python scripts/train.py --epochs 50 --batch_size 128
```

### ImageNet 256×256

```bash
# Train DiT-B/2 model
python scripts/train.py model=dit_b training=imagenet256

# Train DiT-XL/2 model (multi-GPU)
accelerate launch --multi_gpu scripts/train.py model=dit_xl training=imagenet256
```

---

## 🎨 Sampling

```bash
# Generate samples with CFG
python scripts/sample.py \
    --checkpoint /path/to/checkpoint.pt \
    --num_samples 50000 \
    --cfg_scale 1.5 \
    --output_dir ./samples
```

---

## 📊 Results

### ImageNet 256×256 (1-NFE FID-50K)

| Model | MeanFlow | SoFlow | Improvement |
|:-----:|:--------:|:------:|:-----------:|
| B/2 | 6.17 | **4.85** | 🔻 21% |
| M/2 | 5.01 | **3.73** | 🔻 26% |
| L/2 | 3.84 | **3.20** | 🔻 17% |
| XL/2 | 3.43 | **2.96** | 🔻 14% |

---

## 📚 Documentation

Comprehensive documentation with visualizations and mathematical explanations:

👉 **[Read the Full Documentation](./docs/README.md)**

| Chapter | Topic |
|:-------:|-------|
| 1 | Introduction & Motivation |
| 2 | Flow Matching Basics |
| 3 | Solution Function |
| 4 | Training Objectives |
| 5 | Mathematical Proofs |
| 6 | Classifier-Free Guidance |
| 7 | Model Architecture |
| 8 | Comparison with Other Methods |
| 9 | vs Diffusion Models |

---

## 📜 Citation

```bibtex
@article{luo2024soflow,
  title={SoFlow: Solution Flow Models for One-Step Generative Modeling},
  author={Luo, Tianze and Yuan, Haotian and Liu, Zhuang},
  journal={arXiv preprint arXiv:2512.15657},
  year={2024}
}
```

---

## 🙏 Acknowledgments

This implementation builds upon:
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformer architecture
- [Flow Matching](https://github.com/atong01/conditional-flow-matching) - Flow Matching framework

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**Made with ❤️ by [Gaurav](https://github.com/Gaurav14cs17)**

</div>
