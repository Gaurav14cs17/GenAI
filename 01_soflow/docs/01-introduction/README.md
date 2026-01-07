# 📖 Chapter 1: Introduction to SoFlow

<div align="center">

*Why do we need one-step generation? What problem does SoFlow solve?*

</div>

---

## 🎯 The Problem: Slow Generation

Modern generative models produce **stunning images**, but there's a catch...

![Efficiency Problem](./images/efficiency-problem.svg)

> ⏱️ **The Bottleneck**: Generating a single image requires **50-1000 forward passes** through a neural network. That's painfully slow!

---

## 🤔 Why So Many Steps?

Traditional approaches (Diffusion, Flow Matching) learn a **velocity function**:

```
dx/dt = v(x_t, t)
```

To generate an image, you must **numerically solve this ODE**:

```python
# 😫 The slow way
x = noise
for t in [1.0, 0.99, 0.98, ..., 0.01, 0.0]:  # Many steps!
    v = model(x, t)
    x = x + v * dt
# Finally! An image... after 100 forward passes
```

> 🐌 Each step = One full model forward pass = Expensive!

---

## 💡 SoFlow's Insight

### What if we skip the ODE entirely?

Instead of learning *how to move* (velocity), learn *where to end up* (solution):

| Approach | Learns | Generation |
|:--------:|:------:|:----------:|
| **Flow Matching** | `v(x_t, t)` velocity | Solve ODE (many steps) |
| **SoFlow** | `f(x_t, t, s)` solution | Direct! (one step) |

---

## ✨ The Solution Function

The **solution function** `f(x_t, t, s)` tells us:

> "Given state `x_t` at time `t`, what's the state at time `s`?"

```
f(x_t, t, s) = x_s
```

### One-Step Generation

For generating images, we just need:

```python
# 🚀 The SoFlow way
noise = torch.randn(batch_size, 3, 256, 256)
image = model(noise, t=1, s=0)  # ONE step! That's it!
```

> 🎉 **1000× speedup** compared to DDPM!

---

## 🏆 Why SoFlow Matters

<table>
<tr>
<td width="50%">

### ❌ Problems with Other Approaches

- **Diffusion**: 1000 steps, slow
- **DDIM**: Still 50+ steps
- **Consistency Models**: Need JVP (expensive)
- **MeanFlow**: Also needs JVP

</td>
<td width="50%">

### ✅ SoFlow Advantages

- ⚡ **1 step** generation
- 🚫 **No JVP** computation
- 🎨 **Native CFG** support
- 📈 **Best FID** scores

</td>
</tr>
</table>

---

## 📊 Performance Comparison

| Model | MeanFlow FID | SoFlow FID | Winner |
|:-----:|:-----------:|:----------:|:------:|
| B/2 | 6.17 | **4.85** | 🏆 SoFlow |
| M/2 | 5.01 | **3.73** | 🏆 SoFlow |
| L/2 | 3.84 | **3.20** | 🏆 SoFlow |
| XL/2 | 3.43 | **2.96** | 🏆 SoFlow |

> 📉 Lower FID = Better quality images

---

## 🗺️ Paper Overview

![Paper Structure](./images/paper-structure.svg)

The SoFlow framework combines two training objectives:
1. **Flow Matching Loss** — Learn correct denoising
2. **Consistency Loss** — Ensure self-consistency

Together, they train a model that generates in **one step**!

---

## 📚 What's Next?

Now that you understand *why* SoFlow exists, let's learn *how* it works:

<div align="center">

**[→ Chapter 2: Flow Matching Basics](../02-flow-matching/README.md)**

</div>

---

<div align="center">

*Chapter 1 of 9 • [Back to Index](../README.md)*

</div>
