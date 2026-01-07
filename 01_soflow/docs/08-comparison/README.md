# 📖 Chapter 8: Comparison with Other Methods

<div align="center">

*How SoFlow stacks up against the competition*

</div>

---

## 🏆 The Leaderboard

![Methods Comparison](./images/comparison.svg)

---

## 📊 One-Step Methods Comparison

| Method | Steps | JVP? | CFG? | FID (XL/2) |
|:------:|:-----:|:----:|:----:|:----------:|
| Flow Matching | 20+ | ❌ | ✅ | ~2.5 |
| Consistency | 1-2 | ✅ | ⚠️ | ~4-5 |
| MeanFlow | 1 | ✅ | ✅ | 3.43 |
| **SoFlow** | **1** | **❌** | **✅** | **2.96** |

> 🏆 **SoFlow wins**: Best FID, no JVP, native CFG!

---

## 🆚 SoFlow vs Consistency Models

![SoFlow vs Consistency](./images/soflow-vs-consistency.svg)

### Key Differences

| Aspect | Consistency | SoFlow |
|:------:|:-----------:|:------:|
| **What it learns** | `f(x_t, t) → x₀` | `f(x_t, t, s) → x_s` |
| **CFG** | ⚠️ Difficult | ✅ Natural |
| **Training** | Unstable targets | Stable (stop-grad) |
| **JVP needed** | ✅ Yes | ❌ No |

---

## 🆚 SoFlow vs MeanFlow

| Aspect | MeanFlow | SoFlow |
|:------:|:--------:|:------:|
| **Core idea** | Mean velocity | Solution function |
| **JVP** | 😫 Required | 🚀 Not needed |
| **Training speed** | Slow | Fast |
| **FID** | 3.43 | **2.96** |

### Why JVP Matters

```python
# MeanFlow (with JVP) - SLOW
def train_step_meanflow():
    velocity = model(x, t)
    jvp = compute_jvp(model, x, t, direction)  # 😫 Expensive!
    # ...

# SoFlow (no JVP) - FAST
def train_step_soflow():
    pred1 = model(x_t, t, s)
    with torch.no_grad():
        pred2 = model(x_l, l, s)  # Just forward pass!
    # ...
```

> ⚡ SoFlow is **2-3× faster** per training step!

---

## 📈 FID Results on ImageNet 256×256

### 1-NFE (One-Step Generation)

| Model | MeanFlow | SoFlow | Δ |
|:-----:|:--------:|:------:|:-:|
| **B/2** | 6.17 | **4.85** | -21% |
| **M/2** | 5.01 | **3.73** | -26% |
| **L/2** | 3.84 | **3.20** | -17% |
| **XL/2** | 3.43 | **2.96** | -14% |

> 📉 Lower FID = Better quality

### Multi-Step

| Steps | SoFlow FID |
|:-----:|:----------:|
| 1 | 2.96 |
| 2 | 2.54 |
| 4 | 2.31 |

---

## 💻 Computational Cost

| Method | Forward | Backward | JVP | Total |
|:------:|:-------:|:--------:|:---:|:-----:|
| Flow Matching | 1× | 1× | 0× | 2× |
| Consistency | 2× | 2× | 1× | 5× |
| MeanFlow | 2× | 2× | 1× | 5× |
| **SoFlow** | 2× | 1× | 0× | **3×** |

---

## 🎯 When to Use What?

### ✅ Use SoFlow When:
- Training from scratch (no teacher)
- One-step generation is critical
- CFG is needed
- Memory/compute is limited

### ✅ Use Distillation When:
- High-quality teacher exists
- Willing to pay distillation cost

### ✅ Use Multi-Step When:
- Highest quality needed
- Latency is not critical

---

## 🔑 Key Takeaways

<table>
<tr>
<td>

### 📊 Numbers
- Best 1-NFE FID: **2.96**
- Speedup vs MeanFlow: **2-3×**
- JVP needed: **No!**

</td>
<td>

### 🏆 Winner
SoFlow achieves:
- Best quality
- Fastest training
- Simplest implementation

</td>
</tr>
</table>

---

## 📚 What's Next?

Detailed comparison with Diffusion Models!

<div align="center">

**[← Chapter 7: Architecture](../07-architecture/README.md)** | **[Chapter 9: vs Diffusion →](../09-diffusion/README.md)**

</div>

---

<div align="center">

*Chapter 8 of 9 • [Back to Index](../README.md)*

</div>
