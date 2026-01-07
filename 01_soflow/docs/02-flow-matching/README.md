# 📖 Chapter 2: Flow Matching Basics

<div align="center">

*Understanding the foundation that SoFlow builds upon*

</div>

---

## 🌊 What is Flow Matching?

Flow Matching is a framework for **transporting** one distribution to another along a continuous path.

![Flow Matching Concept](./images/flow-matching-concept.svg)

> 🎯 **Goal**: Learn to transform random noise into meaningful data (images!)

---

## 📐 The Math Behind It

### Linear Interpolation

We define a path from data `x₀` to noise `x₁`:

![Interpolation](./images/interpolation.svg)

**The formula is beautifully simple:**

```
x_t = (1 - t) · x₀ + t · x₁
```

| Time | State | Meaning |
|:----:|:-----:|:-------:|
| `t = 0` | `x₀` | Pure data (image) |
| `t = 0.5` | Mix | 50% data + 50% noise |
| `t = 1` | `x₁` | Pure noise |

---

## 🏃 The Velocity Field

At each point on the path, there's a **velocity** — how fast and in what direction we're moving:

```
v(x_t, t) = dx_t/dt = x₁ - x₀
```

> 💡 **Key Insight**: The velocity is constant along each trajectory!

### The ODE

This velocity defines an **Ordinary Differential Equation**:

```
dx/dt = v(x, t)
```

---

## 🎓 Training Flow Matching

### Objective

Learn a neural network `v_θ(x_t, t)` that predicts the velocity:

```python
def train_step(model, x_0):
    # Sample noise
    x_1 = torch.randn_like(x_0)
    t = torch.rand(batch_size)
    
    # Interpolate
    x_t = (1 - t) * x_0 + t * x_1
    
    # True velocity is simple!
    v_true = x_1 - x_0
    
    # Predict and compute loss
    v_pred = model(x_t, t)
    loss = F.mse_loss(v_pred, v_true)
    
    return loss
```

---

## 🎨 Generation: The Slow Part

To generate an image, we must **solve the ODE backward**:

```python
def generate(model, num_steps=50):
    x = torch.randn(shape)  # Start with noise
    dt = -1.0 / num_steps
    
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        v = model(x, t)  # 😫 Forward pass
        x = x + v * dt
    
    return x  # After 50 forward passes...
```

> 🐌 **The Problem**: Each step needs a forward pass!

---

## 📉 Quality vs Speed Trade-off

![Quality vs Steps](./images/quality-steps.svg)

| Steps | Quality | Speed |
|:-----:|:-------:|:-----:|
| 50 | ⭐⭐⭐⭐⭐ | Slow |
| 20 | ⭐⭐⭐⭐ | Medium |
| 10 | ⭐⭐⭐ | Fast |
| **1** | ⭐ (standard) / ⭐⭐⭐⭐ (SoFlow!) | **Instant!** |

---

## 🤔 The Key Question

> *What if we could skip all these steps and directly predict the final image?*

That's exactly what **SoFlow's Solution Function** does!

Instead of learning:
- ❌ `v(x_t, t)` → then solving ODE

We learn:
- ✅ `f(x_t, t, s)` → direct mapping!

---

## 🔑 Key Takeaways

<table>
<tr>
<td>

### 📚 What We Learned
- Flow Matching interpolates data ↔ noise
- Velocity field describes the flow
- ODE solving requires many steps

</td>
<td>

### 🚀 Where We're Going
- Solution function skips the ODE
- One-step generation is possible
- SoFlow makes it work!

</td>
</tr>
</table>

---

## 📚 What's Next?

Ready to see how SoFlow reformulates this problem?

<div align="center">

**[← Chapter 1: Introduction](../01-introduction/README.md)** | **[Chapter 3: Solution Function →](../03-solution-function/README.md)**

</div>

---

<div align="center">

*Chapter 2 of 9 • [Back to Index](../README.md)*

</div>
