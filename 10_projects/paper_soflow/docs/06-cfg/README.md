# 📖 Chapter 6: Classifier-Free Guidance

<div align="center">

*How SoFlow enables CFG without expensive JVP computation*

</div>

---

## 🎨 What is Classifier-Free Guidance (CFG)?

**Classifier-Free Guidance** is a technique to improve generation quality by amplifying class-conditional signals:

![CFG Concept](./images/cfg-concept.svg)

> 💡 CFG interpolates between unconditional and conditional generation, with the conditional direction amplified.

---

## 📐 Mathematical Formulation

### Standard CFG for Velocity Fields

Given:
- Conditional velocity: $v(x_t, t, y)$ — generation guided by class $y$
- Unconditional velocity: $v(x_t, t, \varnothing)$ — unguided generation

The **guided velocity** is:

$$\tilde{v}(x_t, t, y) = v(x_t, t, \varnothing) + w \cdot \left(v(x_t, t, y) - v(x_t, t, \varnothing)\right)$$

where $w \geq 1$ is the **guidance scale**.

### Simplified Form

$$\tilde{v} = (1 - w) \cdot v_{uncond} + w \cdot v_{cond}$$

| $w$ | Effect |
|:---:|:-------|
| 1.0 | No guidance (pure conditional) |
| 1.5 | Light guidance |
| 2.0 | Standard guidance |
| 4.0+ | Strong guidance |

### Interpretation

$$\tilde{v} = v_{uncond} + w \cdot \underbrace{(v_{cond} - v_{uncond})}_{\text{class direction}}$$

CFG **amplifies the direction** that points toward the target class!

---

## 😫 The Problem with One-Step Models

### Standard Consistency Models

Consistency models learn a direct mapping:

$$F_\theta: x_t \mapsto x_0$$

**Problem**: No velocity field, so standard CFG doesn't apply directly.

```python
# Consistency Model - no velocity!
image = model(noise)  # Can't apply CFG naturally
```

### Workarounds (Suboptimal)

1. **Separate models**: Train conditional and unconditional models
2. **Post-hoc mixing**: Blend outputs (artifacts)
3. **Distillation**: Requires a teacher with CFG (expensive)

---

## ✨ SoFlow's Elegant Solution

### Key Insight: Velocity Extraction

From the solution function, we can **extract velocity** using the ODE property:

$$v_\theta(x_t, t) = \frac{\partial f_\theta(x_t, t, s)}{\partial s} \bigg|_{s=t}$$

Using finite differences:

$$v_\theta(x_t, t) \approx \frac{f_\theta(x_t, t, t-\epsilon) - x_t}{-\epsilon}$$

### Now CFG Works!

With extracted velocity, standard CFG applies:

$$\tilde{v} = v_\theta(x_t, t, \varnothing) + w \cdot \left(v_\theta(x_t, t, y) - v_\theta(x_t, t, \varnothing)\right)$$

---

## 📐 CFG for Solution Functions

### Direct CFG on Outputs

For one-step generation, we can apply CFG directly to solution function outputs:

$$\tilde{f}(x_t, t, s, y) = f_\theta(x_t, t, s, \varnothing) + w \cdot \left(f_\theta(x_t, t, s, y) - f_\theta(x_t, t, s, \varnothing)\right)$$

### Mathematical Justification

For linear interpolation paths, the solution function and velocity are related:

$$f(x_t, t, 0) = x_t + \int_t^0 v(x_\tau, \tau) \, d\tau$$

If we apply CFG to velocity at each step:

$$\tilde{x}_0 = x_t + \int_t^0 \tilde{v}(x_\tau, \tau) \, d\tau$$

For one-step (where $x_t = x_1$), this becomes:

$$\tilde{x}_0 = f(x_1, 1, 0, \varnothing) + w \cdot \left(f(x_1, 1, 0, y) - f(x_1, 1, 0, \varnothing)\right)$$

---

## 💻 Implementation

### Velocity Extraction

```python
def extract_velocity(model, x_t, t, y, eps=1e-4):
    """
    Extract velocity from solution function.
    
    v(x,t) = ∂f(x,t,s)/∂s |_{s=t}
    
    Approximated via finite difference.
    """
    s = t - eps
    f_out = model(x_t, t, s, y)
    velocity = (f_out - x_t) / (-eps)
    return velocity
```

### CFG with Velocity

```python
def cfg_velocity(model, x_t, t, y, cfg_scale=2.0, eps=1e-4):
    """
    Compute CFG-guided velocity.
    """
    # Conditional velocity
    v_cond = extract_velocity(model, x_t, t, y, eps)
    
    # Unconditional velocity (y=None for null class)
    v_uncond = extract_velocity(model, x_t, t, y=None, eps)
    
    # CFG combination
    v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
    
    return v_cfg
```

### One-Step Sampling with CFG

```python
def sample_with_cfg(model, noise, y, cfg_scale=2.0):
    """
    One-step generation with CFG.
    
    Args:
        model: Trained SoFlow model
        noise: Starting noise [B, C, H, W]
        y: Class labels [B]
        cfg_scale: Guidance scale (≥1.0)
    
    Returns:
        Generated images [B, C, H, W]
    """
    B = noise.shape[0]
    device = noise.device
    
    t = torch.ones(B, device=device)   # Start time
    s = torch.zeros(B, device=device)  # Target time
    
    # Conditional prediction
    f_cond = model(noise, t, s, y)
    
    # Unconditional prediction
    f_uncond = model(noise, t, s, y=None)  # Null class
    
    # CFG combination
    output = f_uncond + cfg_scale * (f_cond - f_uncond)
    
    return output
```

---

## 📊 CFG Scale Effects

### Quality vs Diversity Trade-off

| Scale $w$ | Quality | Diversity | Class Accuracy |
|:---------:|:-------:|:---------:|:--------------:|
| 1.0 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 1.5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **2.0** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 3.0 | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 4.0+ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |

> 💡 **Sweet spot**: CFG scale 1.5-2.5 for best balance

### Mathematical Intuition

- **$w = 1$**: Standard conditional generation
- **$w > 1$**: Amplifies class-specific features
- **$w \to \infty$**: Over-saturation, mode collapse

### Visual Effect

```
w = 1.0:  Diverse but sometimes off-target
w = 2.0:  Good class match, good variety
w = 4.0:  Strong class match, reduced variety
w = 7.0:  Oversaturated, artifacts may appear
```

---

## 📐 Training with CFG

### The Dropout Trick

During training, randomly drop class conditioning:

$$y_{train} = \begin{cases} 
y & \text{with probability } 1 - p_{drop} \\
\varnothing & \text{with probability } p_{drop}
\end{cases}$$

Typical: $p_{drop} = 0.1$

### Training Code

```python
def train_step_with_cfg(model, x_0, y, cfg_dropout=0.1):
    """
    Training step with CFG dropout.
    """
    B = x_0.shape[0]
    
    # Randomly drop class labels
    drop_mask = torch.rand(B) < cfg_dropout
    y_train = y.clone()
    y_train[drop_mask] = NULL_CLASS  # Special null token
    
    # Continue with standard training...
    x_1 = torch.randn_like(x_0)
    t = torch.rand(B) * 0.95 + 0.05
    
    # ... rest of training
```

### Why This Works

- Model learns both: $f_\theta(x, t, s, y)$ and $f_\theta(x, t, s, \varnothing)$
- At test time, we have both conditional and unconditional predictions
- CFG combines them for enhanced generation

---

## 🆚 Method Comparison

| Method | CFG Support | How | Computational Cost |
|:------:|:-----------:|:---:|:------------------:|
| DDPM | ✅ | Per step | $N \times 2 \times$ forward |
| Consistency | ⚠️ | Post-hoc | Medium (artifacts) |
| Distillation | ✅ | Teacher | High (distillation) |
| MeanFlow | ✅ | JVP | **High** (JVP expensive) |
| **SoFlow** | ✅ | Velocity extraction | **Low** (2 forward passes) |

### Cost Analysis for One-Step Generation

| Method | Operation | Cost |
|:------:|:---------:|:----:|
| **SoFlow** | 2 forward passes | **2×** |
| MeanFlow | 2 forward + JVP | **5×** |

> 🚀 SoFlow CFG is **2.5× faster** than JVP-based CFG!

---

## 📐 Multi-Step Generation with CFG

For cases where higher quality is needed, SoFlow supports multi-step generation:

```python
def sample_multistep_cfg(model, noise, y, cfg_scale=2.0, steps=4):
    """
    Multi-step generation with CFG.
    
    Each step uses extracted velocity with CFG.
    """
    x = noise
    
    for i in range(steps):
        t_curr = 1.0 - i / steps
        t_next = 1.0 - (i + 1) / steps
        dt = t_next - t_curr  # Negative
        
        # Extract CFG velocity
        v_cond = extract_velocity(model, x, t_curr, y)
        v_uncond = extract_velocity(model, x, t_curr, y=None)
        v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
        
        # Euler step
        x = x + v_cfg * dt
    
    return x
```

### Quality vs Steps

| Steps | FID | Speed |
|:-----:|:---:|:-----:|
| 1 | 2.96 | 🚀 Instant |
| 2 | 2.54 | Very Fast |
| 4 | 2.31 | Fast |
| 8 | 2.15 | Medium |

---

## 🔑 Key Takeaways

<table>
<tr>
<td>

### 📚 What We Learned
- CFG: $\tilde{v} = v_{uncond} + w \cdot (v_{cond} - v_{uncond})$
- Velocity extraction: $v = \partial f / \partial s$
- Works at training and test time
- Dropout training enables CFG

</td>
<td>

### 🎉 Benefits
- Natural CFG integration
- No JVP computation
- Works with one-step generation
- Multi-step optional boost
- 2.5× faster than alternatives

</td>
</tr>
</table>

---

## 📚 What's Next?

Let's look at the model architecture!

<div align="center">

**[← Chapter 5: Proofs](../05-proofs/README.md)** | **[Chapter 7: Architecture →](../07-architecture/README.md)**

</div>

---

<div align="center">

*Chapter 6 of 9 • [Back to Index](../README.md)*

</div>
