# 📖 Chapter 4: Training Objectives

<div align="center">

*Two complementary losses that make one-step generation possible*

</div>

---

## 🎯 The Training Recipe

SoFlow uses **two losses** working together:

$$\mathcal{L}_{total} = \mathcal{L}_{FM} + \lambda \cdot \mathcal{L}_{cons}$$

where $\lambda$ is typically set to $0.1$.

![Training Losses](./images/training-losses.svg)

---

## 📐 Mathematical Setup

### Data and Noise

Given:
- Data samples: $x_0 \sim p_{data}(x)$
- Noise samples: $x_1 \sim \mathcal{N}(0, I)$
- Time: $t \sim \mathcal{U}(0, 1)$

### Linear Interpolation Path

The noisy sample at time $t$ is:

$$x_t = (1 - t) \cdot x_0 + t \cdot x_1$$

This defines a straight-line path from data ($t=0$) to noise ($t=1$).

### Model Output

The solution function model $f_\theta(x_t, t, s)$ predicts the state at target time $s$ given the current state $x_t$ at time $t$.

---

## 1️⃣ Flow Matching Loss ($\mathcal{L}_{FM}$)

### Mathematical Definition

$$\mathcal{L}_{FM} = \mathbb{E}_{x_0 \sim p_{data}, x_1 \sim \mathcal{N}(0,I), t \sim \mathcal{U}(0,1)} \left[ \left\| f_\theta(x_t, t, 0) - x_0 \right\|_2^2 \right]$$

### What This Loss Does

- **Input**: Noisy sample $x_t$ at time $t$
- **Target**: Clean data $x_0$ (ground truth)
- **Prediction**: $f_\theta(x_t, t, 0)$ — model's estimate of the clean data
- **Goal**: Train the model to denoise perfectly to $s=0$

### Derivation

For the linear interpolation path, the true solution function satisfies:

$$f^*(x_t, t, 0) = x_0$$

**Proof:**
The ODE velocity for linear interpolation is $v = x_1 - x_0$.
Integrating from $t$ to $0$:

$$f^*(x_t, t, 0) = x_t + \int_t^0 v \, d\tau = x_t - t(x_1 - x_0)$$
$$= (1-t)x_0 + tx_1 - t(x_1 - x_0) = (1-t)x_0 + tx_1 - tx_1 + tx_0 = x_0 \quad \blacksquare$$

### Code Implementation

```python
def flow_matching_loss(model, x_0, x_1, t):
    """
    Flow Matching Loss: Train model to predict clean data.
    
    Args:
        model: Solution function f_θ(x, t, s)
        x_0: Clean data samples [B, C, H, W]
        x_1: Noise samples [B, C, H, W]
        t: Time values [B]
    
    Returns:
        Scalar loss value
    """
    # Reshape t for broadcasting: [B] -> [B, 1, 1, 1]
    t_expanded = t.view(-1, 1, 1, 1)
    
    # Create noisy sample via linear interpolation
    x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
    
    # Target time is 0 (clean data)
    s = torch.zeros_like(t)
    
    # Model predicts the clean data
    x_0_pred = model(x_t, t, s)
    
    # MSE loss
    loss = F.mse_loss(x_0_pred, x_0)
    
    return loss
```

---

## 2️⃣ Solution Consistency Loss ($\mathcal{L}_{cons}$)

### Mathematical Definition

$$\mathcal{L}_{cons} = \mathbb{E}_{x_0, x_1, t, l, s} \left[ \left\| f_\theta(x_t, t, s) - \text{sg}\left[f_\theta(x_l, l, s)\right] \right\|_2^2 \right]$$

where:
- $t > l > s$ (decreasing times along trajectory)
- $x_t = (1-t)x_0 + tx_1$
- $x_l = (1-l)x_0 + lx_1$
- $\text{sg}[\cdot]$ is the stop-gradient operator

### The Core Idea: Self-Consistency

For a valid solution function, by the **composition property**:

$$f(x_t, t, s) = f(f(x_t, t, l), l, s)$$

This means:
- **Direct path**: $x_t \xrightarrow{f(\cdot, t, s)} x_s$
- **Indirect path**: $x_t \xrightarrow{f(\cdot, t, l)} x_l \xrightarrow{f(\cdot, l, s)} x_s$

Both paths should reach the **same destination**!

### The Stop-Gradient Trick

Without stop-gradient, both predictions would push toward each other, potentially collapsing to trivial solutions.

With stop-gradient:
- **Target** $f_\theta(x_l, l, s)$ is **fixed** (no gradient)
- **Prediction** $f_\theta(x_t, t, s)$ is **trained** to match it
- This creates a **moving target** that gradually improves

This is similar to:
- **DQN**: Target network vs. online network
- **EMA**: Exponential moving average targets
- **Consistency Models**: Stop-gradient on teacher predictions

### Code Implementation

```python
def consistency_loss(model, x_0, x_1, t, l, s):
    """
    Solution Consistency Loss: Enforce self-consistency.
    
    Args:
        model: Solution function f_θ(x, t, s)
        x_0: Clean data samples [B, C, H, W]
        x_1: Noise samples [B, C, H, W]
        t: Starting time values [B]
        l: Intermediate time values [B], where l < t
        s: Target time values [B], where s < l
    
    Returns:
        Scalar loss value
    """
    # Reshape for broadcasting
    t_exp = t.view(-1, 1, 1, 1)
    l_exp = l.view(-1, 1, 1, 1)
    
    # Create noisy samples at times t and l
    x_t = (1 - t_exp) * x_0 + t_exp * x_1
    x_l = (1 - l_exp) * x_0 + l_exp * x_1
    
    # Direct prediction: x_t → x_s
    pred_direct = model(x_t, t, s)
    
    # Indirect prediction: x_l → x_s (STOP GRADIENT!)
    with torch.no_grad():
        pred_indirect = model(x_l, l, s)
    
    # They should match!
    loss = F.mse_loss(pred_direct, pred_indirect)
    
    return loss
```

---

## 📈 Training Schedule: Curriculum Learning

### The Problem

Early in training:
- Model is poorly trained
- $f_\theta(x_l, l, s)$ produces noisy targets
- Large gaps $(t - l)$ cause unstable training

### The Solution: Progressive Schedule

Start with **small gaps** (easy) and gradually increase to **large gaps** (hard):

$$l = r(k, K) \cdot t$$

where the schedule function is:

$$r(k, K) = \max\left(0.1, 1 - \frac{k}{K}\right)$$

| Training Phase | Step $k$ | $r(k,K)$ | Gap $t-l$ | Difficulty |
|:--------------:|:--------:|:--------:|:---------:|:----------:|
| Start | 0 | 1.0 | 0 | Easy |
| 25% | $K/4$ | 0.75 | $0.25t$ | Low |
| 50% | $K/2$ | 0.5 | $0.5t$ | Medium |
| 75% | $3K/4$ | 0.25 | $0.75t$ | High |
| End | $K$ | 0.1 | $0.9t$ | Hard |

### Why This Works

1. **Early training**: Small gaps $\Rightarrow$ targets are close to predictions $\Rightarrow$ stable gradients
2. **Late training**: Large gaps $\Rightarrow$ model learns long-range consistency
3. **Final stage**: $l \approx 0.1t$ $\Rightarrow$ almost full trajectory consistency

---

## 🔗 Complete Training Algorithm

### Mathematical Formulation

$$\mathcal{L}_{total}(k) = \underbrace{\mathbb{E}\left[\|f_\theta(x_t, t, 0) - x_0\|^2\right]}_{\mathcal{L}_{FM}} + \lambda \cdot \underbrace{\mathbb{E}\left[\|f_\theta(x_t, t, s) - \text{sg}[f_\theta(x_l, l, s)]\|^2\right]}_{\mathcal{L}_{cons}(k)}$$

where:
- $t \sim \mathcal{U}(0.05, 1.0)$
- $l = r(k, K) \cdot t$
- $s = 0$ (target is clean data)
- $\lambda = 0.1$

### Complete Code

```python
def train_step(model, x_0, y, step, total_steps, lambda_cons=0.1):
    """
    Complete SoFlow training step.
    
    Args:
        model: Solution function model f_θ
        x_0: Clean data batch [B, C, H, W]
        y: Class labels [B]
        step: Current training step
        total_steps: Total number of training steps
        lambda_cons: Weight for consistency loss
    
    Returns:
        Dictionary with loss values
    """
    B = x_0.shape[0]
    device = x_0.device
    
    # Sample noise
    x_1 = torch.randn_like(x_0)
    
    # Sample time uniformly from [0.05, 1.0]
    # Avoid t=0 to prevent numerical issues
    t = torch.rand(B, device=device) * 0.95 + 0.05
    
    # === Flow Matching Loss ===
    t_exp = t.view(-1, 1, 1, 1)
    x_t = (1 - t_exp) * x_0 + t_exp * x_1
    
    s = torch.zeros(B, device=device)
    x_0_pred = model(x_t, t, s, y)
    
    loss_fm = F.mse_loss(x_0_pred, x_0)
    
    # === Consistency Loss with Schedule ===
    # Schedule: r starts at 1.0, decreases to 0.1
    r = max(0.1, 1.0 - step / total_steps)
    l = r * t  # Intermediate time
    
    l_exp = l.view(-1, 1, 1, 1)
    x_l = (1 - l_exp) * x_0 + l_exp * x_1
    
    # Stop gradient on target!
    with torch.no_grad():
        x_0_target = model(x_l, l, s, y)
    
    loss_cons = F.mse_loss(x_0_pred, x_0_target)
    
    # === Combined Loss ===
    loss_total = loss_fm + lambda_cons * loss_cons
    
    return {
        "loss": loss_total,
        "loss_fm": loss_fm,
        "loss_cons": loss_cons,
        "schedule_r": r
    }
```

---

## 📊 Loss Behavior During Training

### Expected Curves

| Phase | $\mathcal{L}_{FM}$ | $\mathcal{L}_{cons}$ | Notes |
|:-----:|:------------------:|:--------------------:|:------|
| Initial | High | Low | Easy consistency (small gap) |
| Early | Decreasing | Stable | FM dominates learning |
| Middle | Low | Increasing | Gap grows, consistency harder |
| Late | Low | Decreasing | Model masters long-range |
| Final | ~0.01 | ~0.01 | Both converged |

### Interpretation

- **$\mathcal{L}_{FM}$ decreasing** → Model learns to denoise
- **$\mathcal{L}_{cons}$ bump then decrease** → Schedule effect
- **Both low at end** → One-step generation works!

---

## 🤔 Why Two Losses? Ablation Study

| Configuration | FM Loss | Cons Loss | Result |
|:-------------:|:-------:|:---------:|:------:|
| FM only | ✅ | ❌ | ⚠️ Learns denoising but no trajectory consistency |
| Cons only | ❌ | ✅ | ❌ No ground truth signal, collapses |
| **Both** | ✅ | ✅ | ✅ Perfect: consistent denoising |

### Mathematical Intuition

- **FM Loss** provides the **ground truth anchor**: $f_\theta(\cdot, \cdot, 0) \to x_0$
- **Consistency Loss** propagates this to **all intermediate times**: $f_\theta(\cdot, t, s)$ self-consistent
- **Together** they enforce that $f_\theta$ is a valid solution function

---

## 🔑 Key Takeaways

<table>
<tr>
<td>

### 📚 Summary
- **$\mathcal{L}_{FM}$**: $\|f_\theta(x_t, t, 0) - x_0\|^2$ — supervised denoising
- **$\mathcal{L}_{cons}$**: $\|f_\theta(x_t, t, s) - \text{sg}[f_\theta(x_l, l, s)]\|^2$ — self-consistency
- **Schedule**: $r(k,K) = \max(0.1, 1 - k/K)$ — curriculum learning

</td>
<td>

### 💪 Benefits
- Stable training (no mode collapse)
- No JVP computation needed
- Curriculum handles difficulty progression
- One-step generation at test time

</td>
</tr>
</table>

---

## 📚 What's Next?

Want the mathematical guarantees and proofs?

<div align="center">

**[← Chapter 3: Solution Function](../03-solution-function/README.md)** | **[Chapter 5: Proofs →](../05-proofs/README.md)**

</div>

---

<div align="center">

*Chapter 4 of 9 • [Back to Index](../README.md)*

</div>
