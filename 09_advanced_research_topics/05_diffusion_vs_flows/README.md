# 🔀 Diffusion vs Flow Models

<div align="center">

![Diffusion vs Flows](./images/comparison_hero.svg)

*Two perspectives on the same goal: transforming noise to data*

[![Diffusion](https://img.shields.io/badge/SDE-Diffusion-f97316?style=for-the-badge)](#)
[![Flows](https://img.shields.io/badge/ODE-Flows-22c55e?style=for-the-badge)](#)
[![Unified](https://img.shields.io/badge/Theory-Unified-a855f7?style=for-the-badge)](#)

</div>

---

## 🎯 Where & Why: Understanding the Landscape

### Why Does This Comparison Matter?

| Decision Point | What to Consider |
|---------------|------------------|
| 🎯 **Choosing an approach** | When to use diffusion vs flow matching |
| 🔬 **Research direction** | Understanding theoretical foundations |
| 🛠️ **Implementation** | Converting between formulations |
| 📚 **Paper reading** | Understanding different notations |
| ⚡ **Optimization** | Leveraging best of both worlds |

### When to Use Which?

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| **Production deployment** | Flow Matching | Fewer steps, straight paths |
| **Maximum quality** | Diffusion | More mature, proven |
| **Exact likelihood** | Flows | Tractable computation |
| **Pretrained models** | Diffusion | Larger ecosystem |
| **New architecture** | Flow Matching | Simpler to train |
| **Research flexibility** | Both | Unified theory enables conversion |

---

## 📖 The Two Paradigms

### Diffusion Models (SDE Perspective)

Diffusion models define a **forward process** that gradually adds noise:

$$dx = f(x, t)dt + g(t)dW_t$$

where:
- \(f(x, t)\): Drift coefficient
- \(g(t)\): Diffusion coefficient  
- \(dW_t\): Wiener process (Brownian motion)

**Common choice (VP-SDE):**
$$dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dW_t$$

**Reverse process (for generation):**
$$dx = \left[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\right]dt + g(t)d\bar{W}_t$$

### Flow Matching (ODE Perspective)

Flow matching learns a **velocity field** directly:

$$\frac{dx}{dt} = v_\theta(x, t)$$

No stochastic term — this is a deterministic ODE!

**Training:** Match the learned velocity to the true velocity:
$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - u_t(x_t | x_1)\|^2\right]$$

---

## 🧮 Mathematical Connections

### The Probability Flow ODE

**Key Insight:** Every diffusion SDE has an equivalent ODE with the same marginal distributions!

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

This is called the **Probability Flow ODE (PF-ODE)**.

### Score Function ↔ Velocity Field

The score function and velocity field are related:

$$v_t(x) = f(x,t) - \frac{1}{2}g(t)^2 \nabla_x \log p_t(x)$$

**For the VP-SDE specifically:**
$$v_t(x) = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

### Converting Between Formulations

```python
def score_to_velocity(score, x, t, beta_t):
    """Convert score function to velocity field (VP-SDE)."""
    return -0.5 * beta_t * (x + score)

def velocity_to_score(velocity, x, t, beta_t):
    """Convert velocity field to score function (VP-SDE)."""
    return -2 * velocity / beta_t - x

def eps_to_score(eps_pred, x_t, t, alpha_t, sigma_t):
    """Convert epsilon prediction to score."""
    # score = -eps / sigma
    return -eps_pred / sigma_t

def score_to_eps(score, sigma_t):
    """Convert score to epsilon prediction."""
    return -score * sigma_t
```

---

## 📊 Detailed Comparison

### Formulation Comparison

| Aspect | Diffusion (SDE) | Flow Matching (ODE) |
|--------|-----------------|---------------------|
| **Process type** | Stochastic | Deterministic |
| **Equation** | \(dx = f\,dt + g\,dW\) | \(dx/dt = v(x,t)\) |
| **Training target** | Score \(\nabla \log p_t\) or \(\epsilon\) | Velocity \(v\) |
| **Path shape** | Curved (typically) | Can be straight |
| **Noise schedule** | Critical hyperparameter | Built into interpolation |
| **Likelihood** | Via ODE conversion | Exact (CNF formula) |

### Training Comparison

| Aspect | Score Matching | Flow Matching |
|--------|---------------|---------------|
| **Loss** | \(\|\epsilon_\theta - \epsilon\|^2\) | \(\|v_\theta - (x_1-x_0)\|^2\) |
| **Target** | Added noise | Velocity direction |
| **Weighting** | SNR-based | Uniform (typically) |
| **Simplicity** | Medium | Simple |

### Sampling Comparison

| Aspect | Diffusion | Flow |
|--------|-----------|------|
| **Solvers** | DDPM, DDIM, DPM++ | Euler, Heun, RK45 |
| **Stochasticity** | Optional | Deterministic |
| **Typical steps** | 20-50 | 4-20 |
| **CFG integration** | Standard | Same |

---

## 🔧 Implementation: Unified View

### A Unified Generator Class

```python
class UnifiedGenerator:
    """
    A unified interface for diffusion and flow-based generation.
    
    Shows how both paradigms can be implemented with shared code.
    """
    
    def __init__(self, model, mode='flow'):
        self.model = model
        self.mode = mode  # 'diffusion' or 'flow'
    
    def get_velocity(self, x_t, t):
        """Get velocity field (works for both modes)."""
        if self.mode == 'flow':
            # Flow matching: model directly predicts velocity
            return self.model(x_t, t)
        else:
            # Diffusion: convert noise prediction to velocity
            eps_pred = self.model(x_t, t)
            # For VP-SDE with specific schedule
            alpha_t, sigma_t = self.get_alpha_sigma(t)
            # v = (alpha_t * eps - sigma_t * x) / (alpha_t^2 + sigma_t^2)
            # Simplified for linear schedule
            return -eps_pred  # Approximate conversion
    
    def sample_euler(self, shape, num_steps=10):
        """Euler sampling (works for both)."""
        x = torch.randn(shape)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = torch.full((shape[0],), i * dt)
            v = self.get_velocity(x, t)
            x = x + v * dt
        
        return x
    
    def sample_heun(self, shape, num_steps=10):
        """Heun's method (2nd order, works for both)."""
        x = torch.randn(shape)
        dt = 1.0 / num_steps
        
        for i in range(num_steps):
            t = i * dt
            t_next = t + dt
            
            # First evaluation
            v1 = self.get_velocity(x, torch.full((shape[0],), t))
            x_euler = x + v1 * dt
            
            # Second evaluation at predicted point
            v2 = self.get_velocity(x_euler, torch.full((shape[0],), t_next))
            
            # Average
            x = x + (v1 + v2) * dt / 2
        
        return x
```

### Training: Side by Side

```python
# ===== DIFFUSION TRAINING =====
def diffusion_training_step(model, x_0, noise_schedule):
    # Sample timestep
    t = torch.randint(0, 1000, (x_0.shape[0],))
    
    # Get noise schedule parameters
    alpha_t = noise_schedule.alpha(t)
    sigma_t = noise_schedule.sigma(t)
    
    # Add noise
    eps = torch.randn_like(x_0)
    x_t = alpha_t * x_0 + sigma_t * eps
    
    # Predict noise
    eps_pred = model(x_t, t)
    
    # MSE loss on noise
    loss = F.mse_loss(eps_pred, eps)
    return loss


# ===== FLOW MATCHING TRAINING =====
def flow_matching_training_step(model, x_1):  # x_1 is data
    # Sample timestep uniformly
    t = torch.rand(x_1.shape[0], 1, 1, 1)
    
    # Sample noise
    x_0 = torch.randn_like(x_1)
    
    # Linear interpolation
    x_t = (1 - t) * x_0 + t * x_1
    
    # Target velocity
    v_target = x_1 - x_0
    
    # Predict velocity
    v_pred = model(x_t, t.squeeze())
    
    # MSE loss on velocity
    loss = F.mse_loss(v_pred, v_target)
    return loss
```

---

## 🌊 Hybrid Approaches

### Stochastic Interpolants

Combine the determinism of flows with optional stochasticity:

$$x_t = \alpha(t) x_0 + \beta(t) x_1 + \gamma(t) W_t$$

where \(W_t\) is controlled noise injection.

```python
def stochastic_interpolant(x_0, x_1, t, gamma=0.1):
    """
    Stochastic interpolant with controlled noise.
    
    gamma=0: Pure flow matching (deterministic)
    gamma>0: Add controlled stochasticity
    """
    # Deterministic component
    mean = (1 - t) * x_0 + t * x_1
    
    # Stochastic component (maximal at t=0.5)
    std = gamma * torch.sqrt(t * (1 - t))
    noise = torch.randn_like(x_0)
    
    return mean + std * noise
```

### EDM: Unifying Framework

The EDM (Elucidating the Design Space) framework provides a unified view:

$$D_\theta(x; \sigma) = c_{\text{skip}}(\sigma) x + c_{\text{out}}(\sigma) F_\theta(c_{\text{in}}(\sigma) x; c_{\text{noise}}(\sigma))$$

This can represent both diffusion and flow models with different choices of \(c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}\).

---

## 📐 Key Equations Summary

| Concept | Diffusion (SDE) | Flow (ODE) |
|---------|-----------------|------------|
| **Forward** | \(dx = f\,dt + g\,dW\) | \(x_t = (1-t)x_0 + tx_1\) |
| **Reverse** | \(dx = [f - g^2\nabla\log p]\,dt + g\,d\bar{W}\) | \(dx/dt = v_\theta(x,t)\) |
| **Training target** | \(\epsilon\) or \(\nabla\log p\) | \(v = x_1 - x_0\) |
| **PF-ODE** | \(dx/dt = f - \frac{1}{2}g^2\nabla\log p\) | Same as generation |
| **Likelihood** | \(\log p(x) = \log p(x_T) - \int \nabla \cdot v\,dt\) | Same formula |

---

## 🎓 When Theory Meets Practice

### Practical Recommendations

| Situation | Choice | Reasoning |
|-----------|--------|-----------|
| **New project, 2024+** | Flow Matching | Simpler, state-of-the-art |
| **Using SD/SDXL** | Diffusion | Pretrained ecosystem |
| **Maximum control** | Diffusion | More research on conditioning |
| **Speed critical** | Rectified Flow | Fewest steps |
| **Research paper** | Know both | Reviewers expect it |

### The Convergence

The field is converging toward flow-based methods:
- **SD3:** Uses flow matching (rectified flow)
- **Flux:** Uses flow matching
- **Sora:** Likely uses flow-based training

But diffusion's ecosystem and research depth remain valuable.

---

## 📚 References

1. **Song, Y., et al.** (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR*. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

2. **Lipman, Y., et al.** (2023). "Flow Matching for Generative Modeling." *ICLR*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)

3. **Albergo, M. & Vanden-Eijnden, E.** (2023). "Building Normalizing Flows with Stochastic Interpolants." *ICLR*. [arXiv:2209.15571](https://arxiv.org/abs/2209.15571)

4. **Karras, T., et al.** (2022). "Elucidating the Design Space of Diffusion-Based Generative Models." *NeurIPS*. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364)

---

## ✏️ Exercises

<details>
<summary><b>Exercise 1: Convert Between Formulations</b></summary>

Given a trained diffusion model (noise predictor):
1. Convert to velocity predictor
2. Sample using both formulations
3. Verify identical outputs (for deterministic sampling)

</details>

<details>
<summary><b>Exercise 2: Compare Training</b></summary>

On the same dataset and architecture:
1. Train with diffusion loss
2. Train with flow matching loss
3. Compare convergence speed
4. Compare final sample quality

</details>

<details>
<summary><b>Exercise 3: Implement Stochastic Interpolants</b></summary>

Implement stochastic interpolants with varying \(\gamma\):
1. \(\gamma = 0\) (pure flow)
2. \(\gamma = 0.1, 0.5, 1.0\)
3. Compare sample diversity
4. Measure quality-diversity trade-off

</details>

---

<div align="center">

**[← Diffusion Distillation](../04_diffusion_distillation/)** | **[Next: Video Generation →](../06_generative_video_models/)**

</div>
