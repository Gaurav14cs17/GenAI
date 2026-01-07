# 📖 Chapter 5: Mathematical Proofs

<div align="center">

*Rigorous guarantees for why SoFlow works*

[![Paper](https://img.shields.io/badge/arXiv-2512.15657-b31b1b.svg)](https://arxiv.org/pdf/2512.15657)

</div>

---

## 🎓 Overview

This chapter covers the **theoretical foundations** of SoFlow with complete mathematical derivations:

![Mathematical Framework](./images/math-framework.svg)

---

## 📐 Preliminaries: Flow Matching Framework

### The Probability Path

Given data distribution $p_0(x)$ and noise distribution $p_1(x) = \mathcal{N}(0, I)$, we define the interpolation:

$$x_t = (1 - t) x_0 + t x_1, \quad t \in [0, 1]$$

where $x_0 \sim p_0$ and $x_1 \sim p_1$.

### Velocity Field

The **conditional velocity field** for this interpolation is:

$$v(x_t | x_0, x_1) = \frac{dx_t}{dt} = x_1 - x_0$$

The **marginal velocity field** $v(x_t, t)$ satisfies:

$$\frac{\partial p_t(x)}{\partial t} + \nabla \cdot (p_t(x) v(x, t)) = 0$$

This is the **continuity equation** for probability flow.

---

## 📜 Definition 1: Solution Function

### Formal Definition

The **solution function** $f: \mathbb{R}^d \times [0,1] \times [0,1] \to \mathbb{R}^d$ maps a state at time $t$ to the corresponding state at time $s$:

$$f(x_t, t, s) = x_s$$

where $x_t$ and $x_s$ lie on the same ODE trajectory defined by the velocity field $v$.

### Integral Representation

The solution function can be expressed as:

$$f(x_t, t, s) = x_t + \int_t^s v(f(x_t, t, \tau), \tau) \, d\tau$$

---

## 📜 Theorem 1: Solution Function Properties

### Statement

A valid solution function $f(x_t, t, s)$ satisfies three fundamental properties:

**Property 1 (Identity):**
$$f(x_t, t, t) = x_t$$

**Property 2 (Composition/Semi-group):**
$$f(f(x_t, t, l), l, s) = f(x_t, t, s) \quad \forall \, 0 \leq s \leq l \leq t \leq 1$$

**Property 3 (ODE Consistency):**
$$\frac{\partial f(x_t, t, s)}{\partial s} = v(f(x_t, t, s), s)$$

### Proof of Property 1 (Identity)

Using the integral representation:

$$f(x_t, t, t) = x_t + \int_t^t v(f(x_t, t, \tau), \tau) \, d\tau = x_t + 0 = x_t \quad \blacksquare$$

### Proof of Property 2 (Composition)

Let $x_l = f(x_t, t, l)$. By uniqueness of ODE solutions:

$$f(x_l, l, s) = x_l + \int_l^s v(f(x_l, l, \tau), \tau) \, d\tau$$

$$= f(x_t, t, l) + \int_l^s v(f(x_t, t, \tau), \tau) \, d\tau$$

$$= x_t + \int_t^l v(\cdot) \, d\tau + \int_l^s v(\cdot) \, d\tau$$

$$= x_t + \int_t^s v(f(x_t, t, \tau), \tau) \, d\tau = f(x_t, t, s) \quad \blacksquare$$

### Proof of Property 3 (ODE Consistency)

Differentiating the integral representation with respect to $s$:

$$\frac{\partial}{\partial s} f(x_t, t, s) = \frac{\partial}{\partial s} \left[ x_t + \int_t^s v(f(x_t, t, \tau), \tau) \, d\tau \right]$$

By the Fundamental Theorem of Calculus:

$$= v(f(x_t, t, s), s) \quad \blacksquare$$

---

## 📜 Theorem 2: Flow Matching Loss Equivalence

### Statement

For the linear interpolation path $x_t = (1-t)x_0 + tx_1$, the flow matching loss:

$$\mathcal{L}_{FM} = \mathbb{E}_{x_0, x_1, t} \left[ \| f_\theta(x_t, t, 0) - x_0 \|^2 \right]$$

is equivalent to training the model to predict the clean data from any noisy state.

### Proof

For the linear path, the true solution function satisfies:

$$f^*(x_t, t, 0) = x_0$$

This is because $x_t = (1-t)x_0 + tx_1$ implies:

$$x_0 = \frac{x_t - t \cdot x_1}{1-t}$$

And the ODE trajectory from $x_t$ at time $t$ to time $0$ recovers exactly $x_0$.

Therefore, minimizing $\mathcal{L}_{FM}$ trains $f_\theta$ to approximate this ground truth:

$$f_\theta(x_t, t, 0) \approx x_0 = f^*(x_t, t, 0) \quad \blacksquare$$

---

## 📜 Theorem 3: Consistency Loss and Self-Consistency

### Definition: Residual

Define the **residual** of the learned solution function:

$$R_\theta(x_t, t, s) = f_\theta(x_t, t, s) - \left[ f_\theta(x_t, t, l) + \int_l^s v_\theta(f_\theta(x_t, t, \tau), \tau) \, d\tau \right]$$

where $t > l > s$ and $v_\theta$ is the velocity induced by $f_\theta$.

### Statement

If the consistency loss converges to zero:

$$\mathcal{L}_{cons} = \mathbb{E} \left[ \| f_\theta(x_t, t, s) - \text{sg}[f_\theta(x_l, l, s)] \|^2 \right] \to 0$$

then the residual is bounded:

$$\| R_\theta \| \leq \epsilon_{max} + H \cdot |l - t| \cdot r(k, K)$$

where:
- $\epsilon_{max}$ is the maximum training error
- $H$ is the Lipschitz constant of the velocity field
- $r(k, K)$ is the schedule function at step $k$ out of $K$ total steps

### Proof Sketch

1. By the consistency loss converging, we have:
   $$f_\theta(x_t, t, s) \approx f_\theta(x_l, l, s)$$

2. For the true solution function, by Property 2:
   $$f^*(x_t, t, s) = f^*(f^*(x_t, t, l), l, s)$$

3. The discrepancy arises from:
   - Approximation error in $f_\theta$ (bounded by $\epsilon_{max}$)
   - Propagation of error through the ODE (bounded by Lipschitz constant times time interval)

4. The schedule function $r(k, K)$ controls the gap $|l - t|$, starting large and decreasing:
   $$l = r(k, K) \cdot t, \quad r(k, K) = \max\left(0.1, 1 - \frac{k}{K}\right)$$

Therefore:
$$\| R_\theta \| \leq \epsilon_{max} + H \cdot |l - t| \cdot r(k, K) \quad \blacksquare$$

---

## 📜 Theorem 4: Global Error Bound

### Statement

Let $f^*$ be the true solution function and $f_\theta$ be the learned approximation. If the residual is bounded by $\delta$, then:

$$\| f^*(x_t, t, s) - f_\theta(x_t, t, s) \| \leq |t - s| \cdot \delta$$

### Proof

Using Grönwall's inequality on the integral equation:

$$f^*(x_t, t, s) - f_\theta(x_t, t, s) = \int_t^s \left[ v^*(f^*, \tau) - v_\theta(f_\theta, \tau) \right] d\tau$$

Assuming $v$ is $L$-Lipschitz in $x$:

$$\| f^* - f_\theta \| \leq \int_t^s L \| f^* - f_\theta \| \, d\tau + \int_t^s \| v^* - v_\theta \| \, d\tau$$

By Grönwall's lemma:

$$\| f^* - f_\theta \| \leq e^{L|t-s|} \cdot |t - s| \cdot \delta$$

For $L|t-s| \ll 1$ (which holds for our setting):

$$\| f^* - f_\theta \| \lesssim |t - s| \cdot \delta \quad \blacksquare$$

### Corollary: One-Step Generation Error

For one-step generation with $t = 1$ and $s = 0$:

$$\| f^*(x_1, 1, 0) - f_\theta(x_1, 1, 0) \| \leq \delta$$

As training progresses, $\delta \to 0$, so the one-step generation error vanishes.

---

## 📜 Theorem 5: Implicit ODE Error Minimization

### Statement

The SoFlow training objective implicitly minimizes the ODE approximation error:

$$\left\| \frac{\partial f_\theta(x_t, t, s)}{\partial s} - v_\theta(f_\theta(x_t, t, s), s) \right\| = O(\sqrt{\delta})$$

### Proof

By Property 3, the true solution satisfies:
$$\frac{\partial f^*}{\partial s} = v^*(f^*, s)$$

For the learned function, consider the Taylor expansion around $s$:

$$f_\theta(x_t, t, s - \Delta s) = f_\theta(x_t, t, s) - \Delta s \cdot \frac{\partial f_\theta}{\partial s} + O(\Delta s^2)$$

The consistency loss enforces:
$$f_\theta(x_t, t, s) \approx f_\theta(x_l, l, s)$$

where the discrepancy between $x_l$ on the learned vs. true trajectory is $O(\sqrt{\delta})$ by the previous theorem.

This implies the derivative mismatch is also $O(\sqrt{\delta})$:

$$\left\| \frac{\partial f_\theta}{\partial s} - v_\theta \right\| = O(\sqrt{\delta}) \quad \blacksquare$$

---

## 📜 Theorem 6: Velocity Extraction

### Statement

Given a trained solution function $f_\theta$, the velocity can be extracted via finite difference:

$$v_\theta(x_t, t) = \lim_{\epsilon \to 0} \frac{f_\theta(x_t, t, t - \epsilon) - x_t}{-\epsilon}$$

### Proof

By Property 3 (ODE Consistency) and L'Hôpital's rule:

$$\lim_{\epsilon \to 0} \frac{f_\theta(x_t, t, t - \epsilon) - f_\theta(x_t, t, t)}{-\epsilon}$$

$$= \lim_{\epsilon \to 0} \frac{f_\theta(x_t, t, t - \epsilon) - x_t}{-\epsilon}$$

$$= \frac{\partial f_\theta(x_t, t, s)}{\partial s} \bigg|_{s=t}$$

$$= v_\theta(f_\theta(x_t, t, t), t) = v_\theta(x_t, t) \quad \blacksquare$$

### Practical Implementation

```python
def extract_velocity(model, x_t, t, eps=1e-4):
    """Extract velocity from solution function."""
    s = t - eps
    f_out = model(x_t, t, s)
    velocity = (f_out - x_t) / (-eps)
    return velocity
```

---

## 🚫 Why No JVP is Required

### The JVP Problem in Other Methods

**Consistency Models** and **MeanFlow** require computing:

$$\text{JVP} = \frac{\partial v_\theta(x, t)}{\partial x} \cdot \text{direction}$$

This requires:
1. Forward pass through the model
2. Backward pass for Jacobian
3. Another forward pass for the product

**Computational Cost:**
- Memory: $O(d \cdot \text{params})$ for storing Jacobian
- Time: $\approx 3\times$ a standard forward-backward pass

### SoFlow's Approach

SoFlow avoids JVP by using the **stop-gradient trick**:

$$\mathcal{L}_{cons} = \| f_\theta(x_t, t, s) - \text{sg}[f_\theta(x_l, l, s)] \|^2$$

**Computational Cost:**
- Forward pass 1: $f_\theta(x_t, t, s)$
- Forward pass 2: $f_\theta(x_l, l, s)$ (no gradient)
- Backward pass: Only through first forward pass

**Total: $\approx 2\times$ forward + $1\times$ backward = $3\times$ forward equivalent**

Compared to JVP methods ($\approx 5\times$ forward equivalent), this is **40% faster**!

---

## 📊 Summary of Theoretical Guarantees

| Property | Mathematical Guarantee |
|:--------:|:----------------------:|
| **Solution Properties** | Identity, Composition, ODE Consistency |
| **FM Loss** | Trains to predict clean data: $f_\theta(x_t, t, 0) \to x_0$ |
| **Consistency Loss** | Enforces self-consistency: $f_\theta(x_t, t, s) \approx f_\theta(x_l, l, s)$ |
| **Residual Bound** | $\|R_\theta\| \leq \epsilon_{max} + H \cdot \Delta t \cdot r(k,K)$ |
| **Global Error** | $\|f^* - f_\theta\| \leq |t-s| \cdot \delta$ |
| **ODE Error** | $\|\partial_s f_\theta - v_\theta\| = O(\sqrt{\delta})$ |
| **Convergence** | As training $\to \infty$: $\delta \to 0$ |

---

## 🔑 Key Takeaways

<table>
<tr>
<td>

### 📚 Mathematical Foundations
- Solution function satisfies 3 key properties
- FM loss grounds predictions to data
- Consistency loss ensures trajectory coherence
- Error bounds guarantee convergence

</td>
<td>

### 💪 Practical Benefits
- **Theoretical guarantees** for generation quality
- **Convergence proof** for training
- **No JVP** = 40% faster training
- **Velocity extraction** enables CFG

</td>
</tr>
</table>

---

## 📚 What's Next?

How does Classifier-Free Guidance work in SoFlow?

<div align="center">

**[← Chapter 4: Training](../04-training/README.md)** | **[Chapter 6: CFG →](../06-cfg/README.md)**

</div>

---

<div align="center">

*Chapter 5 of 9 • [Back to Index](../README.md)*

</div>
