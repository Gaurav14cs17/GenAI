# 📡 Information Theory for Generative AI

<div align="center">

![Information Theory in Generative AI](images/information_theory_genai.svg)

*Quantifying information in generative models*

</div>

---

## 📖 Introduction

Information theory, founded by Claude Shannon in 1948, provides the mathematical framework for quantifying, storing, and communicating information. In generative AI, information-theoretic concepts appear everywhere: entropy measures the complexity of distributions, KL divergence quantifies distribution mismatch, and mutual information captures learned representations.

This guide provides rigorous foundations in information theory with deep connections to generative modeling.

---

## 🎯 Where and Why Use Information Theory in Generative AI

### Where It's Used

| Application | Information Theory Concepts | Example Models |
|-------------|---------------------------|----------------|
| **VAE Training** | KL divergence, ELBO, rate-distortion tradeoff | VAE, β-VAE, VQ-VAE |
| **GAN Objectives** | Jensen-Shannon divergence, f-divergences | Original GAN, f-GAN |
| **Wasserstein GANs** | Optimal transport, Earth Mover's distance | WGAN, WGAN-GP |
| **Representation Learning** | Mutual information, information bottleneck | InfoGAN, VIB, contrastive learning |
| **Language Models** | Cross-entropy loss, perplexity, bits-per-character | GPT, BERT, all LLMs |
| **Compression** | Rate-distortion theory, entropy coding | Neural compression, VQ-VAE |
| **Disentanglement** | Total correlation, mutual information | β-VAE, FactorVAE |
| **Model Evaluation** | Log-likelihood, entropy of generated samples | All generative models |

### Why It's Essential

1. **Loss Functions Are Information-Theoretic:**
   - Cross-entropy loss = $H(p, q) = H(p) + D_{KL}(p \| q)$
   - VAE loss = Reconstruction + $D_{KL}(q(z|x) \| p(z))$
   - GAN (original) minimizes Jensen-Shannon divergence

2. **Understanding KL Divergence:**
   - Why VAE's KL term encourages smooth latent spaces
   - Forward vs Reverse KL: mode-covering vs mode-seeking
   - Why MLE (forward KL) can lead to blurry outputs

3. **Mutual Information for Disentanglement:**
   - InfoGAN: $\max I(c; G(z,c))$ to learn interpretable codes
   - β-VAE: Penalize $I(X; Z)$ for disentanglement
   - Contrastive learning: Maximize $I(view_1; view_2)$

4. **Evaluating Generative Models:**
   - Bits-per-dimension for likelihood models
   - Perplexity for language models: $2^{H(p, q)}$
   - Entropy of generated distribution (diversity)

5. **Training Stability:**
   - KL vanishing in VAEs (posterior collapse)
   - Mode collapse in GANs (low entropy)
   - Understanding why certain divergences work better

### What Happens Without This Knowledge

- ❌ Can't understand why VAE produces blurry images (forward KL)
- ❌ Can't interpret or modify loss functions meaningfully
- ❌ Can't understand InfoGAN or disentanglement methods
- ❌ Can't evaluate language models (perplexity)
- ❌ Can't understand WGAN's advantages over original GAN
- ❌ Can't diagnose posterior collapse or mode collapse

### Key Insight: Why Different Divergences Matter

| Divergence | Behavior | Used In |
|------------|----------|---------|
| **Forward KL** $D_{KL}(p_{data} \| p_\theta)$ | Mode-covering, blurry | VAE decoder, MLE |
| **Reverse KL** $D_{KL}(q \| p)$ | Mode-seeking, sharp | VAE encoder, VI |
| **Jensen-Shannon** | Symmetric, bounded | Original GAN |
| **Wasserstein** | Smooth gradients everywhere | WGAN |

Understanding these tradeoffs is crucial for choosing and designing generative models.

---

## 📊 Representation Comparison

| Representation | Pros | Cons |
|----------------|------|------|
| **Entropy** | Measures uncertainty, fundamental quantity | Requires known distribution |
| **Cross-Entropy** | Direct loss function, tractable | Asymmetric, mode-covering |
| **KL Divergence** | Principled, information-theoretic | Asymmetric, can be infinite |
| **Mutual Information** | Captures all dependencies, symmetric | Hard to estimate in high-d |
| **Fisher Information** | Local geometry, efficient estimation | Only local information |

---

## 1. Entropy: Quantifying Uncertainty

### 1.1 Shannon Entropy

**Definition:** For a discrete random variable $X$ with PMF $p(x)$:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

**Convention:** $0 \log 0 = 0$ (justified by continuity: $\lim_{p \to 0^+} p \log p = 0$)

**Units:** 
- Base 2: bits
- Base $e$: nats
- Base 10: hartleys

**Interpretation:** Expected number of bits needed to encode outcomes from $X$.

### 1.2 Properties of Entropy

**Theorem 1 (Non-negativity):** $H(X) \geq 0$

**Proof:** $-\log p(x) \geq 0$ for $p(x) \in [0, 1]$, so expectation is non-negative.

**Theorem 2 (Maximum Entropy):** For $|\mathcal{X}| = n$:

$$H(X) \leq \log n$$

with equality iff $X$ is uniformly distributed.

**Proof:** Let $u(x) = 1/n$ be uniform. Then:
$$H(X) - \log n = -\sum_x p(x) \log p(x) - \sum_x p(x) \log n = -\sum_x p(x) \log(p(x) \cdot n) = -D_{KL}(p \| u) \leq 0$$

### 1.3 Differential Entropy

For continuous random variables with PDF $f(x)$:

$$h(X) = -\int f(x) \log f(x) \, dx$$

**Warning:** Differential entropy can be negative! For $X \sim \text{Uniform}(0, a)$:
$$h(X) = -\int_0^a \frac{1}{a} \log \frac{1}{a} \, dx = \log a$$

which is negative for $a < 1$.

**Theorem (Maximum Entropy under Constraints):**

1. **Fixed support $[a, b]$:** Maximum entropy distribution is $\text{Uniform}(a, b)$
2. **Fixed mean and variance:** Maximum entropy distribution is Gaussian

**Proof of (2):** Use calculus of variations. For constraint $\mathbb{E}[X] = \mu$, $\mathbb{E}[(X-\mu)^2] = \sigma^2$:

Maximize $-\int f(x) \log f(x) \, dx$ subject to constraints.

Lagrangian: $\mathcal{L} = -\int f \log f \, dx - \lambda_0(\int f \, dx - 1) - \lambda_1(\int xf \, dx - \mu) - \lambda_2(\int (x-\mu)^2 f \, dx - \sigma^2)$

Setting $\frac{\delta \mathcal{L}}{\delta f} = 0$:
$$-\log f(x) - 1 - \lambda_0 - \lambda_1 x - \lambda_2(x-\mu)^2 = 0$$

$$f(x) \propto \exp(-\lambda_2(x-\mu)^2)$$

which is Gaussian. The maximum differential entropy is:

$$h(X) = \frac{1}{2}\log(2\pi e \sigma^2)$$

---

## 2. Joint and Conditional Entropy

### 2.1 Joint Entropy

$$H(X, Y) = -\sum_{x, y} p(x, y) \log p(x, y)$$

### 2.2 Conditional Entropy

$$H(Y|X) = \sum_x p(x) H(Y|X=x) = -\sum_{x, y} p(x, y) \log p(y|x)$$

**Chain Rule:**
$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

**General Chain Rule:**
$$H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i | X_1, \ldots, X_{i-1})$$

### 2.3 Properties

**Theorem (Conditioning Reduces Entropy):**
$$H(Y|X) \leq H(Y)$$

with equality iff $X$ and $Y$ are independent.

**Proof:** 
$$H(Y) - H(Y|X) = I(X; Y) \geq 0$$

(See mutual information below)

---

## 3. Kullback-Leibler Divergence

### 3.1 Definition and Properties

**Definition:**
$$D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_P\left[\log \frac{P(X)}{Q(X)}\right]$$

**Continuous version:**
$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

**Properties:**

1. **Non-negativity (Gibbs' Inequality):** $D_{KL}(P \| Q) \geq 0$
2. **Zero condition:** $D_{KL}(P \| Q) = 0 \Leftrightarrow P = Q$ a.e.
3. **Asymmetry:** $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general
4. **Not a metric:** Doesn't satisfy triangle inequality
5. **Convexity:** $D_{KL}(P \| Q)$ is convex in $(P, Q)$

### 3.2 Proof of Gibbs' Inequality

**Theorem:** $D_{KL}(P \| Q) \geq 0$ with equality iff $P = Q$.

**Proof using Jensen's inequality:**

$$D_{KL}(P \| Q) = -\mathbb{E}_P\left[\log \frac{Q(X)}{P(X)}\right] \geq -\log \mathbb{E}_P\left[\frac{Q(X)}{P(X)}\right]$$

(since $-\log$ is convex)

$$= -\log \sum_x p(x) \frac{q(x)}{p(x)} = -\log \sum_x q(x) = -\log 1 = 0$$

Equality holds iff $\frac{Q(X)}{P(X)}$ is constant, i.e., $P = Q$.

### 3.3 Forward vs Reverse KL

The asymmetry of KL divergence leads to different behaviors:

**Forward KL: $D_{KL}(P \| Q)$ (minimize over $Q$)**
- Mean-seeking: $Q$ covers all modes of $P$
- Penalizes $q(x) \approx 0$ when $p(x) > 0$
- Used in: Maximum likelihood estimation

**Reverse KL: $D_{KL}(Q \| P)$ (minimize over $Q$)**
- Mode-seeking: $Q$ concentrates on modes of $P$
- Allows $q(x) > 0$ when $p(x) \approx 0$
- Used in: Variational inference

**Relevance to GenAI:**
- VAE uses reverse KL: $D_{KL}(q_\phi(z|x) \| p(z))$
- MLE training minimizes forward KL: $D_{KL}(p_{data} \| p_\theta)$

### 3.4 KL Divergence for Exponential Families

For $p(x|\theta_p)$ and $q(x|\theta_q)$ in exponential family:

$$p(x|\theta) = h(x) \exp(\theta^T T(x) - A(\theta))$$

$$D_{KL}(p \| q) = A(\theta_q) - A(\theta_p) - (\theta_q - \theta_p)^T \nabla A(\theta_p)$$

This is the **Bregman divergence** associated with $A$.

### 3.5 KL Divergence for Gaussians

**Univariate:**
$$D_{KL}(\mathcal{N}(\mu_1, \sigma_1^2) \| \mathcal{N}(\mu_2, \sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**Multivariate:**
$$D_{KL}(\mathcal{N}(\mu_1, \Sigma_1) \| \mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2-\mu_1)^T\Sigma_2^{-1}(\mu_2-\mu_1)\right]$$

**VAE KL term (diagonal covariance):**
$$D_{KL}(\mathcal{N}(\mu, \text{diag}(\sigma^2)) \| \mathcal{N}(0, I)) = \frac{1}{2}\sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)$$

---

## 4. Mutual Information

### 4.1 Definition

**Mutual Information:**
$$I(X; Y) = D_{KL}(p(x, y) \| p(x)p(y)) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}$$

**Equivalent forms:**
$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X, Y)$$

### 4.2 Properties

1. **Non-negativity:** $I(X; Y) \geq 0$
2. **Symmetry:** $I(X; Y) = I(Y; X)$
3. **Independence:** $I(X; Y) = 0 \Leftrightarrow X \perp Y$
4. **Self-information:** $I(X; X) = H(X)$
5. **Data Processing Inequality:** For Markov chain $X \to Y \to Z$:
   $$I(X; Y) \geq I(X; Z)$$

### 4.3 Conditional Mutual Information

$$I(X; Y | Z) = H(X|Z) - H(X|Y, Z) = \mathbb{E}_{p(z)}[I(X; Y | Z=z)]$$

**Chain Rule for MI:**
$$I(X_1, X_2; Y) = I(X_1; Y) + I(X_2; Y | X_1)$$

### 4.4 Applications in Generative AI

**InfoGAN:** Maximize $I(c; G(z, c))$ where $c$ is latent code and $G(z, c)$ is generated image.

Since $I(c; G(z, c))$ is hard to compute, use variational lower bound:

$$I(c; G(z, c)) \geq \mathbb{E}_{c \sim p(c), x = G(z, c)}[\log Q(c|x)] + H(c)$$

where $Q(c|x)$ is auxiliary network.

**Variational Information Bottleneck:** Learn representation $Z$ of $X$ that predicts $Y$:

$$\max_{p(z|x)} I(Z; Y) - \beta I(X; Z)$$

---

## 5. Cross-Entropy

### 5.1 Definition

$$H(P, Q) = -\sum_x p(x) \log q(x) = \mathbb{E}_P[-\log Q(X)]$$

**Relationship to KL:**
$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

### 5.2 Cross-Entropy Loss

For classification with true labels $y$ and predictions $\hat{p}$:

$$\mathcal{L}_{CE} = -\sum_{c=1}^C y_c \log \hat{p}_c$$

For binary classification:
$$\mathcal{L}_{BCE} = -[y \log \hat{p} + (1-y) \log(1-\hat{p})]$$

**Why Cross-Entropy?**

Minimizing cross-entropy is equivalent to:
1. Maximum likelihood estimation
2. Minimizing KL divergence from true to predicted distribution

---

## 6. Fisher Information

### 6.1 Definition

For parametric family $p(x|\theta)$:

$$I(\theta) = \mathbb{E}_{p(x|\theta)}\left[\left(\frac{\partial}{\partial\theta}\log p(X|\theta)\right)^2\right]$$

**Matrix form (for vector $\theta$):**
$$[I(\theta)]_{ij} = \mathbb{E}\left[\frac{\partial \log p}{\partial \theta_i} \frac{\partial \log p}{\partial \theta_j}\right]$$

### 6.2 Alternative Expression

Under regularity conditions:
$$I(\theta) = -\mathbb{E}\left[\frac{\partial^2}{\partial\theta^2}\log p(X|\theta)\right]$$

**Proof:** 
$$\frac{\partial}{\partial\theta}\int p(x|\theta) dx = \int \frac{\partial p}{\partial\theta} dx = 0$$

Differentiating again and using $\frac{\partial \log p}{\partial\theta} = \frac{1}{p}\frac{\partial p}{\partial\theta}$:

$$0 = \int \frac{\partial^2 p}{\partial\theta^2} dx = \int p \frac{\partial^2 \log p}{\partial\theta^2} dx + \int p \left(\frac{\partial \log p}{\partial\theta}\right)^2 dx$$

### 6.3 Cramér-Rao Bound

**Theorem:** For unbiased estimator $\hat{\theta}$ of $\theta$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

MLE achieves this bound asymptotically.

### 6.4 Fisher Information and KL Divergence

$$D_{KL}(p_\theta \| p_{\theta+\delta\theta}) \approx \frac{1}{2}\delta\theta^T I(\theta) \delta\theta$$

Fisher information is the Hessian of KL divergence!

**Relevance to GenAI:** Natural gradient methods use $I(\theta)^{-1}\nabla_\theta$ instead of $\nabla_\theta$.

---

## 7. Rate-Distortion Theory

### 7.1 The Rate-Distortion Function

**Problem:** Compress source $X$ to representation $\hat{X}$ with:
- **Rate:** $R = I(X; \hat{X})$ bits
- **Distortion:** $D = \mathbb{E}[d(X, \hat{X})]$

**Rate-Distortion Function:**
$$R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(X,\hat{X})] \leq D} I(X; \hat{X})$$

### 7.2 Connection to VAE

The VAE objective can be viewed through rate-distortion:

$$\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[-\log p(x|z)]}_{\text{Distortion}} + \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{Rate}}$$

The KL term is an upper bound on $I(X; Z)$ under $q$.

**β-VAE** explicitly trades off rate and distortion:
$$\mathcal{L}_\beta = \text{Distortion} + \beta \cdot \text{Rate}$$

---

## 8. Other Divergences and Distances

### 8.1 f-Divergences

**Definition:** For convex function $f$ with $f(1) = 0$:

$$D_f(P \| Q) = \mathbb{E}_Q\left[f\left(\frac{p(X)}{q(X)}\right)\right]$$

**Examples:**
| Name | $f(t)$ | $D_f(P \| Q)$ |
|------|--------|---------------|
| KL | $t \log t$ | $\sum p \log(p/q)$ |
| Reverse KL | $-\log t$ | $\sum q \log(q/p)$ |
| Total Variation | $\frac{1}{2}|t-1|$ | $\frac{1}{2}\sum |p - q|$ |
| Chi-squared | $(t-1)^2$ | $\sum (p-q)^2/q$ |
| Jensen-Shannon | $t \log t - (t+1)\log\frac{t+1}{2}$ | See below |

### 8.2 Jensen-Shannon Divergence

$$D_{JS}(P \| Q) = \frac{1}{2}D_{KL}(P \| M) + \frac{1}{2}D_{KL}(Q \| M)$$

where $M = \frac{1}{2}(P + Q)$.

**Properties:**
- Symmetric: $D_{JS}(P \| Q) = D_{JS}(Q \| P)$
- Bounded: $0 \leq D_{JS} \leq \log 2$
- $\sqrt{D_{JS}}$ is a metric

**Relevance to GenAI:** Original GAN minimizes JS divergence between $p_{data}$ and $p_G$.

### 8.3 Wasserstein Distance

**1-Wasserstein (Earth Mover's) Distance:**

$$W_1(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

where $\Gamma(P, Q)$ is the set of all joint distributions with marginals $P$ and $Q$.

**Kantorovich-Rubinstein Duality:**
$$W_1(P, Q) = \sup_{\|f\|_L \leq 1} \left[\mathbb{E}_{x \sim P}[f(x)] - \mathbb{E}_{y \sim Q}[f(y)]\right]$$

where $\|f\|_L \leq 1$ means $f$ is 1-Lipschitz.

**Relevance to GenAI:** WGAN uses Wasserstein distance, leading to more stable training.

---

## 9. Information-Theoretic Bounds

### 9.1 Data Processing Inequality

**Theorem:** For Markov chain $X \to Y \to Z$:
$$I(X; Z) \leq I(X; Y)$$

**Corollary:** Post-processing cannot increase information.

**Proof:** 
$$I(X; Y, Z) = I(X; Z) + I(X; Y|Z) = I(X; Y) + I(X; Z|Y)$$

Since $X \to Y \to Z$, we have $I(X; Z|Y) = 0$, so:
$$I(X; Z) \leq I(X; Y)$$

### 9.2 Fano's Inequality

**Theorem:** For random variables $X$ and $\hat{X}$ with $P_e = P(X \neq \hat{X})$:

$$H(X|\hat{X}) \leq H_b(P_e) + P_e \log(|\mathcal{X}| - 1)$$

where $H_b(p) = -p\log p - (1-p)\log(1-p)$ is binary entropy.

**Implication:** Lower bounds reconstruction error from information content.

---

## 10. Information Geometry

### 10.1 Statistical Manifold

The space of probability distributions forms a **Riemannian manifold** with Fisher information as the metric.

**Fisher-Rao Metric:**
$$ds^2 = \sum_{ij} I_{ij}(\theta) d\theta_i d\theta_j$$

### 10.2 Natural Gradient

**Standard gradient:** $\nabla_\theta \mathcal{L}$

**Natural gradient:** $I(\theta)^{-1} \nabla_\theta \mathcal{L}$

The natural gradient is the steepest descent direction in the space of distributions.

**Relevance:** Natural gradient methods (K-FAC, etc.) can improve training of generative models.

---

## Key Results Summary

| Concept | Formula | Application |
|---------|---------|-------------|
| Entropy | $H(X) = -\sum p \log p$ | Uncertainty quantification |
| KL Divergence | $D_{KL}(p\|q) = \sum p \log(p/q)$ | VAE loss, distribution matching |
| Cross-Entropy | $H(p, q) = -\sum p \log q$ | Classification loss |
| Mutual Information | $I(X;Y) = H(X) - H(X|Y)$ | InfoGAN, representation learning |
| Jensen-Shannon | $\frac{1}{2}D_{KL}(p\|m) + \frac{1}{2}D_{KL}(q\|m)$ | Original GAN objective |

---

## References

### Foundational Papers
1. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379-423.
2. **Kullback, S., & Leibler, R. A.** (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics*, 22(1), 79-86.

### Textbooks
1. **Cover, T. M., & Thomas, J. A.** (2006). *Elements of Information Theory* (2nd ed.). Wiley.
2. **MacKay, D. J. C.** (2003). *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press. [Free online](http://www.inference.org.uk/mackay/itila/)

### GenAI Applications
1. **Alemi, A. A., et al.** (2017). "Deep Variational Information Bottleneck." *ICLR*. [arXiv:1612.00410](https://arxiv.org/abs/1612.00410)
2. **Chen, X., et al.** (2016). "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets." *NeurIPS*. [arXiv:1606.03657](https://arxiv.org/abs/1606.03657)
3. **Higgins, I., et al.** (2017). "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework." *ICLR*.

---

## Exercises

1. **Prove** the chain rule for mutual information: $I(X_1, X_2; Y) = I(X_1; Y) + I(X_2; Y | X_1)$.

2. **Show** that $D_{JS}(P \| Q) = H\left(\frac{P+Q}{2}\right) - \frac{1}{2}H(P) - \frac{1}{2}H(Q)$.

3. **Derive** the Fisher information for the Bernoulli distribution $p(x|\theta) = \theta^x(1-\theta)^{1-x}$.

4. **Prove** the data processing inequality using the chain rule for mutual information.

5. **Compute** $D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1))$ and verify the VAE KL formula.

---

<div align="center">

**[← Probability & Statistics](../02_probability_and_statistics/)** | **[Next: Optimization Methods →](../04_optimization_methods/)**

</div>
