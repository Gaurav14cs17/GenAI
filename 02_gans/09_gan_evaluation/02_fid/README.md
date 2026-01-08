# 📏 Fréchet Inception Distance (FID)

<div align="center">

![FID](./images/fid_metric.svg)

*The gold standard for GAN evaluation*

</div>

---

## 🎯 Where & Why Use FID?

### 🌍 Real-World Applications

| Scenario | Why FID? |
|----------|----------|
| **Model Comparison** | Standard metric for benchmarking |
| **Training Monitoring** | Correlates with visual quality |
| **Publication** | Required for GAN papers |
| **Production QA** | Objective quality measure |

### 💡 Why Master FID?

> *"FID is the yardstick by which all GANs are measured. Know it well."*

1. **Industry Standard** — Most widely used metric
2. **Correlates with Quality** — Matches human perception
3. **Uses Real Data** — Reference-based evaluation
4. **Captures Diversity** — Measures distribution match

---

## 📖 Definition

### The FID Formula

$$\text{FID} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where:
- \( \mu_r, \Sigma_r \): mean and covariance of **real** image features
- \( \mu_g, \Sigma_g \): mean and covariance of **generated** image features
- Features extracted from Inception-v3 network (2048-dim)

### Interpretation

**Low FID:** Generated images similar to real images
- Similar visual concepts
- Similar quality and diversity

**High FID:** Generated images different from real images
- Poor quality, missing modes, wrong domain

---

## 📊 Representation Comparison

| Representation | Pros | Cons |
|----------------|------|------|
| **FID (InceptionV3)** | Standard, comparable | Gaussian assumption |
| **Clean-FID** | Resize consistency | Slightly different values |
| **FID-infinity** | No sample variance | Requires many samples |
| **CLIP-FID** | Better semantics | Different features |
| **SwAV-FID** | Self-supervised | Less standardized |

---

## Typical FID Values

<div align="center">

| Model/Dataset | FID | Quality |
|---------------|:---:|---------|
| Perfect match | 0 | Identical distributions |
| StyleGAN2 (FFHQ) | ~3 | Excellent |
| BigGAN (ImageNet) | ~7 | Very good |
| Good GAN | 10-30 | Good |
| Decent GAN | 30-50 | Acceptable |
| Poor GAN | >50 | Poor |

</div>

---

## Practical Considerations

### Sample Size

| Samples | Reliability |
|---------|-------------|
| 1,000 | Unreliable |
| 10,000 | Acceptable |
| 50,000 | **Recommended** |

### Common Mistakes

❌ Using too few samples (<10,000)
❌ Different preprocessing for real vs generated
❌ Comparing FID across different datasets

---

## 📊 Key Equations

<div align="center">

| Concept | Formula |
|---------|---------|
| **FID** | \( \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r\Sigma_g}) \) |
| **Feature dim** | 2048 (Inception pool3) |
| **Min samples** | 10,000-50,000 |

</div>

---

## 📚 References

1. **Heusel, M., et al.** (2017). "GANs Trained by a Two Time-Scale Update Rule." *NeurIPS*. [arXiv:1706.08500](https://arxiv.org/abs/1706.08500)

---

<div align="center">

**[← Back to Inception Score](../01_inception_score/)** | **[Next: Precision & Recall →](../03_precision_recall/)**

</div>
