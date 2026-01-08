# 🔲 Convolutional Neural Networks for Generative AI

<div align="center">

![CNNs in Generative AI](images/cnn_genai.svg)

*The backbone of image-based generative models*

</div>

---

## 📖 Introduction

Convolutional Neural Networks (CNNs) are the backbone of image-based generative models. From the discriminator in GANs to the U-Net architecture in diffusion models, understanding convolutions and their properties is essential for generative AI.

This guide covers CNN fundamentals with rigorous mathematical treatment and connections to generative modeling architectures.

---

## 🎯 Where and Why Use CNNs in Generative AI

### Where It's Used

| Application | CNN Components | Example Models |
|-------------|---------------|----------------|
| **Image Generation** | Conv, TransposedConv, Residual blocks | DCGAN, StyleGAN, ProGAN |
| **Diffusion Models** | U-Net architecture, skip connections | DDPM, Stable Diffusion |
| **Image-to-Image Translation** | Encoder-decoder, U-Net | Pix2Pix, CycleGAN |
| **Super Resolution** | Transposed conv, sub-pixel conv | SRGAN, ESRGAN |
| **VAE for Images** | Conv encoder/decoder | Convolutional VAE |
| **Discriminators** | Strided convolutions, spectral norm | All image GANs |
| **Feature Extraction** | Pretrained CNNs (VGG, ResNet) | Perceptual loss, FID |
| **Neural Style Transfer** | Conv features, Gram matrices | Artistic style transfer |
| **Video Generation** | 3D convolutions, temporal convs | Video diffusion models |

### Why It's Essential

1. **Image Generation Requires CNNs:**
   - Images have spatial structure that CNNs exploit
   - Translation equivariance: pattern recognition regardless of position
   - Parameter efficiency: weight sharing across spatial locations

2. **U-Net Is Everywhere:**
   - Standard architecture for diffusion models (Stable Diffusion)
   - Used in image segmentation, inpainting, translation
   - Skip connections preserve fine details during generation

3. **Understanding Transposed Convolutions:**
   - How generators upsample from latent space to images
   - Checkerboard artifacts and how to avoid them
   - Alternative: resize + conv, sub-pixel convolution

4. **Perceptual Loss Requires CNN Knowledge:**
   - VGG features for measuring perceptual similarity
   - Essential for training high-quality generators
   - Used in VAEs, GANs, diffusion models

5. **Discriminator Design:**
   - Effective discriminators use strided convolutions
   - Spectral normalization for training stability
   - PatchGAN: classify image patches instead of whole image

### What Happens Without This Knowledge

- ❌ Can't understand or modify image generation architectures
- ❌ Can't debug resolution/spatial issues in generated images
- ❌ Can't implement U-Net for diffusion models
- ❌ Can't use perceptual loss effectively
- ❌ Can't understand why generated images have artifacts
- ❌ Can't design efficient generator/discriminator architectures

### Key CNN Architectures in GenAI

| Architecture | Key Features | Used In |
|--------------|-------------|---------|
| **U-Net** | Encoder-decoder + skip connections | Diffusion models, Pix2Pix |
| **ResNet Blocks** | Skip connections, residual learning | StyleGAN, modern generators |
| **Progressive Growing** | Start small, add layers | ProGAN, StyleGAN |
| **PatchGAN** | Discriminate patches, not whole image | Pix2Pix, CycleGAN |
| **StyleGAN Generator** | Mapping network + AdaIN | High-quality face generation |

### Critical Concepts for GenAI

1. **Receptive Field:** How much input area affects each output pixel
2. **Downsampling:** Strided conv vs pooling trade-offs
3. **Upsampling:** TransposedConv vs resize-conv vs sub-pixel
4. **Normalization:** BatchNorm vs LayerNorm vs InstanceNorm vs GroupNorm
5. **Attention in CNNs:** Self-attention for long-range dependencies

---

## 📊 Representation Comparison

| Representation | Pros | Cons |
|----------------|------|------|
| **Standard Conv** | Local features, translation equivariant | Limited receptive field |
| **Transposed Conv** | Learnable upsampling | Checkerboard artifacts |
| **Dilated Conv** | Large receptive field, no params increase | Gridding artifacts |
| **Depthwise Separable** | Parameter efficient | Slightly less expressive |
| **Deformable Conv** | Adaptive receptive field | More computation |

---

## 1. The Convolution Operation

### 1.1 Continuous Convolution

For functions $f, g: \mathbb{R} \to \mathbb{R}$:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) \, d\tau$$

**Properties:**
1. **Commutativity:** $f * g = g * f$
2. **Associativity:** $f * (g * h) = (f * g) * h$
3. **Distributivity:** $f * (g + h) = f * g + f * h$
4. **Differentiation:** $(f * g)' = f' * g = f * g'$

### 1.2 Discrete Convolution

For sequences $x, w$:

$$(x * w)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot w[n - k]$$

### 1.3 2D Discrete Convolution (Images)

For image $X \in \mathbb{R}^{H \times W}$ and kernel $K \in \mathbb{R}^{k_h \times k_w}$:

$$(X * K)[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[i+m, j+n] \cdot K[m, n]$$

**Note:** In deep learning, we typically use **cross-correlation** (no kernel flip), but call it convolution.

### 1.4 Convolution as Matrix Multiplication

Any linear operation can be written as matrix multiplication. For 1D convolution with kernel $[w_0, w_1, w_2]$ on input $[x_0, x_1, x_2, x_3]$:

$$\begin{bmatrix} y_0 \\ y_1 \end{bmatrix} = \begin{bmatrix} w_0 & w_1 & w_2 & 0 \\ 0 & w_0 & w_1 & w_2 \end{bmatrix} \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ x_3 \end{bmatrix}$$

This is a **Toeplitz matrix** structure.

**Implication:** Convolution is a linear operator with weight sharing.

---

## 2. CNN Building Blocks

### 2.1 Convolutional Layers

**Parameters:**
- Input: $X \in \mathbb{R}^{C_{in} \times H \times W}$
- Kernel: $K \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$
- Bias: $b \in \mathbb{R}^{C_{out}}$

**Output:**
$$Y[c_{out}, i, j] = b[c_{out}] + \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} K[c_{out}, c_{in}, m, n] \cdot X[c_{in}, i+m, j+n]$$

**Parameter count:** $C_{out} \times C_{in} \times k_h \times k_w + C_{out}$

### 2.2 Output Size Calculation

For input size $H$, kernel size $k$, stride $s$, padding $p$:

$$H_{out} = \left\lfloor \frac{H + 2p - k}{s} \right\rfloor + 1$$

**Same padding:** $p = \lfloor k/2 \rfloor$ preserves spatial dimensions (with $s=1$)

### 2.3 Receptive Field

The **receptive field** is the region of input affecting a single output unit.

For $L$ layers with kernel size $k$ and stride 1:
$$r_L = 1 + L(k - 1)$$

For stride $s$ at each layer:
$$r_L = 1 + \sum_{l=1}^{L} (k_l - 1) \prod_{i=1}^{l-1} s_i$$

### 2.4 Pooling Layers

**Max Pooling:**
$$Y[c, i, j] = \max_{m, n \in \text{window}} X[c, i \cdot s + m, j \cdot s + n]$$

**Average Pooling:**
$$Y[c, i, j] = \frac{1}{k^2} \sum_{m, n \in \text{window}} X[c, i \cdot s + m, j \cdot s + n]$$

**Global Average Pooling:** Average over entire spatial dimensions.

---

## 3. Transposed Convolutions (Deconvolutions)

### 3.1 Motivation

For generative models, we need to **upsample**: go from latent space to image space.

### 3.2 Definition

If forward convolution is $y = Cx$ where $C$ is the convolution matrix, then transposed convolution is:

$$x' = C^T y$$

**Output size:**
$$H_{out} = (H_{in} - 1) \cdot s - 2p + k + \text{output\_padding}$$

### 3.3 Checkerboard Artifacts

Transposed convolutions can produce **checkerboard artifacts** when kernel size is not divisible by stride.

**Analysis:** When $k \mod s \neq 0$, some output pixels receive more contributions than others.

**Solutions:**
1. Use $k$ divisible by $s$ (e.g., $k=4$, $s=2$)
2. Use resize-convolution: upsample (bilinear) + conv
3. Use sub-pixel convolution (pixel shuffle)

### 3.4 Sub-Pixel Convolution

Rearrange channels to spatial dimensions:

$$Y = \text{PixelShuffle}(X, r)$$

For $X \in \mathbb{R}^{C \cdot r^2 \times H \times W}$, output $Y \in \mathbb{R}^{C \times rH \times rW}$

**Reference:** Shi et al. (2016). "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network."

---

## 4. Normalization Techniques

### 4.1 Batch Normalization

For mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$:

$$\mu_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m} x_i, \quad \sigma^2_{\mathcal{B}} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})^2$$

$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

**For CNNs:** Normalize per-channel across batch and spatial dimensions.

### 4.2 Layer Normalization

Normalize across channels and spatial dimensions, **per sample**:

$$\mu_i = \frac{1}{CHW}\sum_{c,h,w} x_i[c,h,w]$$

### 4.3 Instance Normalization

Normalize per-channel, per-sample (across spatial only):

$$\mu_{i,c} = \frac{1}{HW}\sum_{h,w} x_i[c,h,w]$$

**Popular in style transfer:** Removes instance-specific style information.

### 4.4 Group Normalization

Divide channels into groups, normalize within groups:

$$\mu_{i,g} = \frac{1}{(C/G) \cdot HW}\sum_{c \in g, h, w} x_i[c,h,w]$$

**Advantage:** Works with any batch size (important for large images, small batches).

### 4.5 Adaptive Instance Normalization (AdaIN)

$$\text{AdaIN}(x, y) = \sigma(y) \left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$$

Transfer statistics from style $y$ to content $x$.

**Used in:** StyleGAN, neural style transfer.

---

## 5. Residual Connections

### 5.1 Residual Block

$$y = F(x, \{W_i\}) + x$$

where $F$ represents stacked convolutions.

### 5.2 Why Residuals Help

**Gradient Flow Analysis:**

Without residuals: $\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1} \frac{\partial x_{i+1}}{\partial x_i}$

With residuals: $x_{l+1} = x_l + F(x_l)$

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_L} \prod_{i=l}^{L-1} \left(1 + \frac{\partial F}{\partial x_i}\right)$$

The "1" term provides a **gradient highway** directly to earlier layers.

### 5.3 Pre-activation ResNet

Original: Conv → BN → ReLU → Conv → BN → Add → ReLU

Pre-activation: BN → ReLU → Conv → BN → ReLU → Conv → Add

**Advantage:** Cleaner gradient paths, better optimization.

---

## 6. Attention Mechanisms in CNNs

### 6.1 Squeeze-and-Excitation (SE)

Channel attention:

$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot \text{GAP}(X)))$$
$$Y = s \odot X$$

where GAP is global average pooling, $\odot$ is channel-wise multiplication.

### 6.2 Self-Attention for Images

Flatten spatial dimensions and apply attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Computational cost:** $O((HW)^2)$ — expensive for high resolution.

### 6.3 Efficient Attention Variants

**Local Attention:** Attend only within windows.

**Axial Attention:** Factorize into row and column attention.

**Linear Attention:** Approximate softmax with kernel trick.

---

## 7. Architectures for Generative AI

### 7.1 Encoder-Decoder Architecture

**Encoder:** Downsample to latent representation
- Strided convolutions or pooling
- Increase channels, decrease spatial

**Decoder:** Upsample to output
- Transposed convolutions or resize-conv
- Decrease channels, increase spatial

### 7.2 U-Net Architecture

Encoder-decoder with **skip connections**:

$$\text{Decoder}_i = \text{Conv}(\text{Concat}(\text{Up}(\text{Decoder}_{i-1}), \text{Encoder}_i))$$

**Why Skip Connections?**
- Preserve fine-grained spatial information
- Help gradient flow
- Essential for image-to-image translation

**Original Paper:** Ronneberger et al. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation."

**Used in:** Diffusion models, image segmentation, pix2pix.

### 7.3 Progressive Growing

Start training at low resolution, progressively add layers:

$$4 \times 4 \to 8 \times 8 \to 16 \times 16 \to \cdots \to 1024 \times 1024$$

**Advantages:**
- Stabilizes training
- Learns coarse-to-fine structure
- Enables high-resolution generation

**Reference:** Karras et al. (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation."

### 7.4 Multi-Scale Architectures

**Feature Pyramid Network (FPN):** Combine features from multiple scales.

**HRNet:** Maintain high-resolution representations throughout.

---

## 8. Dilated (Atrous) Convolutions

### 8.1 Definition

Insert gaps (dilation rate $d$) between kernel elements:

$$(X *_d K)[i, j] = \sum_{m, n} X[i + d \cdot m, j + d \cdot n] \cdot K[m, n]$$

### 8.2 Effective Receptive Field

For kernel size $k$ and dilation $d$:
$$k_{eff} = k + (k-1)(d-1) = d(k-1) + 1$$

### 8.3 Applications

- **Large receptive field** without increasing parameters
- **Multi-scale context** via different dilation rates (ASPP)
- **WaveNet** for audio generation

---

## 9. Depthwise Separable Convolutions

### 9.1 Factorization

Standard conv: $C_{out} \times C_{in} \times k \times k$ parameters

**Depthwise Separable:**
1. **Depthwise:** $C_{in}$ separate $k \times k$ convolutions
2. **Pointwise:** $C_{out} \times C_{in} \times 1 \times 1$ convolution

**Parameter reduction:**
$$\frac{C_{in} \cdot k^2 + C_{in} \cdot C_{out}}{C_{out} \cdot C_{in} \cdot k^2} = \frac{1}{C_{out}} + \frac{1}{k^2}$$

For $k=3$, $C_{out}=256$: ~9× fewer parameters.

### 9.2 Applications

**MobileNet:** Efficient architectures for mobile devices.

**Xception:** Extreme version of Inception using depthwise separable convs.

---

## 10. Equivariance and Invariance

### 10.1 Translation Equivariance

Convolution is **translation equivariant**:

$$T_a(f * g) = (T_a f) * g = f * (T_a g)$$

where $T_a$ is translation by $a$.

**Proof:** $(T_a f * g)(x) = \int f(t-a)g(x-t)dt = \int f(\tau)g(x-a-\tau)d\tau = (f*g)(x-a)$

### 10.2 Building Invariance

**Pooling:** Provides (approximate) translation invariance.

**Global operations:** Global average pooling gives full translation invariance.

### 10.3 Equivariant Neural Networks

Generalize to other symmetries:
- **Rotation equivariance:** G-CNNs
- **Scale equivariance:** Scale-equivariant networks
- **Permutation equivariance:** Graph neural networks

**Reference:** Cohen & Welling (2016). "Group Equivariant Convolutional Networks."

---

## 11. Spectral View of Convolutions

### 11.1 Convolution Theorem

Convolution in spatial domain = multiplication in frequency domain:

$$\mathcal{F}(f * g) = \mathcal{F}(f) \cdot \mathcal{F}(g)$$

where $\mathcal{F}$ is the Fourier transform.

### 11.2 Frequency Analysis of CNNs

**Low-frequency bias:** CNNs tend to learn low frequencies first.

**High-frequency artifacts:** Can indicate overfitting or adversarial vulnerability.

**Fourier Features:** Encode positions with Fourier features to capture high frequencies:

$$\gamma(p) = [\sin(2^0 \pi p), \cos(2^0 \pi p), \ldots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p)]$$

**Reference:** Tancik et al. (2020). "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains."

---

## Key Architectures Summary

| Architecture | Key Feature | Use in GenAI |
|--------------|-------------|--------------|
| VGG | Simple, deep | Perceptual loss |
| ResNet | Residual connections | Feature extraction |
| U-Net | Skip connections | Diffusion models |
| Progressive | Multi-scale training | StyleGAN |
| SE-Net | Channel attention | Improved generators |

---

## References

### Foundational Papers
1. **LeCun, Y., et al.** (1998). "Gradient-Based Learning Applied to Document Recognition." *Proceedings of the IEEE*.
2. **Krizhevsky, A., et al.** (2012). "ImageNet Classification with Deep Convolutional Neural Networks." *NeurIPS*.
3. **He, K., et al.** (2016). "Deep Residual Learning for Image Recognition." *CVPR*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
4. **Ronneberger, O., et al.** (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

### Architecture Advances
5. **Huang, G., et al.** (2017). "Densely Connected Convolutional Networks." *CVPR*. [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)
6. **Hu, J., et al.** (2018). "Squeeze-and-Excitation Networks." *CVPR*. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)
7. **Howard, A., et al.** (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." [arXiv:1704.04861](https://arxiv.org/abs/1704.04861)

### GenAI Applications
8. **Karras, T., et al.** (2018). "Progressive Growing of GANs for Improved Quality, Stability, and Variation." *ICLR*. [arXiv:1710.10196](https://arxiv.org/abs/1710.10196)
9. **Radford, A., et al.** (2016). "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks." *ICLR*. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

---

## Exercises

1. **Derive** the output size formula for a 2D convolution with kernel $k$, stride $s$, padding $p$, and dilation $d$.

2. **Show** that the Toeplitz matrix for 1D convolution with kernel $[1, 2, 1]$ produces Gaussian-like smoothing.

3. **Compute** the receptive field of a network with 5 layers of $3 \times 3$ convolutions with stride 1.

4. **Prove** that convolution is translation equivariant.

5. **Analyze** why transposed convolution with kernel size 3 and stride 2 produces checkerboard artifacts.

---

<div align="center">

**[← PyTorch Basics](../05_pytorch_basics/)** | **[Next: Datasets & Preprocessing →](../07_datasets_and_preprocessing/)**

</div>
