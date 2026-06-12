# 🎨 Deeploy CV — Deep Learning Models for Computer Vision

> **Synthesizing photorealistic images from noise using Generative Adversarial Networks.**
> Built and fine-tuned DCGAN architectures on a custom 10,000+ image dataset as part of the
> **Google Developer Group | IIT Kanpur** deep learning program.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/DCGAN-GAN_Architecture-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Google_Developer_Group-IIT_Kanpur-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-10K+_Images-orange?style=for-the-badge"/>
</p>


##  Project Overview

This project was completed as part of **Deeploy CV**, a structured deep learning program run by
the **Google Developer Group at IIT Kanpur** (Dec 2024 – Feb 2025).

The goal was to deeply understand and implement **Generative Adversarial Networks** — one of the
most influential architectures in modern deep learning — by building, training, and evaluating
models from scratch rather than using pre-trained weights.

| | |
|---|---|
| **Topic** | Generative AI · Image Synthesis · Feature Extraction |
| **Architecture** | DCGAN + advanced variants |
| **Dataset** | Custom curated — 10,000+ images |
| **Duration** | Dec 2024 – Feb 2025 |
| **Organisation** | Google Developer Group, IIT Kanpur |

---

##  What is a GAN?

A **Generative Adversarial Network** consists of two neural networks locked in competition:

```
Random Noise (z)
      │
      ▼
┌─────────────┐        Fake images         ┌──────────────────┐
│  Generator  │ ────────────────────────▶  │                  │
│     (G)     │                            │  Discriminator   │──▶ Real / Fake?
└─────────────┘                            │       (D)        │
                                           │                  │
Real images ──────────────────────────▶   └──────────────────┘

G tries to fool D.   D tries to catch G.
Training stops when D can no longer tell real from fake.
```

The generator never sees real images directly — it only receives gradient signals from the
discriminator. This adversarial dynamic forces the generator to produce increasingly realistic outputs.

---

##  Architecture — DCGAN

**DCGAN (Deep Convolutional GAN)** replaces the fully-connected layers of the original GAN
with convolutional layers, making it far more stable to train and more effective at capturing
spatial structure in images.

### Generator
```
Latent vector z (100-dim)
        │
        ▼
  Linear → Reshape
        │
  ConvTranspose2d  4×4   → BatchNorm → ReLU
        │
  ConvTranspose2d  8×8   → BatchNorm → ReLU
        │
  ConvTranspose2d  16×16 → BatchNorm → ReLU
        │
  ConvTranspose2d  32×32 → BatchNorm → ReLU
        │
  ConvTranspose2d  64×64             → Tanh
        │
  Generated Image (64×64×3)
```

### Discriminator
```
Real / Fake Image (64×64×3)
        │
  Conv2d → LeakyReLU
        │
  Conv2d → BatchNorm → LeakyReLU
        │
  Conv2d → BatchNorm → LeakyReLU
        │
  Conv2d → BatchNorm → LeakyReLU
        │
  Flatten → Linear → Sigmoid
        │
  P(real) — single probability score
```

### Key Design Choices
| Choice | Reason |
|---|---|
| BatchNorm in Generator | Stabilises training; prevents mode collapse |
| LeakyReLU in Discriminator | Prevents dead neurons; lets gradients flow for fake images |
| Tanh output activation | Normalises output to [-1, 1], matching normalised real images |
| No pooling layers | Strided convolutions learn their own spatial downsampling |
| Adam optimiser (β1=0.5) | Lower momentum helps GAN training converge more stably |

---

##  Dataset

- **Size:** 10,000+ images curated and preprocessed from scratch
- **Preprocessing pipeline:**
  - Resize all images to uniform resolution (64×64)
  - Normalize pixel values to [-1, 1] (matching Tanh output range)
  - Apply augmentations: random horizontal flip, random crop
  - Filter and remove corrupted / low-quality images manually
- **Storage format:** PyTorch `ImageFolder` compatible directory structure

```
dataset/
├── train/
│   └── images/
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
└── val/
    └── images/
        └── ...
```

---

##  Training Details

| Hyperparameter | Value |
|---|---|
| Latent vector size (z_dim) | 100 |
| Image resolution | 64 × 64 |
| Batch size | 128 |
| Learning rate (G and D) | 0.0002 |
| Adam β1 | 0.5 |
| Adam β2 | 0.999 |
| Epochs | 100+ |
| Feature maps (G) | 64 |
| Feature maps (D) | 64 |

### Loss Curves

<!-- Replace with your actual loss plot -->
![Loss Curves](assets/loss_curves.png)

> A healthy GAN shows Generator loss and Discriminator loss oscillating around each other —
> neither collapses. Discriminator loss near 0.5 means it is genuinely uncertain.

---

##  Setup & Training

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/deeploy-cv.git
cd deeploy-cv
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your dataset
Place your images in the structure shown in the Dataset section above, or
update `DATA_DIR` in `config.py` to point to your image folder.

### 5. Train the model
```bash
python train.py
```

Generated samples are saved to `outputs/samples/` every N epochs.
Model checkpoints are saved to `outputs/checkpoints/`.

### 6. Generate new images from a trained checkpoint
```bash
python generate.py --checkpoint outputs/checkpoints/checkpoint_epoch_100.pt --n 64
```

---

##  Project Structure

```
deeploy-cv/
│
├── train.py              # Main training loop — G and D updated alternately
├── generate.py           # Load checkpoint → generate + save image grid
├── model.py              # Generator and Discriminator class definitions
├── dataset.py            # Custom Dataset class + preprocessing transforms
├── config.py             # All hyperparameters in one place
│
├── utils/
│   ├── visualise.py      # Plot loss curves, save image grids
│   └── metrics.py        # FID score computation (optional)
│
├── outputs/
│   ├── samples/          # Generated image grids saved every N epochs
│   ├── checkpoints/      # Model weights (.pt files)
│   └── loss_curves.png   # Training loss plot
│
├── assets/               # README images
├── requirements.txt
└── README.md
```

---

##  Results

### What was achieved
- Visually convincing synthetic image generation across multiple dataset domains
- Consistent output quality and texture realism at 64×64 resolution
- Stable training without mode collapse through careful architectural choices
- Full understanding of the GAN training dynamic from scratch implementation

### Challenges & How They Were Solved

| Challenge | Solution |
|---|---|
| **Mode collapse** (G produces only one type of image) | Added BatchNorm; reduced learning rate; used feature matching |
| **Training instability** (loss explodes or vanishes) | Set Adam β1=0.5; used label smoothing (real labels = 0.9 not 1.0) |
| **Discriminator dominates too early** | Trained G twice per D update when D loss was too low |
| **Checkerboard artefacts** in generated images | Replaced some ConvTranspose with Upsample + Conv layers |
| **Dataset quality inconsistency** | Manual curation pass to remove blurry/corrupted images |

---

## 🔬 Experiments & Variants

Beyond the baseline DCGAN, the following variants were explored:

- **Progressive growing** — train at 8×8, upscale to 16×16, 32×32, 64×64 progressively
- **Conditional GAN** — conditioned generation on class labels for targeted synthesis
- **Different noise distributions** — compared Gaussian vs Uniform latent vectors
- **Spectral Normalisation** — applied to discriminator weights for Lipschitz constraint

---

##  References

- [Original GAN paper — Goodfellow et al., 2014](https://arxiv.org/abs/1406.2661)
- [DCGAN paper — Radford et al., 2015](https://arxiv.org/abs/1511.06434)
- [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [GAN Hacks — Soumith Chintala](https://github.com/soumith/ganhacks)

---

##  Acknowledgements

This project was completed as part of **Deeploy CV**, a computer vision deep learning program
organised by the **Google Developer Group at IIT Kanpur**.

Thanks to the GDG IIT Kanpur team for structuring a curriculum that emphasised building
from scratch over using pre-built pipelines — the hard way turned out to be the right way.

---

##  Author

**Gaurav Tripathi**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [linkedin.com/in/YOUR_PROFILE](https://linkedin.com/in/YOUR_PROFILE)

---

<p align="center">
  <i>Part of a computer vision deep learning program — Google Developer Group, IIT Kanpur</i><br/>
  <i>Dec 2024 – Feb 2025</i>
</p>
