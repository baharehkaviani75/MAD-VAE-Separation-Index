# MAD-VAE: Separation Index Analysis for Adversarial Robustness

Official PyTorch implementation of **MAD-VAE (Manifold-Adaptive Deep Variational Autoencoder)** for evaluating adversarial robustness through latent space analysis using the **Separation Index (SI)**.

This repository provides a complete framework for:

- Training clean and adversarially trained MAD-VAE models
- Generating adversarial examples (FGSM, R-FGSM, MI-FGSM, PGD)
- Measuring latent class separability using the Separation Index (SI)
- Visualizing latent representations with UMAP
- Comparing clean and adversarial latent spaces across multiple datasets

---

## Overview

Deep neural networks are highly vulnerable to adversarial perturbations, which can significantly degrade classification performance while remaining visually imperceptible.

Rather than evaluating robustness only through classification accuracy, this project investigates how adversarial attacks affect the **structure of the latent representation**.

The workflow consists of:

1. Training a clean or adversarially trained MAD-VAE.
2. Extracting deterministic latent representations using the encoder mean (**μ**).
3. Computing the **Separation Index (SI)** to quantify class separability.
4. Visualizing the latent space using **UMAP**.

Using the latent mean (μ) instead of sampled latent vectors ensures deterministic and reproducible evaluations.

---

## Pipeline

```
                Input Images
                      │
                      ▼
        Adversarial Attack (Optional)
      FGSM • R-FGSM • MI-FGSM • PGD
                      │
                      ▼
              MAD-VAE Encoder
                      │
                      ▼
          Deterministic Latent Mean (μ)
                 ┌──────────────┐
                 │              │
                 ▼              ▼
      Separation Index       UMAP
        Quantitative      Visualization
```

---

## Features

- Deterministic latent representation using encoder mean (μ)
- Separation Index implementation with batched computation
- Support for clean-trained and adversarially trained models
- Multiple adversarial attacks:
  - FGSM
  - R-FGSM
  - MI-FGSM
  - PGD
- Automatic dataset-dependent attack strength (epsilon)
- GPU acceleration
- Reproducible experiments with fixed random seeds
- Publication-quality latent space visualizations

---

## Repository Structure

```
MAD-VAE-Separation-Index/
│
├── models/
│   ├── madvae_mnist.py
│   └── madvae_resnet.py
│
├── utils/
│   ├── adversarial.py
│   ├── dataset.py
│   ├── loss_function.py
│   ├── scheduler.py
│   └── separation_index.py
│
├── train/
│   ├── train_clean.py
│   └── train_madvae.py
│
├── evaluation/
│   └── evaluate_si.py
│
├── visualization/
│   └── visualize_latent.py
│
├── checkpoints/
├── data/
├── results/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/baharehkaviani75/MAD-VAE-Separation-Index.git

cd MAD-VAE-Separation-Index
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Supported Datasets

The framework currently supports:

| Dataset | Classes | Image Size |
|---------|---------|------------|
| MNIST | 10 | 28 × 28 |
| CIFAR-10 | 10 | 32 × 32 |
| SVHN | 10 | 32 × 32 |
| GTSRB | 43 | 32 × 32 |
| CelebA (Binary) | 2 | 32 × 32 |

Datasets should be stored as NumPy arrays:

```
data/
├── xs_mnist.npy
├── ys_mnist.npy
├── xs_cifar.npy
├── ys_cifar.npy
...
```

---

## Training

### Clean Training

```bash
python train/train_clean.py --dataset mnist
```

### Adversarial Training

```bash
python train/train_madvae.py \
    --dataset mnist \
    --attack pgd
```

---

## Separation Index Evaluation

Evaluate latent class separability:

```bash
python evaluation/evaluate_si.py \
    --dataset mnist \
    --model_type madvae \
    --model_path checkpoints/mnist/madvae_pgd.pt \
    --attack pgd
```

Example output:

```
Dataset: MNIST
Model: MAD-VAE
Attack: PGD

Feature Representation : μ

Separation Index (SI): 0.842351
```

Evaluation results are automatically saved as JSON files.

---

## Latent Space Visualization

Generate UMAP visualizations of the latent space:

```bash
python visualization/visualize_latent.py \
    --dataset mnist \
    --model_type madvae \
    --model_path checkpoints/mnist/madvae_pgd.pt \
    --attack pgd
```

The resulting figures are saved in:

```
results/
```

---

## Methodology

Instead of using stochastic latent samples

```
z = μ + σϵ
```

this implementation extracts the deterministic latent representation

```
z = μ
```

to ensure stable and reproducible Separation Index measurements.

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@inproceedings{kaviani2023density,
  title={Density Estimation Helps Adversarial Robustness},
  author={Kaviani Baghbaderani, Bahareh},
  booktitle={13th International Conference on Computer and Knowledge Engineering (ICCKE)},
  year={2023}
}
```

---

## License

This project is released under the MIT License.

