# MAD-VAE: Separation Index Analysis for Adversarial Robustness

Official implementation for evaluating adversarial robustness through latent space analysis using **MAD-VAE (Manifold-Adaptive Deep Variational Autoencoder)** and the **Separation Index (SI)**.

This repository provides:
- MAD-VAE models for robust representation learning
- Adversarial attack evaluation
- Latent space visualization using UMAP
- Separation Index computation for measuring class separability


## Overview

Deep neural networks are vulnerable to adversarial perturbations that can significantly alter their predictions.

This project investigates robustness from a latent representation perspective:

1. Train clean and adversarially trained models.
2. Extract deterministic latent representations (μ).
3. Evaluate latent class separation using Separation Index (SI).
4. Analyze the effect of adversarial attacks on latent manifolds.


The main pipeline is:


Input Image
|
|
Adversarial Attack
(FGSM / R-FGSM / MI-FGSM / PGD)
|
|
MAD-VAE Encoder
|
|
Latent Representation μ
|
|
+----------------+
| |
SI Evaluation UMAP Visualization
| |
+----------------+



## Features

- Deterministic latent representation using encoder mean (μ)
- Support for clean-trained and adversarially trained models
- Adversarial attack evaluation:
    - FGSM
    - R-FGSM
    - MI-FGSM
    - PGD

- Dataset-dependent perturbation strength (epsilon)
- GPU acceleration
- Reproducible experiments with fixed random seeds
- Batched Separation Index computation
- Latent space visualization with UMAP


## Repository Structure


MAD-VAE-Separation-Index/

├── models/
│ ├── madvae_mnist.py
│ └── madvae_resnet.py
│
├── utils/
│ ├── dataset.py
│ ├── adversarial.py
│ ├── separation_index.py
│ ├── loss_function.py
│ └── scheduler.py
│
├── train/
│ ├── train_madvae.py
│ └── train_clean.py
│
├── evaluation/
│ └── evaluate_si.py
│
├── visualization/
│ └── visualize_latent.py
│
├── data/
├── checkpoints/
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore



# Installation

Clone the repository:

```bash
git clone https://github.com/your_username/MAD-VAE-Separation-Index.git

cd MAD-VAE-Separation-Index

Install dependencies:

pip install -r requirements.txt
Requirements
Python >= 3.9
PyTorch >= 2.0
CUDA recommended
Dataset Preparation

The repository expects preprocessed numpy datasets:

data/

├── xs_mnist.npy
├── ys_mnist.npy

├── xs_cifar.npy
├── ys_cifar.npy

...

Each dataset contains:

xs_*.npy: images
ys_*.npy: labels

Supported datasets:

Dataset	Classes	Image Size
MNIST	10	28×28
CIFAR-10	10	32×32
SVHN	10	32×32
GTSRB	43	32×32
CelebA subset	2	32×32
Training
Clean Training

Example:

python train/train_clean.py \
--dataset mnist
MAD-VAE Adversarial Training

Example:

python train/train_madvae.py \
--dataset mnist \
--attack pgd
Separation Index Evaluation

Evaluate latent class separation:

python evaluation/evaluate_si.py \
--dataset mnist \
--model_type madvae \
--model_path checkpoints/mnist/madvae_pgd.pt \
--attack pgd

Example output:

Dataset:           MNIST
Model:             madvae
Attack:            pgd

Feature type:      latent mean (μ)

Separation Index:  0.842351

Results are saved as:

results/si/
Latent Space Visualization

Generate UMAP visualization:

python visualization/visualize_latent.py \
--dataset cifar \
--model_type madvae \
--model_path checkpoints/cifar/madvae_pgd.pt \
--attack pgd

Output:

results/figures/

cifar_madvae_pgd_latent_mu_umap.png
Adversarial Attacks

Supported attacks:

Attack	Description
FGSM	Fast Gradient Sign Method
R-FGSM	Randomized FGSM
MI-FGSM	Momentum Iterative FGSM
PGD	Projected Gradient Descent

Example:

--attack pgd
--epsilon 0.031
Methodology

The latent representation is extracted as:

z=μ(x)

Instead of sampling:

z=μ+σϵ

the deterministic latent mean is used to ensure reproducible SI measurement.

Citation

If you use this repository, please cite:

@inproceedings{density2023,
title={Density Estimation Helps Adversarial Robustness},
author={Kaviani Baghbaderani, Bahareh},
booktitle={13th International Conference on Computer and Knowledge Engineering},
year={2023}
}
License

This project is released for research purposes.