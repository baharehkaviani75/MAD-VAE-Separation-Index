"""
Visualize Latent Space with UMAP

This script visualizes the latent space of trained MAD-VAE models using UMAP.
It supports both clean-trained (baseline) and adversarially trained models.

Key Features:
    - Uses deterministic latent means (mu) instead of sampled z
    - Supports adversarial attack evaluation (FGSM, R-FGSM, MI-FGSM, PGD)
    - Dataset-dependent epsilon values
    - Fixed random seed for reproducibility
    - High-quality visualizations for papers

Usage Examples:
    # Visualize baseline model on clean data
    python visualize_latent.py \\
        --dataset mnist \\
        --model_type clean \\
        --model_path pretrained_model/mnist_baseline.pt

    # Visualize adversarially trained model under PGD attack
    python visualize_latent.py \\
        --dataset mnist \\
        --model_type madvae \\
        --model_path pretrained_model/mnist_pgd.pt \\
        --attack pgd
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import umap
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet
from utils.dataset import Dataset2
from utils.adversarial import add_adv


# ==========================
# arguments
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize latent space with UMAP"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar', 'svhn', 'gtsrb', 'celeb'],
        help='Dataset to visualize'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='clean',
        choices=['clean', 'madvae'],
        help='clean-trained or adversarially trained model'
    )
    
    parser.add_argument(
        '--attack',
        type=str,
        default='none',
        choices=['fgsm', 'r-fgsm', 'mi-fgsm', 'pgd', 'none'],
        help='Attack type for visualization (none = clean data)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model .pt file'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5000,
        help='Number of samples to visualize'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loading'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=None,
        help='Attack epsilon (auto selected by dataset if None)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='visualizations',
        help='Directory to save plots'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    # UMAP parameters
    parser.add_argument(
        '--n_neighbors',
        type=int,
        default=30,
        help='UMAP n_neighbors parameter'
    )
    
    parser.add_argument(
        '--min_dist',
        type=float,
        default=0.3,
        help='UMAP min_dist parameter'
    )
    
    parser.add_argument(
        '--max_classes',
        type=int,
        default=10,
        help='Maximum number of classes to display'
    )
    
    return parser.parse_args()


# ==========================
# dataset parameters
# ==========================

def set_params(args):
    """Set dataset-specific parameters based on dataset name."""
    
    if args.dataset == 'mnist':
        args.image_channels = 1
        args.image_size = 28
        args.num_classes = 10
        args.z_dim = 128
        if args.epsilon is None:
            args.epsilon = 0.3
    
    elif args.dataset in ['cifar', 'svhn']:
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 10
        args.z_dim = 256
        if args.epsilon is None:
            args.epsilon = 8/255
    
    elif args.dataset == 'gtsrb':
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 43
        args.z_dim = 256
        if args.epsilon is None:
            args.epsilon = 8/255
    
    elif args.dataset == 'celeb':
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 2
        args.z_dim = 256
        if args.epsilon is None:
            args.epsilon = 8/255
    
    return args


# ==========================
# reproducibility
# ==========================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================
# model
# ==========================

def load_model(args, device):
    """Load trained model from checkpoint."""
    
    if args.dataset == 'mnist':
        model = MADVAEMNIST(args)
    else:
        model = MADVAEResNet(args)
    
    state_dict = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {args.model_path}")
    print(f"Model type: {args.model_type}")
    
    return model


# ==========================
# extract latent features
# ==========================

def extract_latent_features(model, dataloader, attack_type=None, epsilon=0.3):
    """
    Extract deterministic latent features (mu) from the model.
    
    ✅ Uses mean latent (mu) instead of sampled z for deterministic results
    ✅ Attack is applied to the FULL model (encoder + classifier)
    ✅ NO torch.no_grad() around attack generation (needs gradients)
    ✅ torch.no_grad() only for final forward pass
    """
    
    all_features = []
    all_labels = []
    
    device = next(model.parameters()).device
    
    print(f"\nExtracting features on device: {device}")
    print(f"Attack type: {attack_type if attack_type else 'clean'}")
    print(f"Epsilon: {epsilon}")
    print("-" * 50)
    
    for batch_idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        
        # Generate adversarial attack WITHOUT torch.no_grad()
        if attack_type is not None and attack_type != 'none':
            _, adv_image = add_adv(
                model,
                image,
                label,
                attack_type,
                epsilon=epsilon
            )
            input_image = adv_image
        else:
            input_image = image
        
        # Only forward pass with torch.no_grad()
        with torch.no_grad():
            _, mu, _, _ = model(input_image)
        
        all_features.append(mu.cpu().numpy())
        all_labels.append(label.cpu().numpy())
        
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} batches")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nExtracted {len(features)} features with dimension {features.shape[1]}")
    
    return features, labels


# ==========================
# visualization
# ==========================

def visualize_latent(
    features,
    labels,
    title,
    output_path,
    max_classes=10,
    n_neighbors=30,
    min_dist=0.3,
    seed=42
):
    """
    Visualize latent features using UMAP.
    
    ✅ Uses seed from args for reproducibility
    ✅ Deterministic class selection (first max_classes)
    """
    
    print("\nReducing to 2D with UMAP...")
    print(f"  n_neighbors: {n_neighbors}")
    print(f"  min_dist: {min_dist}")
    print(f"  random_state: {seed}")
    
    # ✅ FIX 2: Use seed from args
    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='euclidean'
    )
    embedding = reducer.fit_transform(features)
    
    print(f"UMAP embedding shape: {embedding.shape}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # ✅ FIX 3: Deterministic class selection (first max_classes)
    unique_labels = np.unique(labels)
    if len(unique_labels) > max_classes:
        unique_labels = unique_labels[:max_classes]
        print(f"Displaying {len(unique_labels)} out of {np.unique(labels).shape[0]} classes")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=[colors[i]],
            label=f'Class {int(label)}',
            alpha=0.6,
            s=5,
            edgecolors='none'
        )
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {output_path}")


# ==========================
# main
# ==========================

def main():
    """Main visualization loop."""
    
    args = parse_args()
    args = set_params(args)
    
    set_seed(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("LATENT SPACE VISUALIZATION")
    print("=" * 70)
    print(f"Dataset:           {args.dataset}")
    print(f"Model type:        {args.model_type}")
    print(f"Attack:            {args.attack if args.attack != 'none' else 'clean'}")
    print(f"Epsilon:           {args.epsilon}")
    print(f"Model path:        {args.model_path}")
    print(f"Num samples:       {args.num_samples if args.num_samples > 0 else 'all'}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Seed:              {args.seed}")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 70 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    x = np.load(f'data/xs_{args.dataset}.npy')
    y = np.load(f'data/ys_{args.dataset}.npy')
    print(f"Total samples: {len(x)}")
    
    if args.num_samples > 0 and args.num_samples < len(x):
        indices = np.random.choice(len(x), args.num_samples, replace=False)
        x = x[indices]
        y = y[indices]
        print(f"Using {len(x)} samples (deterministic selection with seed={args.seed})")
    
    dataset = Dataset2(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    print(f"Number of batches: {len(dataloader)}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args, device)
    
    # Extract features
    attack_type = args.attack if args.attack != 'none' else None
    features, labels = extract_latent_features(
        model,
        dataloader,
        attack_type=attack_type,
        epsilon=args.epsilon
    )
    
    # ✅ FIX 4: Better filename
    attack_label = attack_type if attack_type else 'clean'
    
    if attack_label == 'clean':
        filename = f"{args.dataset}_{args.model_type}_latent_mu_umap.png"
    else:
        filename = f"{args.dataset}_{args.model_type}_{attack_label}_latent_mu_umap.png"
    
    title = (
        f"{args.dataset.upper()} | "
        f"{args.model_type.upper()} | "
        f"{attack_label.upper()} | "
        "Latent Mean (μ)"
    )
    
    output_path = os.path.join(args.output_dir, filename)
    
    visualize_latent(
        features,
        labels,
        title,
        output_path,
        max_classes=args.max_classes,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        seed=args.seed
    )
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE ✅")
    print("=" * 70)
    print(f"\nOutput: {output_path}")
    
    print("\n📌" * 20)
    print("NEXT STEPS:")
    print("📌" * 20)
    
    print("\n  1. Evaluate Separation Index:")
    print(f"     python evaluate_si.py --dataset {args.dataset} --model_type {args.model_type}")
    
    print("\n  2. Compare visualizations:")
    print(f"     python visualize_latent.py --dataset {args.dataset} --model_type clean")
    print(f"     python visualize_latent.py --dataset {args.dataset} --model_type madvae")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()