"""
Evaluate Separation Index on Trained Models

This script evaluates the Separation Index (SI) on trained MAD-VAE models.
It supports both clean-trained (baseline) and adversarially trained models.

Key Features:
    - Uses deterministic latent means (mu) instead of sampled z
    - Supports adversarial attack evaluation (FGSM, R-FGSM, MI-FGSM, PGD)
    - Batched SI computation for large datasets
    - Fixed random seed for reproducibility
    - Compatible with both MNIST and CIFAR-10 models
    - Dataset-dependent epsilon values

Usage Examples:
    # Evaluate baseline (clean model)
    python evaluate_si.py \\
        --dataset mnist \\
        --model_type clean \\
        --model_path pretrained_model/mnist_baseline.pt

    # Evaluate adversarially trained model under PGD attack
    python evaluate_si.py \\
        --dataset mnist \\
        --model_type madvae \\
        --model_path pretrained_model/mnist_pgd.pt \\
        --attack pgd

    # Evaluate on clean data (no attack)
    python evaluate_si.py \\
        --dataset cifar \\
        --model_type clean \\
        --model_path pretrained_model/cifar_baseline.pt \\
        --attack none
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet
from utils.dataset import Dataset2
from utils.adversarial import add_adv
from utils.separation_index import separation_index


# ==========================
# arguments
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Separation Index on trained models"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar', 'svhn', 'gtsrb', 'celeb'],
        help='Dataset to evaluate on'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='clean',
        choices=['clean', 'madvae'],
        help='Type of model (clean-trained or adversarially trained)'
    )
    
    parser.add_argument(
        '--attack',
        type=str,
        default='none',
        choices=['fgsm', 'r-fgsm', 'mi-fgsm', 'pgd', 'none'],
        help='Attack type for evaluation (none = clean data)'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to trained model .pt file'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for data loading'
    )
    
    parser.add_argument(
        '--num_samples',
        type=int,
        default=10000,
        help='Number of samples to evaluate (use all if 0)'
    )
    
    # ✅ FIX 3: Epsilon with dataset-dependent default
    parser.add_argument(
        '--epsilon',
        type=float,
        default=None,
        help='Maximum perturbation magnitude (auto-set per dataset if None)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
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
        # ✅ FIX 3: Dataset-dependent epsilon
        if args.epsilon is None:
            args.epsilon = 0.3
    
    elif args.dataset in ['cifar', 'svhn']:
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 10
        args.z_dim = 256
        if args.epsilon is None:
            args.epsilon = 8/255  # ~0.03137
    
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
    """
    Load trained model from checkpoint.
    
    ✅ Supports both clean-trained (baseline) and adversarially trained models.
    ✅ Both use the same MAD-VAE architecture (only training differs).
    """
    
    # Both clean and madvae use the same MAD-VAE architecture
    # The only difference is the training procedure
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
    print(f"Model on device: {device}")
    
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
    
    Args:
        model: PyTorch model (MAD-VAE)
        dataloader: DataLoader providing images and labels
        attack_type: Type of adversarial attack (None for clean)
        epsilon: Perturbation magnitude
    
    Returns:
        features: numpy array of latent features (mu)
        labels: numpy array of labels
    """
    
    all_features = []
    all_labels = []
    
    # Get device from model
    device = next(model.parameters()).device
    
    print(f"Extracting features on device: {device}")
    print(f"Attack type: {attack_type if attack_type else 'clean'}")
    print(f"Epsilon: {epsilon}")
    
    for batch_idx, (image, label) in enumerate(dataloader):
        image = image.to(device)
        label = label.to(device)
        
        # ✅ FIX 1: Generate adversarial attack WITHOUT torch.no_grad()
        # Attacks need gradients to compute adversarial perturbations
        if attack_type is not None and attack_type != 'none':
            _, adv_image = add_adv(
                model,  # Full model (encoder + classifier)
                image,
                label,
                attack_type,
                epsilon=epsilon
            )
            input_image = adv_image
        else:
            input_image = image
        
        # ✅ FIX 1: Only forward pass with torch.no_grad()
        # Extract deterministic latent mean (mu)
        with torch.no_grad():
            _, mu, _, _ = model(input_image)  # output, mu, std, z
        
        all_features.append(mu.cpu().numpy())
        all_labels.append(label.cpu().numpy())
        
        # Progress indicator
        if (batch_idx + 1) % 50 == 0:
            print(f"  Processed {batch_idx + 1} batches")
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted {len(features)} features with dimension {features.shape[1]}")
    
    return features, labels


# ==========================
# save results
# ==========================

def save_results(args, si, features, labels, output_dir):
    """Save evaluation results to JSON file."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create results dictionary
    results = {
        "dataset": args.dataset,
        "model_type": args.model_type,
        "attack": args.attack if args.attack != 'none' else 'clean',
        "epsilon": args.epsilon,
        "num_samples": len(features),
        "feature_dim": features.shape[1],
        "SI": float(si),
        "feature_type": "mu",  # Deterministic latent mean
        "normalization": "none",  # Original SI definition
        "model_path": args.model_path,
        "seed": args.seed,
        "timestamp": str(np.datetime64('now'))
    }
    
    # Save to JSON
    filename = f"{args.dataset}_{args.model_type}_{args.attack}_si.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {filepath}")
    
    return filepath


# ==========================
# main
# ==========================

def main():
    """Main evaluation loop."""
    
    args = parse_args()
    args = set_params(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "=" * 70)
    print("SEPARATION INDEX EVALUATION")
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
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    x = np.load(f'data/xs_{args.dataset}.npy')
    y = np.load(f'data/ys_{args.dataset}.npy')
    print(f"Total samples: {len(x)}")
    
    # Limit number of samples (deterministic selection)
    if args.num_samples > 0 and args.num_samples < len(x):
        indices = np.random.choice(len(x), args.num_samples, replace=False)
        x = x[indices]
        y = y[indices]
        print(f"Using {len(x)} samples (deterministic selection with seed={args.seed})")
    
    # Create dataset and dataloader
    dataset = Dataset2(x, y)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No shuffle for evaluation
        num_workers=0
    )
    print(f"Number of batches: {len(dataloader)}\n")
    
    # Load model
    print("Loading model...")
    model = load_model(args, device)
    
    # Extract latent features
    attack_type = args.attack if args.attack != 'none' else None
    print(f"\nExtracting features with attack: {attack_type if attack_type else 'clean'}")
    print("-" * 50)
    
    features, labels = extract_latent_features(
        model,
        dataloader,
        attack_type=attack_type,
        epsilon=args.epsilon
    )
    
    # Compute Separation Index
    print("\n" + "-" * 50)
    print("Computing Separation Index...")
    
    # Always use batched version for memory safety
    si = separation_index(
        features,
        labels,
        batch_size=None,  # Auto-select
        device=device,
        normalize=False  # No normalization (original definition)
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Dataset:           {args.dataset}")
    print(f"Model:             {args.model_type}")
    print(f"Attack:            {attack_type if attack_type else 'clean'}")
    print(f"Epsilon:           {args.epsilon}")
    print(f"Feature type:      mu (deterministic latent mean)")
    print(f"Normalization:     None (original SI definition)")
    print(f"Number of samples: {len(features)}")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"{'='*50}")
    print(f"Separation Index (SI): {si:.6f}")
    print("=" * 70 + "\n")
    
    # Save results
    results_path = save_results(args, si, features, labels, args.output_dir)
    
    print("\n📌" * 20)
    print("EVALUATION COMPLETE ✅")
    print("📌" * 20)
    
    print("\nNext Steps:")
    print("  1. Compare with other models:")
    print(f"     python evaluate_si.py --dataset {args.dataset} --model_type clean")
    print(f"     python evaluate_si.py --dataset {args.dataset} --model_type madvae")
    
    print("\n  2. Visualize latent space:")
    print(f"     python visualize_latent.py --dataset {args.dataset} --model_type {args.model_type}")
    
    print("\n  3. Check results:")
    print(f"     cat {results_path}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()