"""
Baseline Training Script

This script trains the MAD-VAE architecture WITHOUT adversarial example generation.
It serves as the baseline for Separation Index (SI) comparison in the paper.

The model is trained on clean data only, and the latent space is learned
through reconstruction and classification losses simultaneously.

Key Architectural Design:
    - Encoder maps input → latent space (z)
    - Classifier operates on latent space (z), NOT raw images
    - Classification loss is computed on z, matching MAD-VAE architecture
    
Baseline = Clean-data MAD-VAE baseline for latent separation comparison.

Comparison with MAD-VAE:
    - Same architecture (MAD-VAE)
    - Same classifier on latent space (z)
    - Same reconstruction loss on clean images
    - Difference: Input is clean (not adversarial)

After training, use evaluate_si.py to compute the Separation Index.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.data import DataLoader
from models.madvae_mnist import MADVAEMNIST
from models.madvae_resnet import MADVAEResNet
from utils.loss_function import recon_loss_function, classification_loss
from utils.dataset import Dataset2
from utils.scheduler import MinExponentialLR


# ==========================
# arguments
# ==========================

def parse_args():
    parser = argparse.ArgumentParser(
        description="MAD-VAE Baseline Training (without adversarial attacks)"
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=['mnist', 'cifar', 'svhn', 'gtsrb', 'celeb'],
        help='Dataset to train on'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Training batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    # ✅ FIX 1: z_dim default=None
    parser.add_argument(
        '--z_dim',
        type=int,
        default=None,
        help='Latent space dimension (auto-set per dataset if None)'
    )
    
    parser.add_argument(
        '--h_dim',
        type=int,
        default=512,
        help='Hidden layer dimension'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for Adam optimizer'
    )
    
    parser.add_argument(
        '--closs_weight',
        type=float,
        default=0.5,
        help='Weight for classification loss (alpha in paper)'
    )
    
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='Beta parameter for KL divergence in VAE'
    )
    
    parser.add_argument(
        '--model_dir',
        type=str,
        default='pretrained_model',
        help='Directory to save trained models and config'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--save_metric',
        type=str,
        default='total',
        choices=['total', 'recon', 'classification'],
        help='Metric to use for saving best model'
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
        # ✅ FIX 1: Only set if None
        if args.z_dim is None:
            args.z_dim = 128
    
    elif args.dataset in ['cifar', 'svhn']:
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 10
        if args.z_dim is None:
            args.z_dim = 256
    
    elif args.dataset == 'gtsrb':
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 43
        if args.z_dim is None:
            args.z_dim = 256
    
    elif args.dataset == 'celeb':
        args.image_channels = 3
        args.image_size = 32
        args.num_classes = 2
        if args.z_dim is None:
            args.z_dim = 256
    
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
# device setup
# ==========================

def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


# ==========================
# model
# ==========================

def init_model(args, device):
    """Initialize MAD-VAE model based on dataset."""
    
    if args.dataset == 'mnist':
        model = MADVAEMNIST(args)
    else:
        model = MADVAEResNet(args)
    
    model = model.to(device)
    model.train()
    
    print(f"Model initialized on {device}")
    return model


# ==========================
# save config
# ==========================

def save_config(args, model_dir):
    """Save training configuration to JSON file."""
    
    config = {
        'version': 'v1.0',
        'repository': 'MADVAE-SI',
        'dataset': args.dataset,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'z_dim': args.z_dim,
        'h_dim': args.h_dim,
        'lr': args.lr,
        'closs_weight': args.closs_weight,
        'beta': args.beta,
        'seed': args.seed,
        'save_metric': args.save_metric,
        'image_channels': args.image_channels,
        'image_size': args.image_size,
        'num_classes': args.num_classes,
        'model_type': 'baseline',
        # ✅ FIX 2: Better description
        'description': 'Clean-data MAD-VAE baseline for latent separation comparison',
        'device': str(get_device()),
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'architectural_note': 'Classifier operates on latent space (z), not raw images'
    }
    
    config_path = os.path.join(model_dir, f'{args.dataset}_baseline_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuration saved to {config_path}")
    return config_path


# ==========================
# save history
# ==========================

def save_history(history, args, model_dir):
    """Save training history to JSON file."""
    
    history_path = os.path.join(model_dir, f'{args.dataset}_baseline_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to {history_path}")
    return history_path


# ==========================
# training
# ==========================

def train_epoch(args, dataloader, model, optimizer, device, epoch):
    """
    Train for one epoch on clean data (no adversarial examples).
    
    Architecture Flow:
        clean image → MAD-VAE Encoder → z → Classifier → classification loss
        
    Classification loss is computed on z (latent space), matching MAD-VAE.
    
    Args:
        epoch: Current epoch number (for KL annealing)
    
    Returns:
        tuple: (total_loss, recon_loss, kl_loss, cls_loss)
    """
    
    total_losses = []
    recon_losses = []
    kl_losses = []
    cls_losses = []
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with clean data (no adversarial attack)
        output, mu, std, z = model(images)
        
        # Define distribution from VAE encoder
        distribution = Normal(mu, std)
        
        # Reconstruction loss (including KL divergence)
        r_loss, img_recon, kld = recon_loss_function(
            output,
            images,
            distribution,
            step=epoch,
            beta=args.beta
        )
        
        # Classification loss on latent space (z)
        # This matches MAD-VAE architecture: classifier operates on z
        c_loss = classification_loss(
            z,
            labels,
            model.classifier
        )
        
        # Total loss: reconstruction + alpha * classification
        loss = r_loss + args.closs_weight * c_loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Store losses
        total_losses.append(loss.item())
        recon_losses.append(img_recon.item())
        kl_losses.append(kld.item())
        cls_losses.append(c_loss.item())
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)} | "
                  f"Total: {loss.item():.4f} | "
                  f"Recon: {img_recon.item():.4f} | "
                  f"KL: {kld.item():.4f} | "
                  f"Cls: {c_loss.item():.4f}")
    
    return (
        np.mean(total_losses),
        np.mean(recon_losses),
        np.mean(kl_losses),
        np.mean(cls_losses)
    )


# ==========================
# main
# ==========================

def main():
    """Main training loop."""
    
    # Parse arguments
    args = parse_args()
    args = set_params(args)
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Print training configuration
    print("\n" + "=" * 70)
    print("BASELINE TRAINING (Clean-data MAD-VAE)")
    print("=" * 70)
    print(f"Dataset:           {args.dataset}")
    print(f"Epochs:            {args.epochs}")
    print(f"Batch size:        {args.batch_size}")
    print(f"Latent dim (z):    {args.z_dim}")
    print(f"Hidden dim (h):    {args.h_dim}")
    print(f"Learning rate:     {args.lr}")
    print(f"Alpha (cls weight):{args.closs_weight}")
    print(f"Beta (KL weight):  {args.beta}")
    print(f"Seed:              {args.seed}")
    print(f"Save metric:       {args.save_metric}")
    print(f"Device:            {device}")
    print(f"Model dir:         {args.model_dir}")
    print("\nArchitecture:")
    print("  Input → Encoder → z → Classifier → Loss")
    print("  Classification on latent space (z)")
    print("=" * 70 + "\n")
    
    # Load data
    print(f"Loading {args.dataset} dataset...")
    x = np.load(f'data/xs_{args.dataset}.npy')
    y = np.load(f'data/ys_{args.dataset}.npy')
    print(f"Data shape: {x.shape}, Labels shape: {y.shape}")
    
    # Create dataset and dataloader
    dataset = Dataset2(x, y)
    
    # Set pin_memory for faster GPU transfer
    pin_memory = True if device.type == 'cuda' else False
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    print(f"Number of batches: {len(dataloader)}\n")
    
    # Initialize model
    print("Initializing model...")
    model = init_model(args, device)
    
    # Save configuration
    config_path = save_config(args, args.model_dir)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    
    # Initialize learning rate scheduler
    scheduler = MinExponentialLR(
        optimizer,
        gamma=0.998,
        minimum=1e-5
    )
    
    # ✅ FIX 3: Initialize history
    history = {
        "epoch": [],
        "total": [],
        "recon": [],
        "kl": [],
        "cls": []
    }
    
    # Training loop
    best_metric_value = float('inf')
    best_epoch = 0
    best_total = float('inf')
    best_recon = float('inf')
    best_kl = float('inf')
    best_cls = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 50)
        
        # Ensure model is in training mode
        model.train()
        
        # Train one epoch
        total_loss, recon_loss, kl_loss, cls_loss = train_epoch(
            args,
            dataloader,
            model,
            optimizer,
            device,
            epoch
        )
        
        # ✅ FIX 3: Store history
        history["epoch"].append(epoch)
        history["total"].append(total_loss)
        history["recon"].append(recon_loss)
        history["kl"].append(kl_loss)
        history["cls"].append(cls_loss)
        
        # Print epoch summary (consistent with train_madvae.py)
        print(f"\nEpoch {epoch}")
        print(f"  Total = {total_loss:.5f}")
        print(f"  Recon = {recon_loss:.5f}")
        print(f"  KL = {kl_loss:.5f}")
        print(f"  Cls = {cls_loss:.5f}")
        print(f"  LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Update learning rate
        scheduler.step()
        
        # Determine metric for saving best model (consistent with MAD-VAE)
        if args.save_metric == 'total':
            current_metric = total_loss
            metric_name = 'total'
        elif args.save_metric == 'recon':
            current_metric = recon_loss
            metric_name = 'reconstruction'
        elif args.save_metric == 'classification':
            current_metric = cls_loss
            metric_name = 'classification'
        else:
            current_metric = total_loss
            metric_name = 'total'
        
        # Save best model based on chosen metric (consistent with MAD-VAE)
        if current_metric < best_metric_value:
            best_metric_value = current_metric
            best_epoch = epoch
            best_total = total_loss
            best_recon = recon_loss
            best_kl = kl_loss
            best_cls = cls_loss
            
            torch.save(
                model.state_dict(),
                f'{args.model_dir}/{args.dataset}_baseline.pt'
            )
            print(f"  ✅ Best model saved ({metric_name} loss: {best_metric_value:.5f})")
        
        # ✅ FIX 4: Save checkpoint with zero-padded epoch number
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                f'{args.model_dir}/{args.dataset}_baseline_epoch_{epoch:03d}.pt'
            )
            print(f"  💾 Checkpoint saved: epoch_{epoch:03d}.pt")
    
    # Save final model (last epoch)
    torch.save(
        model.state_dict(),
        f'{args.model_dir}/{args.dataset}_baseline_last.pt'
    )
    print(f"\n  💾 Final model saved: {args.model_dir}/{args.dataset}_baseline_last.pt")
    
    # ✅ FIX 3: Save history
    history_path = save_history(history, args, args.model_dir)
    
    # Print completion message
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE ✅")
    print("=" * 70)
    print("\nBaseline model successfully trained.")
    print("This model represents the latent space learned")
    print("without adversarial training and can be directly")
    print("compared with the adversarially trained MAD-VAE")
    print("using evaluate_si.py.\n")
    
    print("Architectural Summary:")
    print("  ✅ Encoder: MAD-VAE Encoder")
    print("  ✅ Latent: z (latent space)")
    print("  ✅ Classifier: operates on z")
    print("  ✅ Input: clean images")
    print("  ✅ Classification Loss: on z")
    print("  ✅ Warmup: Not applicable (clean training only)\n")
    
    print(f"Best model (epoch {best_epoch}):")
    print(f"  Path: {args.model_dir}/{args.dataset}_baseline.pt")
    print(f"  Total Loss: {best_total:.5f}")
    print(f"  Recon Loss: {best_recon:.5f}")
    print(f"  KL Loss:    {best_kl:.5f}")
    print(f"  Cls Loss:   {best_cls:.5f}")
    print(f"  Metric:     {args.save_metric} = {best_metric_value:.5f}")
    
    print(f"\nFinal model (last epoch):")
    print(f"  Path: {args.model_dir}/{args.dataset}_baseline_last.pt")
    print(f"Configuration: {config_path}")
    print(f"History: {history_path}")
    
    print("\n" + "📌" * 20)
    print("NEXT STEPS:")
    print("📌" * 20)
    
    print("\n1️⃣ Evaluate Separation Index (Baseline):")
    print(f"   python evaluate_si.py \\")
    print(f"       --dataset {args.dataset} \\")
    print(f"       --model_type baseline \\")
    print(f"       --model_path {args.model_dir}/{args.dataset}_baseline.pt")
    
    print("\n2️⃣ Visualize Latent Space (Baseline):")
    print(f"   python visualize_latent.py \\")
    print(f"       --dataset {args.dataset} \\")
    print(f"       --model_type baseline \\")
    print(f"       --model_path {args.model_dir}/{args.dataset}_baseline.pt")
    
    print("\n3️⃣ Train the Adversarially Trained MAD-VAE:")
    print(f"   python train_madvae.py --dataset {args.dataset} --attack pgd")
    
    print("\n4️⃣ Evaluate Separation Index (Adversarially Trained):")
    print(f"   python evaluate_si.py \\")
    print(f"       --dataset {args.dataset} \\")
    print(f"       --model_type madvae \\")
    print(f"       --model_path {args.model_dir}/{args.dataset}_pgd.pt")
    
    print("\n5️⃣ Compare SI Scores:")
    print(f"   SI_baseline vs SI_madvae")
    print(f"   Adversarial training should improve latent class separation")
    print(f"   and robustness, reflected by higher or comparable SI scores.")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()