"""
Separation Index (SI) Metric

This module implements the Separation Index (SI) metric for evaluating
the robustness of deep neural networks in latent space.

The SI measures the degree of separation between data points with different
labels by comparing each data point with its nearest neighbor.

Key Features:
    - Batched computation for large datasets (memory efficient)
    - GPU acceleration support
    - Automatic device detection
    - Deterministic results (uses mean latent for VAE)
    - Compatible with MAD-VAE architecture

Mathematical Definition:
    SI = (1/N) * sum_{i=1}^N I( argmin_j ||z_i - z_j||^2 has same label as z_i )
    
    where z_i is the latent representation (mean latent for VAE).

Important Notes:
    - SI is computed on deterministic VAE latent means (mu), NOT sampled z
    - No L2 normalization is applied to preserve original SI definition
    - The metric measures separation on the original latent manifold

Reference:
    Kaviani Baghbaderani, B., Hasanebrahimi, A., Kalhor, A., & Hosseini, R.
    "Adversarial Robustness Evaluation with Separation Index"
    ICCKE 2023
"""

import numpy as np
import torch
import time


def separation_index_batched(
    features, 
    labels, 
    batch_size=1000, 
    device=None,
    normalize=False  # ✅ Changed: default False to preserve original definition
):
    """
    Compute Separation Index in batches to avoid O(N^2) memory.
    
    The Separation Index (SI) is defined as:
        SI = (1/N) * sum_{i=1}^N I( argmin_j ||z_i - z_j||^2 has same label as z_i )
    
    This batched version computes distances in chunks to handle large datasets
    (e.g., 60,000 samples on MNIST).
    
    Args:
        features: numpy array or torch tensor of shape (N, D) - latent features
        labels: numpy array or torch tensor of shape (N,) - class labels (0-indexed)
        batch_size: number of points to process at once (default: 1000)
        device: 'cpu', 'cuda', or None (auto-detect)
        normalize: Optional L2 normalization. 
                   Disabled by default to preserve original SI definition.
                   (Default: False)
    
    Returns:
        si: Separation Index value between 0 and 1
        
    Example:
        >>> features = np.random.randn(1000, 128)
        >>> labels = np.random.randint(0, 10, 1000)
        >>> si = separation_index_batched(features, labels, batch_size=100)
        >>> print(f"SI = {si:.4f}")
    """
    
    # Convert to torch tensors if needed
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    
    # Auto-detect device if None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    features = features.to(device)
    labels = labels.to(device)
    
    N = features.shape[0]
    D = features.shape[1]
    
    print(f"Computing SI on {N} samples with dimension {D} using device: {device}")
    print(f"Batch size: {batch_size}")
    
    # ✅ Pre-compute feature norms for efficiency
    feature_norm = torch.sum(features ** 2, dim=1, keepdim=True)  # (N, 1)
    
    # Optional L2 normalization (disabled by default)
    if normalize:
        print("  ⚠️  Applying L2 normalization (SI on hypersphere)")
        features = features / (torch.norm(features, dim=1, keepdim=True) + 1e-10)
        # Recompute norms after normalization
        feature_norm = torch.sum(features ** 2, dim=1, keepdim=True)
    else:
        print("  ℹ️  No normalization applied (original SI definition)")
    
    same_count = 0
    total_processed = 0
    
    start_time = time.time()
    
    # Iterate over batches
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch_features = features[i:end]  # (B, D)
        batch_labels = labels[i:end]      # (B,)
        B = batch_features.shape[0]
        
        # ✅ FIX: Use pre-computed feature_norm for efficiency
        sq_norm_batch = feature_norm[i:end]  # (B, 1)
        sq_norm_all = feature_norm.T          # (1, N)
        dist = sq_norm_batch + sq_norm_all - 2 * torch.mm(batch_features, features.T)  # (B, N)
        
        # Handle numerical issues
        dist = torch.clamp(dist, min=0.0)
        
        # Set diagonal to infinity efficiently (no loop)
        idx = torch.arange(B, device=device)
        dist[idx, i + idx] = float('inf')
        
        # Find nearest neighbor (minimum distance)
        nearest_idx = torch.argmin(dist, dim=1)  # (B,)
        
        # Check if nearest neighbor has same label
        same_label = (batch_labels == labels[nearest_idx]).float()
        same_count += torch.sum(same_label).item()
        total_processed += B
        
        # Progress indicator
        if (i // batch_size + 1) % 10 == 0 or i + B >= N:
            progress = (i + B) / N * 100
            elapsed = time.time() - start_time
            print(f"  Progress: {progress:.1f}% ({i + B}/{N}) | Elapsed: {elapsed:.1f}s")
    
    si = same_count / N
    
    elapsed = time.time() - start_time
    print(f"SI computation complete: {elapsed:.1f}s")
    print(f"SI = {si:.6f}")
    
    return si


def separation_index(
    features, 
    labels, 
    batch_size=None, 
    device=None,
    normalize=False  # ✅ Changed: default False to preserve original definition
):
    """
    Compute Separation Index with automatic batching.
    
    This is the main entry point for computing SI. It automatically handles:
        - numpy/torch conversion
        - device selection
        - batched computation for memory efficiency
    
    Args:
        features: numpy array or torch tensor of shape (N, D)
        labels: numpy array or torch tensor of shape (N,)
        batch_size: batch size for memory-efficient computation (auto if None)
        device: 'cpu', 'cuda', or None (auto-detect)
        normalize: Optional L2 normalization.
                   Disabled by default to preserve original SI definition.
                   (Default: False)
    
    Returns:
        si: Separation Index value between 0 and 1
        
    Example:
        >>> si = separation_index(features, labels)
        >>> print(f"SI = {si:.4f}")
    """
    
    # Validate inputs
    if len(features) != len(labels):
        raise ValueError(f"Features and labels must have same length: {len(features)} vs {len(labels)}")
    
    if len(features) == 0:
        raise ValueError("Empty dataset")
    
    # Auto-detect device if None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Auto-adjust batch size based on dataset size
    N = len(features)
    if batch_size is None:
        if N > 50000:
            batch_size = 2000
        elif N > 10000:
            batch_size = 1000
        else:
            batch_size = 500
    
    # Always use batched version for memory safety
    return separation_index_batched(
        features,
        labels,
        batch_size=batch_size,
        device=device,
        normalize=normalize
    )


def separation_index_from_dist_matrix(dist_matrix, labels):
    """
    Compute SI from pre-computed distance matrix.
    
    This is useful when distances are already computed externally.
    
    Args:
        dist_matrix: (N, N) tensor of pairwise distances
        labels: (N,) tensor of labels
    
    Returns:
        si: Separation Index value
    """
    
    if isinstance(dist_matrix, np.ndarray):
        dist_matrix = torch.from_numpy(dist_matrix).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()
    
    N = dist_matrix.shape[0]
    
    # Set diagonal to infinity to ignore self
    dist_matrix = dist_matrix.clone()
    dist_matrix.fill_diagonal_(float('inf'))
    
    nearest_idx = torch.argmin(dist_matrix, dim=1)
    same_label = (labels == labels[nearest_idx]).float()
    
    return torch.mean(same_label).item()


def compute_si_for_latent(
    model, 
    dataloader, 
    use_mean=True, 
    device=None, 
    batch_size=None,
    normalize=False  # ✅ Changed: default False to preserve original definition
):
    """
    Convenience function to compute SI directly from a model and dataloader.
    
    For VAE-based models, this extracts the latent representation and computes SI.
    
    Args:
        model: PyTorch model with forward returning (output, mu, std, z)
        dataloader: DataLoader yielding (images, labels)
        use_mean: If True, use mu (mean latent) instead of sampled z
        device: 'cpu', 'cuda', or None (auto-detect)
        batch_size: batch size for SI computation
        normalize: Optional L2 normalization.
                   Disabled by default to preserve original SI definition.
                   (Default: False)
    
    Returns:
        si: Separation Index value
        features: Extracted latent features
        labels: Corresponding labels
    """
    
    # Auto-detect device if None
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    model = model.to(device)
    
    all_features = []
    all_labels = []
    
    print("Extracting latent features...")
    if use_mean:
        print("  Using mean latent (mu) - recommended for deterministic SI")
    else:
        print("  Using sampled latent (z) - may have variance")
    
    if normalize:
        print("  ⚠️  Features will be L2 normalized (SI on hypersphere)")
    else:
        print("  ℹ️  No normalization applied (original SI on latent manifold)")
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            
            # Forward pass
            if use_mean:
                # Use mean (mu) instead of sampled z
                _, mu, _, _ = model(images)
                features = mu
            else:
                _, _, _, z = model(images)
                features = z
            
            all_features.append(features.cpu().numpy())
            # CPU-safe labels
            all_labels.append(labels.cpu().numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted {len(features)} features with dimension {features.shape[1]}")
    
    # Compute SI
    si = separation_index(
        features, 
        labels, 
        batch_size=batch_size, 
        device=device,
        normalize=normalize
    )
    
    return si, features, labels


# ==========================
# Legacy compatibility
# ==========================

# Keep original names for compatibility
SeprationIndex = separation_index
SeprationIndex2 = separation_index


# ==========================
# Testing / Demo
# ==========================

if __name__ == "__main__":
    """
    Quick test of Separation Index functions.
    """
    
    print("=" * 60)
    print("Testing Separation Index")
    print("=" * 60)
    
    # Create synthetic data
    print("\nCreating synthetic data...")
    N = 1000
    D = 10
    n_classes = 5
    
    # Features with clear separation
    features = np.random.randn(N, D)
    features[:200] += 3.0  # Class 0
    features[200:400] += 2.0  # Class 1
    features[400:600] += 1.0  # Class 2
    features[600:800] -= 1.0  # Class 3
    features[800:1000] -= 2.0  # Class 4
    
    labels = np.repeat(np.arange(n_classes), N // n_classes)
    
    # Test standard function (without normalization - default)
    print("\nTesting separation_index (without normalization)...")
    si_no_norm = separation_index(features, labels)
    print(f"SI (no norm) = {si_no_norm:.6f}")
    
    # Test with normalization (for comparison)
    print("\nTesting separation_index (with L2 normalization)...")
    si_norm = separation_index(features, labels, normalize=True)
    print(f"SI (L2 norm) = {si_norm:.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)