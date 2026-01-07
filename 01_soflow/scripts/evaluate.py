#!/usr/bin/env python3
"""
Evaluation script for SoFlow models.

Compute FID scores and other metrics for generated samples.

Usage:
    python scripts/evaluate.py --samples_dir ./samples --reference_dir ./imagenet_stats
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import clean-fid
try:
    from cleanfid import fid
    HAS_CLEANFID = True
except ImportError:
    HAS_CLEANFID = False

# Try to import torch-fidelity
try:
    import torch_fidelity
    HAS_TORCH_FIDELITY = True
except ImportError:
    HAS_TORCH_FIDELITY = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SoFlow samples")
    
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Directory containing generated samples")
    parser.add_argument("--reference_dir", type=str, default=None,
                        help="Directory containing reference images/stats")
    parser.add_argument("--reference_stats", type=str, default=None,
                        help="Path to precomputed reference statistics")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "cifar10", "custom"],
                        help="Reference dataset")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results")
    
    return parser.parse_args()


def compute_fid_cleanfid(samples_dir: str, reference: str, batch_size: int = 64):
    """Compute FID using clean-fid library."""
    if not HAS_CLEANFID:
        raise ImportError("clean-fid not installed. Run: pip install clean-fid")
    
    # Check if samples are images or need to be decoded
    sample_files = list(Path(samples_dir).glob("*.png")) + list(Path(samples_dir).glob("*.jpg"))
    
    if len(sample_files) == 0:
        print("No image files found. Please decode latents first.")
        return None
    
    # Compute FID
    score = fid.compute_fid(
        samples_dir,
        reference,
        mode="clean",
        batch_size=batch_size,
    )
    
    return score


def compute_fid_manual(
    samples: np.ndarray,
    reference_stats: dict,
    batch_size: int = 64,
    device: str = "cuda",
):
    """
    Compute FID manually using Inception features.
    
    Args:
        samples: Generated samples of shape (N, C, H, W) in [0, 1].
        reference_stats: Dict with 'mu' and 'sigma' for reference distribution.
        batch_size: Batch size for feature extraction.
        device: Device to use.
        
    Returns:
        FID score.
    """
    from scipy import linalg
    
    # Load Inception model
    try:
        from torchvision.models import inception_v3
        inception = inception_v3(pretrained=True, transform_input=False)
        inception.fc = torch.nn.Identity()
        inception = inception.to(device)
        inception.eval()
    except Exception as e:
        print(f"Error loading Inception model: {e}")
        return None
    
    # Extract features
    features = []
    num_batches = (len(samples) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting features"):
            batch = samples[i * batch_size:(i + 1) * batch_size]
            batch = torch.from_numpy(batch).float().to(device)
            
            # Resize to 299x299 for Inception
            batch = torch.nn.functional.interpolate(
                batch, size=(299, 299), mode="bilinear", align_corners=False
            )
            
            # Normalize to [-1, 1]
            batch = batch * 2 - 1
            
            feat = inception(batch)
            features.append(feat.cpu().numpy())
    
    features = np.concatenate(features, axis=0)
    
    # Compute statistics
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    # Compute FID
    mu_ref = reference_stats["mu"]
    sigma_ref = reference_stats["sigma"]
    
    diff = mu - mu_ref
    covmean, _ = linalg.sqrtm(sigma @ sigma_ref, disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = diff @ diff + np.trace(sigma + sigma_ref - 2 * covmean)
    
    return fid_score


def compute_metrics_torch_fidelity(
    samples_dir: str,
    reference_dir: str,
):
    """Compute metrics using torch-fidelity."""
    if not HAS_TORCH_FIDELITY:
        raise ImportError("torch-fidelity not installed. Run: pip install torch-fidelity")
    
    metrics = torch_fidelity.calculate_metrics(
        input1=samples_dir,
        input2=reference_dir,
        cuda=True,
        fid=True,
        kid=True,
        prc=True,
    )
    
    return metrics


def main():
    args = parse_args()
    
    print(f"Evaluating samples from: {args.samples_dir}")
    
    results = {}
    
    # Try different FID computation methods
    if HAS_CLEANFID and args.reference_dir:
        print("Computing FID with clean-fid...")
        try:
            fid_score = compute_fid_cleanfid(
                args.samples_dir,
                args.reference_dir,
                args.batch_size,
            )
            if fid_score is not None:
                results["fid_cleanfid"] = fid_score
                print(f"FID (clean-fid): {fid_score:.2f}")
        except Exception as e:
            print(f"clean-fid failed: {e}")
    
    if HAS_TORCH_FIDELITY and args.reference_dir:
        print("Computing metrics with torch-fidelity...")
        try:
            metrics = compute_metrics_torch_fidelity(
                args.samples_dir,
                args.reference_dir,
            )
            results.update(metrics)
            print(f"FID: {metrics.get('frechet_inception_distance', 'N/A'):.2f}")
            print(f"KID: {metrics.get('kernel_inception_distance_mean', 'N/A'):.4f}")
        except Exception as e:
            print(f"torch-fidelity failed: {e}")
    
    # Load precomputed stats if provided
    if args.reference_stats:
        print(f"Loading reference statistics from: {args.reference_stats}")
        ref_stats = np.load(args.reference_stats, allow_pickle=True).item()
        
        # Load samples
        samples_path = os.path.join(args.samples_dir, "samples.npy")
        if os.path.exists(samples_path):
            samples = np.load(samples_path)
            
            # Compute FID
            fid_score = compute_fid_manual(
                samples, ref_stats, args.batch_size, args.device
            )
            if fid_score is not None:
                results["fid_manual"] = fid_score
                print(f"FID (manual): {fid_score:.2f}")
    
    # Save results
    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    return results


if __name__ == "__main__":
    main()

