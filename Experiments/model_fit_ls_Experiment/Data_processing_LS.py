import os
import torch
import numpy as np
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm
import math

def compute_comparative_metrics(base_save_dir: str):
    # Load experimental results
    results_path = os.path.join(base_save_dir, "experimental_results.pt")
    data = torch.load(results_path, map_location='cpu')

    # Extract
    y_true_store = data["y_true"]
    mu_store = data["mu"]
    var_store = data["var"]
    y_init = data["y_init"]

    # Sizing
    n_dims = 1
    n_methods = 2
    n_repeats = 21
    n_samples = 1000
    n_fns = 1
    n_tests = 9
    
    # Initialize metric storage
    metrics = {
        "pred_error": torch.zeros((n_tests, n_dims, n_methods, n_repeats, n_samples, n_fns)), # GP & PFN only
        "total_error": torch.zeros((n_tests, n_dims, n_methods, n_repeats, 1)),
        "EI": torch.zeros((n_tests, n_dims, n_methods, n_repeats, n_samples, 1))
    }

    # Loop through all tests
    for test in tqdm(range(n_tests), desc="Computing Metrics per test"):
        
        # Loop through all dimension changes
        for k in tqdm(range(n_dims), desc="Computing Metrics per Dimension", leave=False):
    
            # Loop through all reapeat runs
            for rep in range(n_repeats):
                    
                # Regret & Prediction Error
                # Looping through methods
                for m_idx in range(n_methods):
                    # Calculate pred error
                    metrics["pred_error"][test, k, m_idx, rep, :, :] = abs(
                        mu_store[k][test, m_idx, rep, :, :] - y_true_store[k][test, m_idx, rep, :, :]
                    )
                    metrics["total_error"][test, k, m_idx, rep, :, :] = torch.sum(
                        metrics["pred_error"][test, k, m_idx, rep, :, :], keepdim=True
                    )

                    # Get ei
                    best_f = torch.max(y_init[k][test, rep, :, :])
                    sigma_sq = var_store[k][test, m_idx, :, :]
                    sigma_safe = torch.sqrt(torch.clamp(sigma_sq, min=1e-6))
                    mu = mu_store[k][test, m_idx, :, :]
                    improvement = mu - best_f # Improvement with exploration term set to zero
                    Z = improvement / sigma_safe
                    cdf = torch.special.ndtr(Z)
                    pdf = torch.exp(-0.5 * Z**2) / math.sqrt(2 * math.pi)
                    ei = improvement * cdf + sigma_safe * pdf
                    
                    metrics["EI"][test, k, m_idx, rep, :, :] = torch.clamp(ei, min=0.0)

    # Save metrics
    save_path = os.path.join(base_save_dir, "metrics.pt")
    torch.save(metrics, save_path)
    print(f"Metrics evaluated and saved successfully to {save_path}.")

if __name__ == "__main__":
    # Substitute with your actual base_save_dir
    base_dir = "LS_1D_varied_results/run_20260729_180254" 
    compute_comparative_metrics(base_dir)