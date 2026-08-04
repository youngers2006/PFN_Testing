import os
import torch
import numpy as np
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

def compute_comparative_metrics(base_save_dir: str):
    # Load experimental results
    results_path = os.path.join(base_save_dir, "experimental_results.pt")
    data = torch.load(results_path, map_location='cpu')
    
    # load minimum of each experiment
    minimas_path = os.path.join(base_save_dir, "maximas.pt")
    minima_data = torch.load(minimas_path, map_location='cpu')

    # Extract minimum values
    true_minimas_y = minima_data["y"] # Shape: (3, 3, n_repeats, 1)

    # Extract
    y_true_store = data["y_true"]
    y_best_store = data["y_best"]
    mu_store = data["mu"]
    
    # Data shape extraction
    x_dims = [2, 5, 10]
    methods = ["GP", "PFN", "Random"]

    # Sizing
    n_dims = 1
    n_methods = 3
    n_repeats = 21
    N_iters = 150
    n_fns = 1
    n_tests = 7
    
    # Initialize metric storage
    metrics = {
        "regret": torch.zeros((n_tests, n_dims, n_methods, n_repeats, N_iters)),
        "pred_error": torch.zeros((n_tests, n_dims, 2, n_repeats, N_iters)), # GP & PFN only
        "rank_of_method": torch.zeros((n_tests, n_dims, n_methods, n_repeats, N_iters)),
        "spearman_gp_pfn": torch.zeros((n_tests, n_dims, N_iters)),
        "spearman_gp_rand": torch.zeros((n_tests, n_dims, N_iters)),
        "spearman_pfn_rand": torch.zeros((n_tests, n_dims, N_iters))
    }

    # Loop through all tests
    for test in tqdm(range(n_tests), desc="Computing Metrics per test"):
        
        # Loop through all dimension changes
        for k in tqdm(range(n_dims), desc="Computing Metrics per Dimension", leave=False):
    
            # Loop through all reapeat runs
            for rep in range(n_repeats):
    
                # Get minimum value for current function
                y_opt_continuous = true_minimas_y[k][test, rep, 0].item()
    
                # Loop through all iterations
                for t in range(N_iters):
                    
                    # Regret & Prediction Error
                    # Looping through methods
                    for m_idx in range(n_methods):
                        # Calculate simple regret
                        y_best_found = y_best_store[k][test, m_idx, rep, t, 0].item()
                        metrics["regret"][test, k, m_idx, rep, t] = abs(y_opt_continuous - y_best_found)
                        
                        # Prediction error of surrogate
                        if m_idx < 2:
                            queried_y = y_true_store[k][test, m_idx, rep, t, 0].item()
                            pred_mu = mu_store[k][test, m_idx, rep, t, 0].item()
                            metrics["pred_error"][test, k, m_idx, rep, t] = abs(pred_mu - queried_y)
                    
                    # Rank between methods
                    y_t_vals = y_true_store[k][test, :, rep, t, 0].numpy()
                    # Negate y_t_vals so the highest objective gets rank 1
                    ranks = rankdata(-y_t_vals, method='min') 
                    for m_idx in range(n_methods):
                        metrics["rank_of_method"][test, k, m_idx, rep, t] = ranks[m_idx]
            
            # Inter-Method Spearman Correlation (Across repeats)
            for t in range(N_iters):
                gp_vals = y_best_store[k][test, 0, :, t, 0].numpy()
                pfn_vals = y_best_store[k][test, 1, :, t, 0].numpy()
                rand_vals = y_best_store[k][test, 2, :, t, 0].numpy()
                
                # Compute correlation. Catch NaNs if vectors have 0 variance
                rho_gp_pfn, _ = spearmanr(gp_vals, pfn_vals)
                rho_gp_rand, _ = spearmanr(gp_vals, rand_vals)
                rho_pfn_rand, _ = spearmanr(pfn_vals, rand_vals)
                
                metrics["spearman_gp_pfn"][test, k, t] = rho_gp_pfn if not np.isnan(rho_gp_pfn) else 0.0
                metrics["spearman_gp_rand"][test, k, t] = rho_gp_rand if not np.isnan(rho_gp_rand) else 0.0
                metrics["spearman_pfn_rand"][test, k, t] = rho_pfn_rand if not np.isnan(rho_pfn_rand) else 0.0

    # Save metrics
    save_path = os.path.join(base_save_dir, "baseline_metrics.pt")
    torch.save(metrics, save_path)
    print(f"Metrics evaluated and saved successfully to {save_path}.")

if __name__ == "__main__":
    # Substitute with your actual base_save_dir
    base_dir = "results_Amp_150/run_20260728_122053" 
    compute_comparative_metrics(base_dir)