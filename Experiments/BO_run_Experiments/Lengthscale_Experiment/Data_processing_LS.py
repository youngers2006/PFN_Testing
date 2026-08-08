import os
import torch
import numpy as np
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

def compute_comparative_metrics(base_save_dir_1: str, base_save_dir_2: str):
    # Load experimental results
    results_path_1 = os.path.join(base_save_dir_1, "experimental_results.pt")
    results_path_2 = os.path.join(base_save_dir_2, "experimental_results.pt")
    data_1 = torch.load(results_path_1, map_location='cpu')
    data_2 = torch.load(results_path_2, map_location='cpu')
    
    # load minimum of each experiment
    minimas_path = os.path.join(base_save_dir_1, "maximas.pt")
    minima_data = torch.load(minimas_path, map_location='cpu')

    # Extract minimum values
    true_minimas_y = minima_data["y"] # Shape: (3, 3, n_repeats, 1)

    # Extract
    y_true_store_main = data_1["y_true"]
    y_best_store_main = data_1["y_best"]
    mu_store_main = data_1["mu"]

    y_true_store_warp = data_2["y_true"]
    y_best_store_warp = data_2["y_best"]
    mu_store_warp = data_2["mu"]
    
    # Data shape extraction
    x_dims = [1]
    methods = ["GP", "PFN", "Random", "Warped_PFN"]

    # Sizing
    n_dims = 1
    n_methods = 4
    n_repeats = 21
    N_iters = 50
    n_fns = 1
    n_tests = 9
    
    # Initialize metric storage
    metrics = {
        "regret": torch.zeros((n_tests, n_dims, n_methods, n_repeats, N_iters)),
        "pred_error": torch.zeros((n_tests, n_dims, n_methods, n_repeats, N_iters)), # GP & PFN only
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
                        if m_idx != 3:
                            y_best_found = y_best_store_main[k][test, m_idx, rep, t, 0].item()
                            metrics["regret"][test, k, m_idx, rep, t] = abs(y_opt_continuous - y_best_found)
                        elif m_idx == 3:
                            y_best_found = y_best_store_warp[k][test, 0, rep, t, 0].item()
                            metrics["regret"][test, k, m_idx, rep, t] = abs(y_opt_continuous - y_best_found)
                        
                        # Prediction error of surrogate
                        if m_idx == 0 or m_idx == 1:
                            queried_y = y_true_store_main[k][test, m_idx, rep, t, 0].item()
                            pred_mu = mu_store_main[k][test, m_idx, rep, t, 0].item()
                            metrics["pred_error"][test, k, m_idx, rep, t] = abs(pred_mu - queried_y)
                        elif m_idx == 3:
                            queried_y = y_true_store_warp[k][test, 0, rep, t, 0].item()
                            pred_mu = mu_store_warp[k][test, 0, rep, t, 0].item()
                            metrics["pred_error"][test, k, 3, rep, t] = abs(pred_mu - queried_y)
                    
                    # Rank between methods
                    y_t_vals_main = y_true_store_main[k][test, :, rep, t, 0].numpy()
                    y_t_vals_warp = y_true_store_warp[k][test, :, rep, t, 0].numpy()
                    y_t_vals = np.concatenate([y_t_vals_main, y_t_vals_warp])

                    # Negate y_t_vals so the highest objective gets rank 1
                    ranks = rankdata(-y_t_vals, method='min') 
                    for m_idx in range(n_methods):
                        metrics["rank_of_method"][test, k, m_idx, rep, t] = ranks[m_idx]

    # Save metrics
    save_path = os.path.join(base_save_dir_2, "metrics_store.pt")
    torch.save(metrics, save_path)
    print(f"Metrics evaluated and saved successfully to {save_path}.")

if __name__ == "__main__":
    # Substitute with your actual base_save_dir
    base_dir_1 = "results_LS_BO_PFN_GP/run_N_50_D_1" 
    base_dir_2 = "results_LS_BO_ATR_PFN/run_N_50_D_1_mode_k4" 
    compute_comparative_metrics(base_dir_1, base_dir_2)