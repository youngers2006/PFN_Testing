import os
from datetime import datetime
import torch
from Aquisition_sampling import generate_sobol_points
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from pfns4bo.scripts.tune_input_warping import fit_input_warping
from tqdm import tqdm
from RFF import RFFSampler
from Model_fit_fns import plot_GP_variance_surface, plot_pfn_variance_surface
from NL_warping import NL_warping

def main():
    # Save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"WP_1D_varied_results/run_{timestamp}"
    os.makedirs(base_save_dir, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
        print('cuda')
    else:
        device = 'cpu'
        print('cpu')

    # Experiments parameters
    n_repeats = 21
    n_methods = 3
    n_methods_UQ = 3
    features = 10000
    x_dims = [1]
    n_fns = 1
    bounds_list = []
    bounds_list.append(torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device))
    n_samples = 1000
    seed = 42
    seed_init = 10
    kernel = "Matern32"

    # number of initialisation points
    n_init = 10

    # Gamma distribution parameters for lengthscale and variance RFF parameters
    lengthscale_concentration = 1.2107
    lengthscale_rate = 1.5212 
    variance_concentration = 0.8452
    variance_rate = 0.3993
    gamma_params = torch.tensor([lengthscale_concentration, lengthscale_rate, variance_concentration, variance_rate], device='cpu')
    n_tests = 11

    # Warping Parameters
    alpha_NL = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 1.0, 1.0, 1.0, 1.0, 0.5]
    beta_NL = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 3.0, 10.0, 12.0, 15.0, 0.5]
    alpha_params = torch.tensor(alpha_NL)
    beta_params = torch.tensor(beta_NL)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    # Initialise storage (test_n, dim_n, repeats_n, sample_n, size_n)
    x_query_store = [torch.zeros((n_tests, n_methods, n_repeats, n_samples, d), dtype=torch.float64, device='cpu') for d in x_dims]
    y_true_store =  [torch.zeros((n_tests, n_methods, n_repeats, n_samples, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    y_UW_store =    [torch.zeros((n_tests, n_methods, n_repeats, n_samples, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    mu_store =      [torch.zeros((n_tests, n_methods_UQ, n_repeats, n_samples, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    var_store =     [torch.zeros((n_tests, n_methods_UQ, n_repeats, n_samples, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    x_init_store =  [torch.zeros((n_tests, n_repeats, n_init, d), dtype=torch.float64, device='cpu') for d in x_dims]
    y_init_store =  [torch.zeros((n_tests, n_repeats, n_init, n_fns), dtype=torch.float64, device='cpu') for d in x_dims]

    # Create PFN
    model_path = pfns4bo.hebo_plus_model
    pfn = TransformerBOMethod(torch.load(model_path, weights_only=False), device='cuda')
    pfn_W = TransformerBOMethod(torch.load(model_path, weights_only=False), fit_encoder=fit_input_warping, device='cuda')

    for test in range(n_tests):
        alpha = alpha_NL[test]
        beta = beta_NL[test]
        for k in range(1):
            x_dim = x_dims[k]
            bounds = bounds_list[k]
            
            for i in tqdm(range(n_repeats), desc=f"Running Experiments for dimension set {k} and test {test}"):
                # Draw from desired kernel
                rff_sampler = RFFSampler(
                    num_features=features, 
                    input_dim=x_dim,
                    number_of_functions=n_fns, 
                    ls_alpha = lengthscale_concentration, 
                    ls_beta = lengthscale_rate,
                    var_alpha = variance_concentration,
                    var_beta = variance_rate,
                    kernel=kernel
                )
                rff_sampler.omegas = rff_sampler.omegas.reshape(
                    rff_sampler.num_features, rff_sampler.input_dim
                )
    
                # Save RFF function draw
                filepath_problem = f"{base_save_dir}/test_{test}/dim_{x_dim}/repeat_{i}"
                rff_sampler.save_problem(filepath_problem, bounds[0], bounds[1])
    
                # Sample space
                x_train = generate_sobol_points(
                    bounds,
                    n_init,
                    seed_init + i * (k * n_repeats),
                    device
                )
                y_train = rff_sampler.sample(NL_warping(x_train, alpha, beta))

                # create eval grid
                x_queries = generate_sobol_points(bounds, n_samples, seed=42, device=device)
                y_true = rff_sampler.sample(NL_warping(x_queries, alpha, beta))
                y_UW = rff_sampler.sample(x_queries)
    
                # Test GP model fit
                mu_arr_GP, var_arr_GP, y_true_arr_GP = plot_GP_variance_surface(
                    x_train, y_train, x_queries, y_true, n_samples=n_samples, device=device
                )
    
                # Test PFN model fit
                mu_arr_PFN, var_arr_PFN, y_true_arr_PFN = plot_pfn_variance_surface(
                    pfn, x_train, y_train, x_queries, y_true, n_samples=n_samples, warping=False, device=device
                )

                # Test PFN model fit
                mu_arr_PFN_W, var_arr_PFN_W, y_true_arr_PFN_W = plot_pfn_variance_surface(
                    pfn_W, x_train, y_train, x_queries, y_true, n_samples=n_samples, warping=True, device=device
                )

                # Store initialising data
                x_init_store[k][test, i, :, :] = x_train.detach().cpu()
                y_init_store[k][test, i, :, :] = y_train.detach().cpu()
                
                # Store Data (in_dim, test, method, repeat, sample, data)
                x_query_store[k][test, 0, i, :, :] = x_queries.detach().cpu()
                y_true_store[k][test, 0, i, :, :] = y_true_arr_GP.detach().cpu()
                y_UW_store[k][test, 0, i, :, :] = y_UW.detach().cpu()
                mu_store[k][test, 0, i, :, :] = mu_arr_GP.detach().cpu()
                var_store[k][test, 0, i, :, :] = var_arr_GP.detach().cpu()
    
                x_query_store[k][test, 1, i, :, :] = x_queries.detach().cpu()
                y_true_store[k][test, 1, i, :, :] = y_true_arr_PFN.detach().cpu()
                y_UW_store[k][test, 1, i, :, :] = y_UW.detach().cpu()
                mu_store[k][test, 1, i, :, :] = mu_arr_PFN.detach().cpu()
                var_store[k][test, 1, i, :, :] = var_arr_PFN.detach().cpu()

                x_query_store[k][test, 2, i, :, :] = x_queries.detach().cpu()
                y_true_store[k][test, 2, i, :, :] = y_true_arr_PFN_W.detach().cpu()
                y_UW_store[k][test, 2, i, :, :] = y_UW.detach().cpu()
                mu_store[k][test, 2, i, :, :] = mu_arr_PFN_W.detach().cpu()
                var_store[k][test, 2, i, :, :] = var_arr_PFN_W.detach().cpu()
    
    # Save all data
    data_dict = {
        "x_init": x_init_store,
        "y_init": y_init_store,
        "x_query": x_query_store,
        "y_UW": y_UW_store, 
        "y_true": y_true_store,
        "mu": mu_store,
        "var": var_store,
        "seed": seed,
        "seed_init": seed_init,
        "gamma_params": gamma_params, 
        "alpha_params": alpha_params, 
        "beta_params": beta_params
    }
    final_save_path = os.path.join(base_save_dir, "experimental_results.pt")
    torch.save(data_dict, final_save_path)
    print(f"Experiment complete. All data saved to: {base_save_dir}")
    return 0

if __name__ == '__main__':
    main()