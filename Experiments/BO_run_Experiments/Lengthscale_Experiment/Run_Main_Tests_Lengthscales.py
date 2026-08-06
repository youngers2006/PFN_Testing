import os
import torch
from Experiment_fns import Experiment_GP, Experiment_PFN, Experiment_Random
from Aquisition_sampling import generate_sobol_points
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from tqdm import tqdm
from RFF import RFFSampler

def main():
    # Save paths
    base_save_dir = f"results_LS_BO_PFN_GP/run_N_50"
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
    n_methods_UQ = 2
    n_dims = 3
    N_iters = 50
    features = 10000
    x_dims = [1, 2, 5]
    n_fns = 1
    bounds_list = []
    bounds_list.append(torch.tensor([[0.0], [1.0]], dtype=torch.float64, device=device))
    bounds_list.append(torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64, device=device))
    bounds_list.append(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device))
    n_samples = 100000
    seed = 42
    seed_init = 10
    kernel = "Matern32"

    # Gamma distribution parameters for lengthscale and variance RFF parameters
    lengthscales = [0.021, 0.03, 0.05, 0.07, 0.1, 0.1385, 0.4, 0.8, 1.5]
    amplitude = 1.0
    lengthscale_store = torch.tensor([lengthscales], device='cpu')
    n_tests = 9

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    # Initialise storage (test_n, dim_n, repeats_n, iter_n, size_n)
    x_query_store = [torch.zeros((n_tests, n_methods, n_repeats, N_iters, d), dtype=torch.float64, device='cpu') for d in x_dims]
    x_init_store =  [torch.zeros((n_tests, n_methods, n_repeats, 5 * d, d), dtype=torch.float64, device='cpu') for d in x_dims]
    y_init_store =  [torch.zeros((n_tests, n_methods, n_repeats, 5 * d, n_fns), dtype=torch.float64, device='cpu') for d in x_dims]
    y_true_store =  [torch.zeros((n_tests, n_methods, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    y_best_store =  [torch.zeros((n_tests, n_methods, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    mu_store =      [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    var_store =     [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    alpha_store =   [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]

    # Create PFN
    model_path = pfns4bo.hebo_plus_model
    pfn = TransformerBOMethod(torch.load(model_path, weights_only=False), device='cuda')

    for test in range(n_tests):
        lengthscale = lengthscales[test]
        
        for k in range(n_dims):
            x_dim = x_dims[k]
            bounds = bounds_list[k]
            n_init = 5 * x_dim
    
            sobol_acq_points = generate_sobol_points(
                bounds, 
                n_samples, 
                seed, 
                device
            )
            
            for i in tqdm(range(n_repeats), desc=f"Running Experiments for dimension set {k} and test {test}"):
                # Draw from desired kernel
                rff_sampler = RFFSampler(
                    num_features=features, 
                    input_dim=x_dim,
                    number_of_functions=n_fns, 
                    lengthscale = lengthscale, 
                    amplitude = amplitude,
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
                    seed_init + i + (k * n_repeats),
                    device
                )
                
                # Run random sampling
                x_query_arr_rs, x_init_rs, y_true_arr_rs, y_init_rs, y_best_arr_rs = Experiment_Random(
                    rff_sampler, x_train, bounds, N_iters
                )
    
                # Run GP experiment
                x_query_arr_GP, x_init_GP, y_true_arr_GP, y_init_GP, y_best_arr_GP, mu_arr_GP, var_arr_GP, alpha_arr_GP = Experiment_GP(
                    rff_sampler, x_train, N_iters, sobol_acq_points
                )
    
                # Run PFN experiment
                x_query_arr_PFN, x_init_PFN, y_true_arr_PFN, y_init_PFN, y_best_arr_PFN, mu_arr_PFN, var_arr_PFN, alpha_arr_PFN = Experiment_PFN(
                    pfn, rff_sampler, x_train, N_iters, sobol_acq_points
                )
    
                # Store Data (in_dim, method, test_iter, opt_iter, data)
                x_query_store[k][test, 0, i, :, :] = x_query_arr_GP.detach().cpu()
                x_init_store[k][test, 0, i, :, :] = x_init_GP.detach().cpu()
                y_init_store[k][test, 0, i, :, :] = y_init_GP.detach().cpu()
                y_true_store[k][test, 0, i, :, :] = y_true_arr_GP.detach().cpu()
                y_best_store[k][test, 0, i, :, :] = y_best_arr_GP.detach().cpu()
                mu_store[k][test, 0, i, :, :] = mu_arr_GP.detach().cpu()
                var_store[k][test, 0, i, :, :] = var_arr_GP.detach().cpu()
                alpha_store[k][test, 0, i, :, :] = alpha_arr_GP.detach().cpu()
    
                x_query_store[k][test, 1, i, :, :] = x_query_arr_PFN.detach().cpu()
                x_init_store[k][test, 1, i, :, :] = x_init_PFN.detach().cpu()
                y_init_store[k][test, 1, i, :, :] = y_init_PFN.detach().cpu()
                y_true_store[k][test, 1, i, :, :] = y_true_arr_PFN.detach().cpu()
                y_best_store[k][test, 1, i, :, :] = y_best_arr_PFN.detach().cpu()
                mu_store[k][test, 1, i, :, :] = mu_arr_PFN.detach().cpu()
                var_store[k][test, 1, i, :, :] = var_arr_PFN.detach().cpu()
                alpha_store[k][test, 1, i, :, :] = alpha_arr_PFN.detach().cpu()
    
                x_query_store[k][test, 2, i, :, :] = x_query_arr_rs.detach().cpu()
                x_init_store[k][test, 2, i, :, :] = x_init_rs.detach().cpu()
                y_init_store[k][test, 2, i, :, :] = y_init_rs.detach().cpu()
                y_true_store[k][test, 2, i, :, :] = y_true_arr_rs.detach().cpu()
                y_best_store[k][test, 2, i, :, :] = y_best_arr_rs.detach().cpu()
    
    # Save all data
    data_dict = {
        "x_query": x_query_store,
        "x_init": x_init_store,
        "y_true": y_true_store,
        "y_init": y_init_store,
        "y_best": y_best_store,
        "mu": mu_store,
        "var": var_store,
        "alpha": alpha_store,
        "seed": seed,
        "seed_init": seed_init,
        "lengthscales": lengthscale_store,
        "amplitudes": amplitude
    }
    final_save_path = os.path.join(base_save_dir, "experimental_results.pt")
    torch.save(data_dict, final_save_path)
    print(f"Experiment complete. All data saved to: {base_save_dir}")
    return 0

if __name__ == '__main__':
    main()