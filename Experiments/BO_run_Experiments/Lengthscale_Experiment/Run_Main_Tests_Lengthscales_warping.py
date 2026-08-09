import os
import torch
from Experiment_fns_warping import Experiment_ATR_PFN
from Aquisition_sampling import generate_sobol_points
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from tqdm import tqdm
from RFF import RFFSampler
from ATR_warping import ATR_warped_PFN

def main():
    # Save paths
    base_save_dir = f"results_LS_BO_ATR_PFN/run_N_50_D_2_mode_k4"
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
    n_methods = 1
    n_methods_UQ = 1
    n_dims = 1
    N_iters = 50
    features = 10000
    x_dims = [2]
    n_fns = 1
    bounds_list = []
    bounds_list.append(torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64, device=device))
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
    y_true_store =  [torch.zeros((n_tests, n_methods, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    y_best_store =  [torch.zeros((n_tests, n_methods, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    mu_store =      [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    var_store =     [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    alpha_store =   [torch.zeros((n_tests, n_methods_UQ, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu') for _ in x_dims]
    tr_store = [torch.zeros((n_tests, n_methods, n_repeats, N_iters, d), dtype=torch.float64, device='cpu') for d in x_dims]

    filepath_data = f"results_LS_BO_PFN_GP/run_N_50_D_2/experimental_results.pt"
    data = torch.load(filepath_data)
    x_init_store = data["x_init"]
    y_init_store = data["y_init"]

    # Create PFN
    model_path = pfns4bo.hebo_plus_model
    ATR_pfn = ATR_warped_PFN(
        l_target=0.1385, 
        model_path=model_path,
        device='cuda'
    )

    for test in range(n_tests):
        
        for k in range(n_dims):
            x_dim = x_dims[k]
            
            for i in tqdm(range(n_repeats), desc=f"Running Experiments for dimension set {k} and test {test}"):
                # Save RFF function draw
                filepath_problem = f"results_LS_BO_PFN_GP/run_N_50_D_2/test_{test}/dim_{x_dim}/repeat_{i}/problem.npz"

                rff_sampler = RFFSampler(
                    num_features=1,
                    input_dim=1,
                    number_of_functions=1,
                    lengthscale = 1, 
                    amplitude = 1,
                )
                rff_sampler.load_problem_from_file(filepath_problem)

                # Get the initialisation points
                x_train = x_init_store[k][test, 1, i, :, :]
                
                # Run PFN experiment
                x_query_arr_PFN, _, y_true_arr_PFN, _, y_best_arr_PFN, mu_arr_PFN, var_arr_PFN, alpha_arr_PFN, tr_arr_PFN = Experiment_ATR_PFN(
                    ATR_pfn, rff_sampler, x_train, N_iters, n_acq_points=n_samples, restarts=True
                )
    
                # Store Data (in_dim, method, test_iter, opt_iter, data)
                x_query_store[k][test, 0, i, :, :] = x_query_arr_PFN.detach().cpu()
                y_true_store[k][test, 0, i, :, :] = y_true_arr_PFN.detach().cpu()
                y_best_store[k][test, 0, i, :, :] = y_best_arr_PFN.detach().cpu()
                mu_store[k][test, 0, i, :, :] = mu_arr_PFN.detach().cpu()
                var_store[k][test, 0, i, :, :] = var_arr_PFN.detach().cpu()
                alpha_store[k][test, 0, i, :, :] = alpha_arr_PFN.detach().cpu()
                tr_store[k][test, 0, i, :, :] = tr_arr_PFN.detach().cpu()
    
    # Save all data
    data_dict = {
        "x_query": x_query_store,
        "x_init": x_init_store,
        "y_true": y_true_store,
        "y_init": y_init_store,
        "y_best": y_best_store,
        "tr_size": tr_store,
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