import os
from datetime import datetime
import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from Experiment_fns import Experiment_GP, Experiment_PFN, Experiment_Random
from Aquisition_sampling import generate_sobol_points
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
from tqdm import tqdm
from RFF import RFFSampler

def main():
    # Save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_save_dir = f"results/run_{timestamp}"
    os.makedirs(base_save_dir, exist_ok=True)

    # Device
    device = 'cuda'

    # Experiments parameters
    n_repeats = 21
    N_iters = 50
    features = 10000
    x_dims = [2, 5, 10]
    n_fns = 1
    ls = 0.1
    bounds_list = []
    bounds_list.append(torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64, device=device))
    bounds_list.append(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device))
    bounds_list.append(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device))
    n_samples = 100000
    seed = 42

    # Initialise storage
    x_query_store = [torch.zeros((3, n_repeats, N_iters, d), dtype=torch.float64, device='cpu') for d in x_dims]
    x_init_store = [torch.zeros((3, n_repeats, 5 * d, d), dtype=torch.float64, device='cpu') for d in x_dims]
    y_init_store = [torch.zeros((3, 3, n_repeats, 5 * d, n_fns), dtype=torch.float64, device='cpu') for d in x_dims]

    y_true_store = torch.zeros((3, 3, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu')
    y_best_store = torch.zeros((3, 3, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu')
    mu_store = torch.zeros((3, 2, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu')
    var_store = torch.zeros((3, 2, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu')
    alpha_store = torch.zeros((3, 2, n_repeats, N_iters, n_fns), dtype=torch.float64, device='cpu')

    # Create PFN
    model_path = pfns4bo.hebo_plus_model
    pfn = TransformerBOMethod(torch.load(model_path, weights_only=False), device='cuda')

    for k in range(3):
        x_dim = x_dims[k]
        bounds = bounds_list[k]
        n_init = 5 * x_dim

        sobol_acq_points = generate_sobol_points(
            bounds, 
            n_samples, 
            seed, 
            device
        )
        
        for i in tqdm(range(n_repeats), desc=f"Running Experiments for dimension set {k}"):
            # Draw from Matern32
            rff_sampler = RFFSampler(
                num_features=features, 
                input_dim=x_dim,
                number_of_functions=n_fns, 
                lengthscale=ls, 
                kernel="Matern32"
            )
            rff_sampler.omegas = rff_sampler.omegas.reshape(
                rff_sampler.num_features, rff_sampler.input_dim
            )

            # Save RFF function draw
            filepath_problem = f"{base_save_dir}/dim_{x_dim}/repeat_{i}"
            rff_sampler.save_problem(filepath_problem, bounds[0], bounds[1])

            # Sample space
            x_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(
                (n_init, x_dim), dtype=torch.float64, device=device
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
            x_query_store[k][0, i, :, :] = x_query_arr_GP.detach().cpu()
            x_init_store[k][0, i, :, :] = x_init_GP.detach().cpu()
            y_init_store[k][0, i, :, :] = y_init_GP.detach().cpu()
            y_true_store[k, 0, i, :, :] = y_true_arr_GP.detach().cpu()
            y_best_store[k, 0, i, :, :] = y_best_arr_GP.detach().cpu()
            mu_store[k, 0, i, :, :] = mu_arr_GP.detach().cpu()
            var_store[k, 0, i, :, :] = var_arr_GP.detach().cpu()
            alpha_store[k, 0, i, :, :] = alpha_arr_GP.detach().cpu()

            x_query_store[k][1, i, :, :] = x_query_arr_PFN.detach().cpu()
            x_init_store[k][1, i, :, :] = x_init_PFN.detach().cpu()
            y_init_store[k][1, i, :, :] = y_init_PFN.detach().cpu()
            y_true_store[k, 1, i, :, :] = y_true_arr_PFN.detach().cpu()
            y_best_store[k, 1, i, :, :] = y_best_arr_PFN.detach().cpu()
            mu_store[k, 1, i, :, :] = mu_arr_PFN.detach().cpu()
            var_store[k, 1, i, :, :] = var_arr_PFN.detach().cpu()
            alpha_store[k, 1, i, :, :] = alpha_arr_PFN.detach().cpu()

            x_query_store[k][2, i, :, :] = x_query_arr_rs.detach().cpu()
            x_init_store[k][2, i, :, :] = x_init_rs.detach().cpu()
            y_init_store[k][2, i, :, :] = y_init_rs.detach().cpu()
            y_true_store[k, 2, i, :, :] = y_true_arr_rs.detach().cpu()
            y_best_store[k, 2, i, :, :] = y_best_arr_rs.detach().cpu()
    
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
        "seed": seed
    }
    final_save_path = os.path.join(base_save_dir, "experimental_results.pt")
    torch.save(data_dict, final_save_path)
    print(f"Experiment complete. All data saved to: {base_save_dir}")
    return 0

if __name__ == '__main__':
    main()