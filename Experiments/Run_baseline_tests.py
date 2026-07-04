import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from Experiment_fns import Experiment_GP, Experiment_PFN
from Aquisition_sampling import generate_sobol_points
from tqdm import tqdm
from RFF import RFFSampler

def main():
    # Experiments parameters
    n_repeats = 21
    N_iters = 50
    features = 2000
    x_dim = 1
    n_fns = 1
    ls = 1.0
    bounds = torch.Tensor([[0.0], [1.0]])
    n_samples = 100000
    n_init = 20
    seed = 42
    device = 'cuda'

    sobol_acq_points = generate_sobol_points(
        bounds, 
        n_samples, 
        seed, 
        device
    )

    for i in tqdm(range(n_repeats), desc="Running Experiments ..."):
        # Draw from Matern32
        rff_sampler = RFFSampler(
            num_features=features, 
            input_dim=x_dim,
            number_of_functions=n_fns, 
            lengthscale=ls, 
            kernel="Matern32"
        )
        rff_sampler.omegas = rff_sampler.omegas.reshape(rff_sampler.num_features, rff_sampler.input_dim)

        # Sample space
        x_train = torch.linspace(bounds[0], bounds[1], n_init, dtype=torch.float64).reshape(-1, 1)

        # Run GP experiment
        x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr = Experiment_GP(
            rff_sampler, x_train, N_iters, sobol_acq_points
        )

        # Run PFN experiment
        x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr = Experiment_PFN(
            rff_sampler, x_train, N_iters, sobol_acq_points
        )

        # Store Data
    return 0

if __name__ == '__main__':
    main()