import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from Experiment_fns import Experiment_GP, Experiment_PFN, Experiment_HEBO
from tqdm import tqdm
from RFF import RFFSampler

def main():
    # Experiments parameters
    n_repeats = 51

    for i in tqdm(range(n_repeats), desc="Running Experiments ..."):
        # Draw from Matern32


        # Run GP experiment
        x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr = Experiment_GP(
            rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points
        )

        # Run PFN experiment
        x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr = Experiment_PFN(
            rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points
        )

        # Run HEBO experiment
        x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr = Experiment_HEBO(
            rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points
        )

        # Store Data
    return 0

if __name__ == '__main__':
    main()