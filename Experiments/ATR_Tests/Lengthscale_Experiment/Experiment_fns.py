import torch
import gpytorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior, SmoothedBoxPrior
from Aquisition_sampling import optimise_EI_GP
from pfn_evaluate import eval_pfn
from tqdm import tqdm
from RFF import RFFSampler
from gpytorch.likelihoods import GaussianLikelihood
from Utils import output_standardise, unscale_outputs
from ATR_warping import ATR_warped_PFN

# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pfns4bo")

def Experiment_ATR_PFN(atr_pfn: ATR_warped_PFN, rff_sampler: RFFSampler, x_train, N_iters, n_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Get best point
    max_idx = torch.argmax(y_train)
    x_best = x_train[max_idx]

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()

    # Initial storage
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    mu_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    var_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    alpha_arr = torch.zeros(N_iters, 1, dtype=torch.float64, device='cpu')
    tr_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')

    # BO test
    for i in tqdm(range(N_iters), desc="PFN", leave=False):

        next_x, acq_value, tr_size, candidate_mean, candidate_var = atr_pfn.observe_and_suggest(
            x_train, y_train, x_best, n_acq_points=n_acq_points, return_prediction=True
        )

        # Evaluate and add the new point to the training set
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()
        mu_arr[i, :] = candidate_mean.detach().cpu()
        var_arr[i, :] = candidate_var.detach().cpu()
        alpha_arr[i, :] = acq_value.detach().cpu().flatten()[0]
        tr_arr[i, :] = tr_size.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr