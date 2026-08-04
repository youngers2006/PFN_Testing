import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from Experiments.BO_run_Experiments.Warping_Experiment.Aquisition_sampling import optimise_EI_GP
from Experiments.BO_run_Experiments.Warping_Experiment.pfn_evaluate import eval_pfn
from tqdm import tqdm
from Experiments.BO_run_Experiments.Warping_Experiment.RFF import RFFSampler
from Utils import output_standardise, unscale_outputs
from Experiments.BO_run_Experiments.Warping_Experiment.HS_warping import HS_noise_sampling
from Experiments.BO_run_Experiments.Warping_Experiment.NL_warping import NL_warping

# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pfns4bo")

def Experiment_GP(rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points, kernel, sigma_min=None, sigma_max=None, alpha_NL=None, beta_NL=None, alpha_HS=None, beta_HS=None):

    # Compute values of sample initial points
    if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
        y_train = rff_sampler.sample(NL_warping(x_train, alpha_NL, beta_NL))
        y_train_true = y_train.clone().detach()
    elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
        y_train = rff_sampler.sample(x_train)
        y_train, y_train_true = HS_noise_sampling(x_train, y_train, sigma_min, sigma_max, alpha_HS, beta_HS)
    else:
        y_train = rff_sampler.sample(x_train)
        y_train_true = y_train.clone().detach()

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()
    y_init_true = y_train_true.clone().detach()

    # Initial storage
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')
    y_clean_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    mu_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    var_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    alpha_arr = torch.zeros(N_iters, 1, dtype=torch.float64, device='cpu')

    # BO test
    for i in tqdm(range(N_iters), desc="GP", leave=False):
        # Setup for GP
        D = x_train.shape[-1]
        matern_32 = MaternKernel(
            nu=1.5, 
            ard_num_dims=D,
            lengthscale_prior=GammaPrior(3.0, 6.0)
        )
        custom_covar = ScaleKernel(
            matern_32,
            outputscale_prior=GammaPrior(2.0, 0.15)
        )
        
        # Setup surrogate model
        model = SingleTaskGP(
            x_train, 
            y_train,
            outcome_transform=Standardize(m=y_train.shape[-1]),
            covar_module=custom_covar
        )

        # Tune Hyperparams to maximise MLL
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise Acquisition function
        f_best = y_train.max()
        qEI = qLogExpectedImprovement(model, f_best)
        candidate, acq_value, candidate_mean, candidate_var = optimise_EI_GP(
            acq_fn=qEI, 
            x_query=sobol_acq_points, 
            q=1
        )
        next_x = candidate.to(dtype=torch.float64).detach()

        # Evaluate and add the new point to the training set
        if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
            next_y = rff_sampler.sample(NL_warping(next_x, alpha_NL, beta_NL))
            next_y_true = next_y.clone().detach()
        elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
            next_y = rff_sampler.sample(next_x)
            next_y, next_y_true = HS_noise_sampling(next_x, next_y, sigma_min, sigma_max, alpha_HS, beta_HS)
        else:
            next_y = rff_sampler.sample(next_x)
            next_y_true = next_y.clone().detach()
        
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_clean_arr[i, :] = next_y_true.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()
        mu_arr[i, :] = candidate_mean.detach().cpu()
        var_arr[i, :] = candidate_var.detach().cpu()
        alpha_arr[i, :] = acq_value.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_clean_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr

def Experiment_PFN(pfn, rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points, kernel, sigma_min=None, sigma_max=None, alpha_NL=None, beta_NL=None, alpha_HS=None, beta_HS=None):

    # Compute values of sample initial points
    if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
        y_train = rff_sampler.sample(NL_warping(x_train, alpha_NL, beta_NL))
        y_train_true = y_train.clone().detach()
    elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
        y_train = rff_sampler.sample(x_train)
        y_train, y_train_true = HS_noise_sampling(x_train, y_train, sigma_min, sigma_max, alpha_HS, beta_HS)
    else:
        y_train = rff_sampler.sample(x_train)
        y_train_true = y_train.clone().detach()

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()
    y_init_true = y_train_true.clone().detach()

    # Initial storage
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_clean_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    mu_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    var_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    alpha_arr = torch.zeros(N_iters, 1, dtype=torch.float64, device='cpu')

    # BO test
    for i in tqdm(range(N_iters), desc="PFN", leave=False):
        # Standardise inputs
        y_train_scaled, mu_y, std_y = output_standardise(y_train)
        
        # Maximise Acquisition function
        candidate_idx, acq_value_scaled = pfn.observe_and_suggest(
            x_train,
            y_train_scaled,
            sobol_acq_points,
            return_actual_ei=True
        )
        candidate_mean_scaled, candidate_var_scaled = eval_pfn(
            pfn, x_train, y_train_scaled, sobol_acq_points[candidate_idx]
        )
        candidate_mean, candidate_var = unscale_outputs(
            candidate_mean_scaled, candidate_var_scaled, mu_y, std_y
        )
        
        acq_value = acq_value_scaled.cpu() * std_y.cpu()
        next_x = sobol_acq_points[candidate_idx].unsqueeze(0)

        # Evaluate and add the new point to the training set
        if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
            next_y = rff_sampler.sample(NL_warping(next_x, alpha_NL, beta_NL))
            next_y_true = next_y.clone().detach()
        elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
            next_y = rff_sampler.sample(next_x)
            next_y, next_y_true = HS_noise_sampling(next_x, next_y, sigma_min, sigma_max, alpha_HS, beta_HS)
        else:
            next_y = rff_sampler.sample(next_x)
            next_y_true = next_y.clone().detach()
        
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_clean_arr[i, :] = next_y_true.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()
        mu_arr[i, :] = candidate_mean.detach().cpu()
        var_arr[i, :] = candidate_var.detach().cpu()
        alpha_arr[i, :] = acq_value.detach().cpu().flatten()[0]

    return x_query_arr, x_init, y_true_arr, y_clean_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr

def Experiment_Random(rff_sampler: RFFSampler, x_train, bounds, N_iters, kernel, sigma_min=None, sigma_max=None, alpha_NL=None, beta_NL=None, alpha_HS=None, beta_HS=None):

    # Compute values of sample initial points
    if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
        y_train = rff_sampler.sample(NL_warping(x_train, alpha_NL, beta_NL))
        y_train_true = y_train.clone().detach()
    elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
        y_train = rff_sampler.sample(x_train)
        y_train, y_train_true = HS_noise_sampling(x_train, y_train, sigma_min, sigma_max, alpha_HS, beta_HS)
    else:
        y_train = rff_sampler.sample(x_train)
        y_train_true = y_train.clone().detach()

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()
    y_init_true = y_train_true.clone().detach()

    # Initial storage (no surrogate or aquisition fn)
    device = x_train.device
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_clean_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')

    for i in tqdm(range(N_iters), desc="Random Baseline", leave=False):
        # Draw random sample point
        u = torch.rand((1, x_train.shape[1]), dtype=torch.float64, device=device)
        next_x = bounds[0] + (bounds[1] - bounds[0]) * u

        # Evaluate sample point and add it to the seen point collection
        if kernel == "Matern32-NL" and alpha_NL is not None and beta_NL is not None:
            next_y = rff_sampler.sample(NL_warping(next_x, alpha_NL, beta_NL))
            next_y_true = next_y.clone().detach()
        elif kernel == "Matern32-HS" and alpha_HS is not None and beta_HS is not None:
            next_y = rff_sampler.sample(next_x)
            next_y, next_y_true = HS_noise_sampling(next_x, next_y, sigma_min, sigma_max, alpha_HS, beta_HS)
        else:
            next_y = rff_sampler.sample(next_x)
            next_y_true = next_y.clone().detach()

        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_clean_arr[i, :] = next_y_true.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_clean_arr, y_init, y_best_arr