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
from Utils import output_standardise, unscale_outputs

# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pfns4bo")

def Experiment_GP(rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()

    # Initial storage
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], device='cpu')
    mu_arr = torch.zeros(N_iters, y_train.shape[1], device='cpu')
    var_arr = torch.zeros(N_iters, y_train.shape[1], device='cpu')
    alpha_arr = torch.zeros(N_iters, 1, device='cpu')

    # BO test
    for i in tqdm(range(N_iters), desc="GP", leave=False):
        # Setup for GP
        D = x_train.shape[-1]
        matern_32 = MaternKernel(
            nu=1.5, 
            ard_num_dims=D,
            lengthscale_prior=SmoothedBoxPrior(a=0.02 * (D ** 0.5), b=2.5 * (D ** 0.5), sigma=0.01)
        )
        matern_32.raw_lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-4)
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
        noiseless_interval = gpytorch.constraints.Interval(1e-5, 1e-3)
        model.likelihood.noise_covar.register_constraint("raw_noise", noiseless_interval)
    
        # Change noise floor
        model.likelihood.noise_covar.noise = torch.tensor(
            1e-4, 
            dtype=y_train.dtype, 
            device=y_train.device
        )

        # Tune Hyperparams to maximise MLL
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise Acquisition function
        f_best = y_train.max()
        qEI = qExpectedImprovement(model, f_best)
        candidate, acq_value, candidate_mean, candidate_var = optimise_EI_GP(
            acq_fn=qEI, 
            x_query=sobol_acq_points, 
            q=1
        )
        next_x = candidate.to(dtype=torch.float64).detach()

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
        alpha_arr[i, :] = acq_value.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr

def Experiment_PFN(pfn, rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

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
        candidate_mean_scaled, candidate_var_scaled = eval_pfn(pfn, x_train, y_train_scaled, sobol_acq_points[candidate_idx])
        candidate_mean, candidate_var = unscale_outputs(
            candidate_mean_scaled, candidate_var_scaled, mu_y, std_y
        )
        acq_value = acq_value_scaled.cpu() * std_y.cpu()
        next_x = sobol_acq_points[candidate_idx].unsqueeze(0)

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

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr

def Experiment_PFN_Warped(pfn, rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

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

    # BO test
    for i in tqdm(range(N_iters), desc="PFN", leave=False):
        # Standardise inputs
        y_train_scaled, mu_y, std_y = output_standardise(y_train)

        # Create placeholder inputs
        x_train_pfn = x_train.clone()
        sobol_acq_pfn = sobol_acq_points.clone()

        fit_encoder = getattr(pfn, "fit_encoder", None)
        if fit_encoder is not None:
            encoder = fit_encoder(pfn.model, x_train.to(torch.float32), y_train_scaled.to(torch.float32))
                
            # Apply warping w(X; alpha, beta) to coordinates
            x_train_pfn = encoder(x_train_pfn)
            sobol_acq_pfn = encoder(sobol_acq_pfn)
            
        # Maximise Acquisition function
        candidate_idx, acq_value_scaled = pfn.observe_and_suggest(
            x_train_pfn,
            y_train_scaled,
            sobol_acq_pfn,
            return_actual_ei=True
        )
        candidate_mean_scaled, candidate_var_scaled = eval_pfn(pfn, x_train_pfn, y_train_scaled, sobol_acq_pfn[candidate_idx])
        candidate_mean, candidate_var = unscale_outputs(
            candidate_mean_scaled, candidate_var_scaled, mu_y, std_y
        )
        acq_value = acq_value_scaled.cpu() * std_y.cpu()
        next_x = sobol_acq_points[candidate_idx].unsqueeze(0)

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

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr

def Experiment_Random(rff_sampler: RFFSampler, x_train, bounds, N_iters):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Record initial data
    x_init = x_train.clone().detach()
    y_init = y_train.clone().detach()

    # Initial storage (no surrogate or aquisition fn)
    device = x_train.device
    x_query_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')
    y_true_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')
    y_best_arr = torch.zeros(N_iters, y_train.shape[1], dtype=torch.float64, device='cpu')

    for i in tqdm(range(N_iters), desc="Random Baseline", leave=False):
        # Draw random sample point
        u = torch.rand((1, x_train.shape[1]), dtype=torch.float64, device=device)
        next_x = bounds[0] + (bounds[1] - bounds[0]) * u

        # Evaluate sample point and add it to the seen point collection
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr