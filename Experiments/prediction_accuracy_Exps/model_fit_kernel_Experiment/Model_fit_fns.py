import torch
import matplotlib.pyplot as plt
from pfn_evaluate import eval_pfn
from Utils import output_standardise
from Aquisition_sampling import generate_sobol_points
from tqdm import tqdm
from botorch.models import SingleTaskGP
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior, SmoothedBoxPrior

def plot_GP_variance_surface(train_x, train_y, x_queries, y_true, n_samples=10000, device='cpu'):
    """
    Evaluates 1000 points across a 2D space sequentially using eval_pfn.
    Plots the coordinates at (x_1, x_2, \mu) with point color mapped to \sigma^2.
    """
    
    # Move data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    mu_list = []
    var_list = []

    D = train_x.shape[-1]
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
        train_x, 
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1]),
        covar_module=custom_covar
    )
    noiseless_interval = gpytorch.constraints.Interval(1e-5, 1e-3)
    model.likelihood.noise_covar.register_constraint("raw_noise", noiseless_interval)

    # Change noise floor
    model.likelihood.noise_covar.noise = torch.tensor(
        1e-4, 
        dtype=train_y.dtype, 
        device=train_y.device
    )

    # Tune Hyperparams to maximise MLL
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()

    # evaluate each point
    for i in tqdm(range(n_samples), desc="Evaluating GP sequentially", leave=False):
        x_q = x_queries[i : i + 1]
        
        with torch.no_grad():
            posterior_latent = model.posterior(x_q, observation_noise=False)
            mu_actual = posterior_latent.mean
            var_actual = posterior_latent.variance
        
        mu_list.append(mu_actual.item())
        var_list.append(var_actual.item())
        
    mu_arr = torch.tensor(mu_list).unsqueeze(-1)
    var_arr = torch.tensor(var_list).unsqueeze(-1)
    
    return mu_arr, var_arr, y_true
    
def plot_pfn_variance_surface(pfn, train_x, train_y, x_queries, y_true, n_samples=10000, warping=False, device='cpu'):
    """
    Evaluates 1000 points across a 2D space sequentially using eval_pfn.
    Plots the coordinates at (x_1, x_2, \mu) with point color mapped to \sigma^2.
    """
    
    # Move data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    pfn.model.to(device)
    
    # standardise
    train_y_scaled, mu_y, std_y = output_standardise(train_y)
    
    mu_list = []
    var_list = []
    
    pfn.model.eval()

    fit_encoder = getattr(pfn, "fit_encoder", None)
    if warping and fit_encoder is not None:
        encoder = fit_encoder(pfn.model, train_x.to(torch.float32), train_y.to(torch.float32))
            
        # Apply warping w(X; alpha, beta) to coordinates
        train_x = encoder(train_x)
        x_queries = encoder(x_queries)
    
    # evaluate each point
    for i in tqdm(range(n_samples), desc="Evaluating PFN sequentially", leave=False):
        x_q = x_queries[i]
        
        with torch.no_grad():
            mu_pred_scaled, var_pred_scaled = eval_pfn(
                pfn, train_x, train_y_scaled, x_q
            )
        
        # unscale output
        mu_actual = mu_pred_scaled * std_y.squeeze() + mu_y.squeeze()
        var_actual = var_pred_scaled * (std_y.squeeze() ** 2)
        
        mu_list.append(mu_actual.item())
        var_list.append(var_actual.item())
        
    mu_arr = torch.tensor(mu_list).unsqueeze(-1)
    var_arr = torch.tensor(var_list).unsqueeze(-1)
    
    return mu_arr, var_arr, y_true