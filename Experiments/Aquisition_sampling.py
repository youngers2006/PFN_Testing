import torch
import botorch
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolEngine

def generate_sobol_points(bounds:torch.Tensor, n_samples, seed, device):
    # Extract dimensions of problem
    dim = bounds.shape[1]
    
    # Instantiate sobol sampling engine
    sobol = SobolEngine(
        dimension=dim, 
        scramble=True, 
        seed=seed
    )
    
    # Draw values in unit hypercube
    x_raw = sobol.draw(n_samples).to(dtype=bounds.dtype).to(device=device)

    # Scale samples to match problem bounds
    x_scaled = bounds[0] + (bounds[1] - bounds[0]) * x_raw
    return x_scaled

def optimise_EI_GP(acq_fn, x_query, q=1):
    # Run optimisation to obtain candidate point to maximise acq fn
    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_fn,
        q=q,
        choices=x_query
    )

    # Evaluate model to obtain 
    with torch.no_grad():
        # Get posterior prediction
        posterior = acq_fn.model.posterior(candidate.unsqueeze(-2))
        
        # Obtain mean and variance prediction
        candidate_mean = posterior.mean.squeeze()
        candidate_variance = posterior.variance.squeeze()
    return candidate, acq_value, candidate_mean, candidate_variance