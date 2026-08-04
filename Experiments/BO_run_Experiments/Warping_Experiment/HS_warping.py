import torch
from scipy.stats import beta as scipy_beta
from torch.distributions.normal import Normal

def HS_noise_sampling(x_targets, true_function_outputs, sigma_min, sigma_max, alpha, beta):
    # Get informatio from inputs
    device, dtype = x_targets.device, x_targets.dtype

    x_scalar = torch.mean(x_targets, dim=1, keepdim=True)
    x_scalar = torch.clamp(x_scalar, min=1e-10, max=1.0-1e-10)

    x_scalar_np = x_scalar.detach().cpu().numpy()
    noise_multiplier_np = scipy_beta.cdf(x_scalar_np, alpha, beta)
    
    noise_multiplier = torch.tensor(noise_multiplier_np, dtype=dtype, device=device)

    t_sig_min = torch.tensor(sigma_min, device=device, dtype=dtype)
    t_sig_max = torch.tensor(sigma_max, device=device, dtype=dtype)

    sigma_x = t_sig_min + (t_sig_max - t_sig_min) * noise_multiplier

    # Sample noise from normal distribution
    noise_dist = Normal(
        loc=torch.zeros_like(sigma_x), 
        scale=sigma_x
    )
    epsilon = noise_dist.rsample()
    
    return true_function_outputs + epsilon, true_function_outputs