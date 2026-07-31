import torch
from scipy.stats import beta as scipy_beta
from torch.distributions import Beta

def NL_warping(x_targets, alpha, beta):
# Warp inputs with beta distribution to create nonlinearity in [0,1]

    # Get x parameters
    device, dtype = x_targets.device, x_targets.dtype

    # Warp x with beta cdf back onto the same bounds [0,1]
    x_targets_clamped = torch.clamp(x_targets, min=1e-10, max=1 - 1e-10)
    x_np = x_targets_clamped.detach().cpu().numpy()
    
    w_x_np = scipy_beta.cdf(x_np, alpha, beta)
    w_x = torch.tensor(w_x_np, dtype=dtype, device=device)
    return w_x

def get_distortion_ratio(
    alpha: float | torch.Tensor, 
    beta_param: float | torch.Tensor, 
    bounds: tuple[float, float] = (0.05, 0.95),
    device: str = "cpu"):
    a, b = bounds
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta_param, dtype=torch.float32, device=device)

    candidates = [
        torch.tensor(a, dtype=torch.float32, device=device),
        torch.tensor(b, dtype=torch.float32, device=device)
    ]
    
    if (alpha_t != 1.0) or (beta_t != 1.0):
        denom = alpha_t + beta_t - 2.0
        if torch.abs(denom) > 1e-12:
            x_crit = (alpha_t - 1.0) / denom
            if a <= x_crit <= b:
                candidates.append(x_crit)
                
    x_candidates = torch.stack(candidates)
    
    dist = Beta(alpha_t, beta_t)
    slopes = torch.exp(dist.log_prob(x_candidates))
    
    f_max = torch.max(slopes)
    f_min = torch.min(slopes)
    f_min = torch.clamp(f_min, min=1e-30)
    
    return f_max / f_min