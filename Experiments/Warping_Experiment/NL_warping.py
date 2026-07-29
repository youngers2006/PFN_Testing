import torch
from scipy.stats import beta as scipy_beta

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