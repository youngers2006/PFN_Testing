import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

class ATR_warped_PFN():
    def __init__(
            self, 
            l_target, 
            L_min=0.01, 
            L_max=1.0, 
            eps=1e-12, 
            model_path=None, 
            device='cpu'
        ):
        if model_path is None:
            model_path = pfns4bo.hebo_plus_model

        self.pfn = TransformerBOMethod(
            torch.load(model_path, weights_only=False), device=device
        )
        self.l_target = l_target
        self.L_min = L_min
        self.L_max = L_max
        self.eps = eps

    def unscale_outputs(self, mu_y_out_s, var_y_out_s, mu_y, std_y):
        mu_y_out_us = (mu_y_out_s * std_y) + mu_y
        var_y_out_us = var_y_out_s * (std_y ** 2)
        return mu_y_out_us, var_y_out_us

    def standardise(self, y):
        mu_y = torch.mean(y, dim=0)
        std_y = torch.std(y, dim=0)
        y = (y - mu_y) / torch.clamp(std_y, min=1e-8)
        return y, mu_y, std_y

    def secant_approx(self, y: torch.Tensor, x: torch.Tensor):
        # Get shape of inputs
        N, _ = x.shape

        # Add second dimension to y if not already present
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        # Create p and q indices to get slopes with an upper triangular matrix
        idx = torch.triu_indices(N, N, offset=1, device=x.device)
        p_idx = idx[0, :] ; q_idx = idx[1, :]

        # Compute secant slopes
        x_p = x[p_idx, :] ; x_q = x[q_idx, :]
        y_p = y[p_idx, :] ; y_q = y[q_idx, :]
        slopes = torch.abs(y_p - y_q) / (torch.abs(x_p - x_q) + self.eps)

        # Compute roughness parameter as the median secant slope
        R = torch.median(slopes, dim=0).values
        return R
    
    def formulate_trust_regions(self, R, x_opt, x, y):
        # Guardrails for approximation
        N, d = x.shape
        if N < 2:
            return torch.ones((d,), dtype=x.dtype, device=x.device)

        # Compute trust region size needed to warp the space globally
        L = torch.clamp(1 / (self.l_target * (R + self.eps)), min=self.L_min, max=self.L_max)

        # Compute raw upper and lower trust region limits when centred around the current optima
        x_L_ideal = x_opt - (L / 2.0)
        x_U_ideal = x_opt + (L / 2.0)

        # Calculate how much the trust region overflows the 0-1 space
        overflow_U = torch.clamp(x_U_ideal - 1.0, min=0.0)
        overflow_L = torch.clamp(0.0 - x_L_ideal, min=0.0)

        # Shift the trust region back into the 0-1 space
        x_L = x_L_ideal - overflow_U + overflow_L
        x_U = x_U_ideal - overflow_U + overflow_L

        # When smoothing the trust region will be larger than the space so return a 0-1 space
        is_smoothing = (L >= 1.0)
        x_L = torch.where(is_smoothing, torch.zeros_like(x_L), torch.clamp(x_L, 0.0, 1.0))
        x_U = torch.where(is_smoothing, torch.ones_like(x_U), torch.clamp(x_U, 0.0, 1.0))

        # Only use the x and y samples inside the trust region
        in_tr = (x >= x_L) & (x <= x_U)
        mask = torch.all(in_tr, dim=-1)
        x_tr = x[mask]
        y_tr = y[mask]

        return x_L, x_U, x_tr, y_tr, L

    def input_warping(self, x, x_U, x_L, R):
        scale = 1.0 / (self.l_target * (R + self.eps))
        u = torch.where(scale >= 1.0, 1.0 / scale, torch.ones_like(scale))
        z = u * (x - x_L) / (x_U - x_L)
        return z

    def output_warping(self, z, x_U, x_L, R):
        scale = 1.0 / (self.l_target * (R + self.eps))
        u = torch.where(scale >= 1.0, 1.0 / scale, torch.ones_like(scale))
        x = (z / u) * (x_U - x_L) + x_L
        return x

    def sample_points(self, n_acq_points, R):
        # Compute scaling factor for global warping
        scale = 1.0 / (self.l_target * (R + self.eps))

        d = scale.shape[0]

        u = torch.where(scale >= 1.0, 1.0 / scale, torch.ones_like(scale))

        sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        z_unit = sobol.draw(n_acq_points).to(device=scale.device, dtype=scale.dtype)

        z_acq_points = z_unit * u
        return z_acq_points

    def observe_and_suggest(self, x, y, x_opt, n_acq_points=10000, return_tr_size=True):
        # Standardise y
        y_scaled, _, _ = self.standardise(y)

        # Compute roughness parameter and find trust region
        R = self.secant_approx(y_scaled, x)
        x_L, x_U, x_tr, y_tr, L = self.formulate_trust_regions(R, x_opt, x, y_scaled)

        # Warp inputs
        z = self.input_warping(x_tr, x_U, x_L, R)

        # Sample inside warped trust region
        acq_points = self.sample_points(n_acq_points, R)
        
        candidate_idx, acq_value = self.pfn.observe_and_suggest(
            z,
            y_tr,
            acq_points,
            return_actual_ei=True
        )
        z_selected = acq_points[candidate_idx]
        x_selected = self.output_warping(z_selected, x_U, x_L, R)

        if return_tr_size:
            return x_selected, acq_value, L
        else:
            return x_selected, acq_value