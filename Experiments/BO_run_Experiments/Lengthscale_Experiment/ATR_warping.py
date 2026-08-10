import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod
import math

class ATR_warped_PFN():
    """
    Anisotropic trust region warping for PFNs
    """
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
        self.device = device

    def unscale_outputs(self, mu_y_out_s, var_y_out_s, mu_y, std_y):
        mu_y_out_us = (mu_y_out_s * std_y) + mu_y
        var_y_out_us = var_y_out_s * (std_y ** 2)
        return mu_y_out_us, var_y_out_us

    def standardise(self, y):
        mu_y = torch.mean(y, dim=0)
        std_y = torch.std(y, dim=0, unbiased=False)
        y = (y - mu_y) / torch.clamp(std_y, min=1e-8)
        return y, mu_y, std_y

    def KTA_approx(self, y: torch.Tensor, x: torch.Tensor, lr=0.1, n_iters=200):
        # Get shape of inputs
        N, d = x.shape

        # Ensure there is enough points to do the approximation
        if N < 2:
            return torch.ones((d,), dtype=x.dtype, device=x.device)

        # Add second dimension to y if not already present
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        y_centred = y - torch.mean(y, dim=0)
        K_y = torch.matmul(y_centred, y_centred.T)
        norm_K_y = torch.sqrt(torch.sum(K_y ** 2))+ self.eps

        diff_sq = (x.unsqueeze(1) - x.unsqueeze(0)) ** 2
        
        # Initialise parameter and optimiser
        initial_log_l = math.log(self.l_target)
        log_l = torch.nn.Parameter(initial_log_l * torch.zeros(d, dtype=x.dtype, device=x.device))
        optimizer = torch.optim.Adam([log_l], lr=lr)

        sqrt_3 = math.sqrt(3.0)

        for _ in range(n_iters):
            optimizer.zero_grad()

            l_sq = torch.exp(2.0 * log_l)
            r_sq = torch.sum(diff_sq / l_sq, dim=-1)
            r = torch.sqrt(r_sq + 1e-8)

            K_x = (1 + sqrt_3 * r) * torch.exp(-sqrt_3 * r)
            norm_K_x = torch.sqrt(torch.sum(K_x ** 2))+ self.eps

            inner_product = torch.sum(K_x * K_y)
            alignment = inner_product / (norm_K_x * norm_K_y)

            loss = -alignment
            loss.backward()
            optimizer.step()

        l_opt = torch.exp(log_l).detach()
        R = 1.0 / (l_opt + self.eps)
        return R
    
    def formulate_trust_regions(self, R, x_opt, x, y, k_min_samples=4):
        # Guardrails for approximation
        N, d = x.shape
        if N < 2:
            ones = torch.ones((d,), dtype=x.dtype, device=x.device)
            zeros = torch.zeros((d,), dtype=x.dtype, device=x.device)
            return zeros, ones, x, y, ones

        # Compute ideal tr size
        L_raw = 1 / (self.l_target * (R + self.eps))
        L_min = self.L_min * torch.ones_like(L_raw)
        L_max = self.L_max * torch.ones_like(L_raw)

        # K-NN guardrail
        if N >= k_min_samples:
            dist_warped = torch.abs(x - x_opt) / (L_raw + self.eps)
            dist_max_warped = torch.max(dist_warped, dim=-1).values
            
            _, top_k_idx = torch.topk(dist_max_warped, k=k_min_samples, largest=False)
            
            x_k = x[top_k_idx]
            
            max_dist_physical = torch.max(torch.abs(x_k - x_opt), dim=0).values
            
            L_density = 2.0 * max_dist_physical
        else:
            L_density = torch.zeros_like(L_raw)

        # Compute trust region size needed to warp the space globally
        L = torch.clamp(L_raw, min=torch.clamp(L_density, min=L_min), max=L_max)

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

    def input_warping(self, x, x_U, x_L, L):
        u = torch.where(L >= 1.0, 1.0 / L, torch.ones_like(L))
        z = u * (x - x_L) / (x_U - x_L)
        return z

    def output_warping(self, z, x_U, x_L, L):
        u = torch.where(L >= 1.0, 1.0 / L, torch.ones_like(L))
        x = (z / u) * (x_U - x_L) + x_L
        return x

    def sample_points(self, n_acq_points, L):
        d = L.shape[0]
        u = torch.where(L >= 1.0, 1.0 / L, torch.ones_like(L))
        sobol = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
        z_unit = sobol.draw(n_acq_points).to(device=L.device, dtype=L.dtype)
        z_acq_points = z_unit * u
        return z_acq_points

    def observe_and_suggest(self, x, y, x_opt, n_acq_points=10000, k_min_samples=4, return_prediction=True):
        # Make sure all on correct device
        x = x.to(device=self.device)
        y = y.to(device=self.device)
        x_opt = x_opt.to(device=self.device)

        # Standardise y
        y_scaled, mu_y, std_y = self.standardise(y)

        # Compute roughness parameter and find trust region
        R = self.KTA_approx(y_scaled, x)
        x_L, x_U, x_tr, y_tr, L = self.formulate_trust_regions(
            R, x_opt, x, y_scaled, k_min_samples=k_min_samples
        )

        # Prevent clustered points causing a crash
        std_tr = torch.std(y_tr, unbiased=False)
        if torch.isnan(std_tr) or std_tr < 1e-4:
            y_tr = y_tr + 1e-4 * torch.randn_like(y_tr)
        else:
            med_tr = torch.median(y_tr)
            mad_tr = torch.median(torch.abs(y_tr - med_tr))
            if mad_tr < 1e-4 * std_tr:
                y_tr = y_tr + (0.05 * std_tr) * torch.randn_like(y_tr)

        # Warp inputs
        z_tr = self.input_warping(x_tr, x_U, x_L, L)

        # Sample inside warped trust region
        acq_points = self.sample_points(n_acq_points, L)

        try:
            candidate_idx, acq_value = self.pfn.observe_and_suggest(
                z_tr,
                y_tr,
                acq_points,
                return_actual_ei=True
            )
        except Exception as e:
            print("\n[YEO-JOHNSON CRASH DIAGNOSTIC]")
            print(f"N points in y_tr: {len(y_tr)}")
            print(f"Raw y_tr values:\n{y_tr.cpu().numpy().flatten()}")
            print(f"y_tr std: {torch.std(y_tr).item():.6f}")
            raise e
        
        z_selected = acq_points[candidate_idx]
        x_selected = self.output_warping(z_selected, x_U, x_L, L)

        if x_selected.ndim == 1:
            x_selected = x_selected.unsqueeze(0)
        if return_prediction:
            mu_US, var_US = self.eval_pfn(z_tr, y_tr, z_selected)
            mu, var = self.unscale_outputs(mu_US, var_US, mu_y, std_y)
            return x_selected, acq_value, L, mu, var
        else:
            return x_selected, acq_value, L

    def eval_pfn(self, train_z, train_y, z):
        raw_model = self.pfn.model
        criterion = raw_model.criterion

        # Compute bin centres
        borders = criterion.borders.clone().detach()
        y_grid = (borders[:-1] + borders[1:]) / 2.0

        # Make input 2d
        if z.dim() == 1:
            z = z.unsqueeze(0)
            
        # add query point into the input sequence
        X_seq = torch.cat([train_z, z], dim=0).unsqueeze(1) 

        # pad output sequence to match input
        dummy_y = torch.zeros((z.shape[0], train_y.shape[1]), dtype=train_y.dtype, device=train_y.device)
        Y_seq = torch.cat([train_y, dummy_y], dim=0).unsqueeze(1)

        with torch.no_grad():
            # Cast inputs to floats to match tansformer internals
            logits = raw_model(
                (X_seq.to(dtype=torch.float32), Y_seq.to(dtype=torch.float32)), 
                single_eval_pos=len(train_z)
            )
            
        # remove sequence and batch single dims
        logits = logits.squeeze()

        # Convert logits to pobabilities over bins
        probabilities = torch.softmax(logits, dim=-1)
        mu_pred = torch.sum(probabilities * y_grid, dim=-1)
        var_pred = torch.sum(probabilities * (y_grid - mu_pred.unsqueeze(-1))**2, dim=-1)

        # Ensure outputs are 1D to match standard BO dimension handling
        if mu_pred.dim() == 0:
            mu_pred = mu_pred.unsqueeze(0)
            var_pred = var_pred.unsqueeze(0)
            
        # Cast back to double to match storage containers
        return mu_pred.to(torch.float64), var_pred.to(torch.float64)