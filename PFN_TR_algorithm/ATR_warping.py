import torch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

class ATR_warped_PFN():
    def __init__(self, model_path, l_target, L_min=0.01, L_max=1.0, eps=1e-12, device='cpu'):
        model_path = pfns4bo.hebo_plus_model
        self.pfn = TransformerBOMethod(
            torch.load(model_path, weights_only=False), device=device
        )
        self.l_target = l_target
        self.L_min = L_min
        self.L_max = L_max

    def unscale_outputs(mu_y_out_s, var_y_out_s, mu_y, std_y):
        mu_y_out_us = (mu_y_out_s * std_y) + mu_y
        var_y_out_us = var_y_out_s * (std_y ** 2)
        return mu_y_out_us, var_y_out_us

    def standardise(y):
        mu_y = torch.mean(y, dim=0)
        std_y = torch.std(y, dim=0)
        y = (y - mu_y) / torch.clamp(std_y, min=1e-8)
        return y, mu_y, std_y

    def secant_approx(self, y, x):
        R = 0
        return R
    
    def formulate_trust_regions(self, R, x_opt, x, y):
        L = torch.clamp(1 / (self.l_target * R), min=self.L_min, max=self.L_max)

        x_L_ideal = x_opt - (L / 2.0)
        x_U_ideal = x_opt + (L / 2.0)

        overflow_U = torch.clamp(x_U_ideal - 1.0, min=0.0)
        overflow_L = torch.clamp(0.0 - x_L_ideal, min=0.0)

        x_L = x_L_ideal - overflow_U + overflow_L
        x_U = x_U_ideal - overflow_L + overflow_U

        is_smoothing = (L >= 1.0)
        x_L = torch.where(is_smoothing, torch.zeros_like(x_L), torch.clamp(x_L, 0.0, 1.0))
        x_U = torch.where(is_smoothing, torch.ones_like(x_U), torch.clamp(x_U, 0.0, 1.0))

        in_tr = (x >= x_L) & (x <= x_U)
        mask = torch.all(in_tr, dim=-1)

        x_tr = x[mask]
        y_tr = y[mask]

        return x_L, x_U, x_tr, y_tr, L

    def limit_and_warp_search_points(x, x_L, x_U):
        return 0

    def input_warping(self, x, x_U, x_L):
        z = (x - x_L) / (x_U - x_L)
        return z

    def output_warping(self, z, x_U, x_L):
        x = z * (x_U - x_L) + x_L
        return x

    def observe_and_suggest(self, x, y, x_opt, acq_points, return_tr_size=True):
        y_scaled, _, _ = self.standardise(y)

        R = self.secant_approx(y_scaled, x)
        x_L, x_U, x_tr, y_tr, L = self.formulate_trust_regions(R, x_opt, x, y_scaled)

        z = self.input_warping(x_tr, x_U, x_L)
        acq_points_warped = self.limit_and_warp_search_points(acq_points, x_L, x_U)

        candidate_idx, acq_value = self.pfn.observe_and_suggest(
            z,
            y_tr,
            acq_points_warped,
            return_actual_ei=True
        )
        z_selected = acq_points[candidate_idx]
        x_selected = self.output_warping(z_selected, x_U, x_L)

        if return_tr_size:
            return x_selected, acq_value, L
        else:
            return x_selected, acq_value




    def eval_pfn(train_x, train_y, x):
        raw_model = pfn.model
        criterion = raw_model.criterion

        # Compute bin centres
        borders = criterion.borders.clone().detach()
        y_grid = (borders[:-1] + borders[1:]) / 2.0
        
        # Make input 2d
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # add query point into the input sequence
        X_seq = torch.cat([train_x, x], dim=0).unsqueeze(1) 
        
        # pad output sequence to match input
        dummy_y = torch.zeros((x.shape[0], train_y.shape[1]), dtype=train_y.dtype, device=train_y.device)
        Y_seq = torch.cat([train_y, dummy_y], dim=0).unsqueeze(1)
        
        with torch.no_grad():
            # Cast inputs to floats to match tansformer internals
            logits = raw_model(
                (X_seq.to(torch.float32), Y_seq.to(torch.float32)), 
                single_eval_pos=len(train_x)
            )
            
        # remove sequence and batch single dims
        logits = logits.squeeze()
        
        # Convert logits to pobabilities over bins
        probabilities = torch.softmax(logits, dim=-1)
        
        # Calculate Expected Value
        mu_pred = torch.sum(probabilities * y_grid, dim=-1)
        
        # Calculate Variance
        var_pred = torch.sum(probabilities * (y_grid - mu_pred.unsqueeze(-1))**2, dim=-1)
        
        # Ensure outputs are 1D to match standard BO dimension handling
        if mu_pred.dim() == 0:
            mu_pred = mu_pred.unsqueeze(0)
            var_pred = var_pred.unsqueeze(0)
            
        # Cast back to double to match storage containers
        return mu_pred.to(torch.float64), var_pred.to(torch.float64)