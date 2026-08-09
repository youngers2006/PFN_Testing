import torch
from tqdm import tqdm
from RFF import RFFSampler
from ATR_warping import ATR_warped_PFN

# Supress warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pfns4bo")

def Experiment_ATR_PFN(atr_pfn: ATR_warped_PFN, rff_sampler: RFFSampler, x_train, N_iters, n_acq_points, restarts=False, restart_bound=5):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Move devices
    y_train = y_train.to(device=atr_pfn.device)
    x_train = x_train.to(device=atr_pfn.device)

    # Get best point
    max_idx = torch.argmax(y_train)
    x_best = x_train[max_idx]

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
    tr_arr = torch.zeros(N_iters, x_train.shape[1], dtype=torch.float64, device='cpu')

    if restarts:
        fail_counter = 0

    x_anchor = x_best
    y_anchor = y_train[max_idx].item()

    # BO test
    for i in tqdm(range(N_iters), desc="PFN", leave=False):

        next_x, acq_value, tr_size, candidate_mean, candidate_var = atr_pfn.observe_and_suggest(
            x_train, y_train, x_anchor, n_acq_points=n_acq_points, return_prediction=True
        )

        # Evaluate and add the new point to the training set
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        if restarts:
            next_y_val = next_y.item()
            if next_y_val > y_anchor + 1e-4:
                x_anchor = next_x
                y_anchor = next_y_val
                fail_counter = 0
            else:
                fail_counter += 1

            if fail_counter > restart_bound:
                restart_x = torch.rand((1, x_train.shape[1]), dtype=x_train.dtype, device=atr_pfn.device)
                restart_y = rff_sampler.sample(restart_x)
                
                x_train = torch.cat([x_train, restart_x])
                y_train = torch.cat([y_train, restart_y])
                
                x_anchor = restart_x
                y_anchor = restart_y.item()
                fail_counter = 0
        else:
            best_idx = torch.argmax(y_train)
            x_anchor = x_train[best_idx].unsqueeze(0) 
            y_anchor = y_train[best_idx].item()

        # Record data
        x_query_arr[i, :] = next_x.detach().cpu()
        y_true_arr[i, :] = next_y.detach().cpu()
        y_best_arr[i, :] = torch.max(y_train, dim=0).values.detach().cpu()
        mu_arr[i, :] = candidate_mean.detach().cpu()
        var_arr[i, :] = candidate_var.detach().cpu()
        alpha_arr[i, :] = acq_value.detach().cpu().flatten()[0]
        tr_arr[i, :] = tr_size.detach().cpu()

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, tr_arr