import os
import torch
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from typing import Tuple
from tqdm import tqdm

from RFF import RFFSampler

# Ensure your RFFSampler is accessible in this namespace
# from rff_sampler import RFFSampler 

def rff_objective_wrapper(x_np: np.ndarray, sampler: 'RFFSampler') -> float:
    # Convert numpy array shape (d,) to tensor shape (1, d)
    x_tensor = torch.tensor(x_np, dtype=torch.float64).unsqueeze(0)
    
    # Disable autograd to speed up evaluations
    with torch.no_grad():
        y_tensor = sampler.sample(x_targets=x_tensor)

    # Negate to ensure maximisation
    y_tensor = -y_tensor
        
    return y_tensor.item()

def compute_true_rff_maximum(
    problem_path: str, 
    num_sobol_points: int = 100000,
    num_restarts: int = 10,
    bounds_tuple: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, float]:
    #Load problem
    sampler = RFFSampler(
        num_features=1,
        input_dim=1,
        number_of_functions=1,
        lengthscale = 1, 
        amplitude = 1,
    )
    sampler.load_problem_from_file(problem_path)
    d = sampler.input_dim
    lb, ub = bounds_tuple
    sobol_engine = qmc.Sobol(d=d, scramble=True)
    sobol_X_unit = sobol_engine.random(n=num_sobol_points)
    
    # Scale points to the defined domain bounds
    sobol_X = sobol_X_unit * (ub - lb) + lb
    
    # Evaluate
    X_tensor = torch.tensor(sobol_X, dtype=torch.float64)
    with torch.no_grad():
        Y_tensor = sampler.sample(x_targets=X_tensor)
    
    # Convert to scipy allowed format
    sobol_Y = Y_tensor.cpu().numpy().flatten()
    
    # Get best positions
    best_indices = np.argsort(-sobol_Y)[:num_restarts]
    initial_guesses = sobol_X[best_indices]
    
    # Optimise at best points
    bnds = [bounds_tuple for _ in range(d)]
    opt_options = {'ftol': 1e-12, 'gtol': 1e-9, 'maxiter': 2000}
    
    global_max_val = -np.inf
    global_max_loc = None
    
    # Optimise
    for i, x0 in enumerate(initial_guesses):
        res = minimize(
            fun=rff_objective_wrapper,
            x0=x0,
            args=(sampler,),
            method='L-BFGS-B',
            bounds=bnds,
            options=opt_options,
            jac=False  # SciPy computes numerical gradients via finite difference
        )
        
        # Update if maxima found
        if res.success and -res.fun > global_max_val:
            global_max_val = -res.fun
            global_max_loc = res.x
            
    # If contiuous surface is not possible to maximise, maintain sobol maxima
    if global_max_loc is None:
        best_idx = np.argmax(sobol_Y)
        return sobol_X[best_idx], sobol_Y[best_idx]
        
    return global_max_loc, global_max_val

def main():
    # List of dimensions
    dims = [5]
    n_dims = 1

    # Number of repeats
    n_repeats = 21

    # Number of tests
    n_tests = 9

    # Initialise storage
    maxima_store_y = [torch.zeros((n_tests, n_repeats, 1), dtype=torch.float64, device='cpu') for _ in dims]
    maxima_store_x = [torch.zeros((n_tests, n_repeats, dim), dtype=torch.float64, device='cpu') for dim in dims]

    # Filepath
    base_save_dir = "results_LS_BO_PFN_GP/run_N_50_D_5"

    for test in tqdm(range(n_tests), desc="Optimisation Progress"):
        for k in range(n_dims):
            dim = dims[k]
            
            for rep in tqdm(range(n_repeats), desc=f"On test {test}, dimension {dim}", leave=False):
                #filepath = f"results_KT_50/run_20260714_231826/test_{test}/dim_{dim}/repeat_{rep}/problem.npz"
                filepath = f"results_LS_BO_PFN_GP/run_N_50_D_5/test_{test}/dim_{dim}/repeat_{rep}/problem.npz"
                max_x, max_y = compute_true_rff_maximum(filepath)
    
                maxima_store_y[k][test, rep, :] = max_y
                max_x_tensor = torch.tensor(max_x)
                maxima_store_x[k][test, rep, :] = max_x_tensor

    maxima_dict = {
        "x": maxima_store_x,
        "y": maxima_store_y
    }

    filepath = final_save_path = os.path.join(base_save_dir, "maximas.pt")
    torch.save(maxima_dict, filepath)
    return 0
    

if __name__ == "__main__":
    main()