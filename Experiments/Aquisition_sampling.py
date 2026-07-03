import torch
import botorch
from botorch.optim import optimize_acqf_discrete

def generate_query_points(bounds, n_samples, seed):
    
    return x_test

def optimise_over_set(acq_fn, x_query, q=1):
    candidate, acq_value = optimize_acqf_discrete(
        acq_function=acq_fn, 
        q=q, 
        choices=x_query
    )
    return candidates, acq_values