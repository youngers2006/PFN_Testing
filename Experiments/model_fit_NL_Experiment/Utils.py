import torch
import numpy as np

def to_numpy(val):
    if torch.is_tensor(val):
        return val.detach().cpu().numpy()
    return val

def input_normalise(x, bounds):
    x_max = bounds[0]
    x_min = bounds[1]
    x = (x - x_min) / torch.clamp(x_max - x_min, min=1e-8)
    return x

def output_standardise(x):
    mu_x = torch.mean(x, dim=0)
    std_x = torch.std(x, dim=0)
    x = (x - mu_x) / torch.clamp(std_x, min=1e-8)
    return x, mu_x, std_x

def unscale_outputs(mu_y_out_s, var_y_out_s, mu_y, std_y):
    mu_y_out_us = (mu_y_out_s * std_y) + mu_y
    var_y_out_us = var_y_out_s * (std_y ** 2)
    return mu_y_out_us, var_y_out_us