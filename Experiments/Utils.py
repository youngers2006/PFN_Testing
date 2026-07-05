import torch
import numpy as np

def to_numpy(val):
    if torch.is_tensor(val):
        return val.detach().cpu().numpy()
    return val