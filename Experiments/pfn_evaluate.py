import torch
import botorch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

def eval_pfn(pfn: TransformerBOMethod, train_x, train_y, x):
    raw_model = pfn.model
    criterion = raw_model.criterion

    y_grid = criterion.borders.clone().detach()
    
    with torch.no_grad():
        logits = raw_model((train_x, train_y), x)
        
    # Convert logits to probability masses over the bins
    probabilities = torch.softmax(logits, dim=-1)
    
    # Calculate Expected Value E[Y]
    mu_pred = torch.sum(probabilities * y_grid, dim=-1)
    
    # Calculate Variance sum(p_i * (y_i - mu)^2) 
    # (Using unsqueeze to properly broadcast the [1] mean against the [B] grid)
    var_pred = torch.sum(probabilities * (y_grid - mu_pred.unsqueeze(-1))**2, dim=-1)
    return mu_pred, var_pred