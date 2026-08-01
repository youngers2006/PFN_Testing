import torch
import botorch
import pfns4bo
from pfns4bo.scripts.acquisition_functions import TransformerBOMethod

def eval_pfn(pfn: TransformerBOMethod, train_x, train_y, x):
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