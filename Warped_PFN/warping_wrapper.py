import torch
import pfns4bo
from pfns4bo import Tra

class Input_warping_PFN_wrapper():
    def init(device='cpu'):
        model_path = pfns4bo.hebo_plus_model
        pfn = TransformerBOMethod(torch.load(model_path, weights_only=False), device='cuda')