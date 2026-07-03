import torch
import numpy as np
from scipy.stats import multivariate_t
import os

from typing import Optional, Dict, Any

class RFFSampler:
    """
    RFFSampler: Random Fourier Features Sampler

    Parameters:
        num_features (Optional[int]): number of features to use for the RFF approximation
        input_dim (Optional[int]): number of input dimensions
        number_of_functions (Optional[int]): number of objective functions
        lengthscale (Optional[float]): length-scale of the RBF kernel
        kernel (Optional[str]): kernel type, either "RBF" or "Matern52"
    """

    def __init__(self, 
                 num_features: Optional[int] = None, 
                 input_dim: Optional[int] = None, 
                 number_of_functions: Optional[int] = None, 
                 lengthscale: Optional[float] = None, 
                 kernel: Optional[str] = "RBF",
                 ):
        
        self.num_features = num_features
        self.input_dim = input_dim
        self.output_dim = number_of_functions
        self.lengthscale = lengthscale

        self.omegas = None
        self.phi = None
        self.weights = None
        self.rff_scaling = None
        self.kernel = kernel

        if self.kernel == "RBF":
            # RBF Kernel
            self.omegas = torch.as_tensor(np.random.normal(size=(self.num_features, self.input_dim))) / self.lengthscale
        elif self.kernel == "Matern52":
            # Matern 5/2 Kernel
            self.omegas = torch.as_tensor(multivariate_t.rvs(df=5, size=(self.num_features, self.input_dim))) / self.lengthscale
        elif self.kernel == "Matern32":
            # Matern 3/2 Kernel
            self.omegas = torch.as_tensor(multivariate_t.rvs(df=3, size=(self.num_features, self.input_dim))) / self.lengthscale
        elif self.kernel == "Matern12":
            # Matern 1/2 Kernel
            self.omegas = torch.as_tensor(multivariate_t.rvs(df=1, size=(self.num_features, self.input_dim))) / self.lengthscale

        self.phi = torch.rand(self.num_features, dtype=torch.float64) * 2 * np.pi
        self.weights = torch.randn(self.output_dim, self.num_features, dtype=torch.float64)
        self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))

    def sample(self, x_targets: torch.Tensor):
        """
        Sample the correlated problem at given target points

        Parameters:
            x_targets (torch.Tensor): target points to sample at in decision space

        Returns:
            torch.Tensor: sampled values in objective space
        """

        Z_target = self.rff_scaling * torch.cos(torch.matmul(x_targets, self.omegas.T) + self.phi)
        output_at_target = Z_target @ self.weights.T
        return output_at_target

    def load_problem(
            self, 
            omegas: torch.Tensor, 
            weights: torch.Tensor, 
            phi: torch.Tensor, 
            num_features: int, 
            lengthscale: float, 
            input_dim: int, 
            output_dim: int, 
            kernel: str
            ):
        """
        If you have the omega values already, you can load them into the class using this function.

        Parameters:
            omegas (torch.Tensor): omega values
            weights (torch.Tensor): weights
            phi (torch.Tensor): phi values
            num_features (int): number of features
            lengthscale (float): lengthscale
            input_dim (int): input dimension
            output_dim (int): output dimension
            kernel (str): kernel type, either "RBF" or "Matern52"
        Returns:
            None
        """
        self.omegas = omegas
        self.weights = weights
        self.phi = phi
        self.num_features = num_features
        self.lengthscale = lengthscale
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel = kernel
        self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))

    def load_problem_from_file(self, npzfile_path: str):
        """
        Load the problem from a file generated from the save_problem() function

        Parameters:
            npzfile_path (str): filepath to the problem file

        Returns:
            None
        """
        npzfile = np.load(npzfile_path)

        self.omegas = torch.as_tensor(npzfile['omegas'])
        self.weights = torch.as_tensor(npzfile['weights'])
        self.phi = torch.as_tensor(npzfile['phi'])
        self.num_features = int(npzfile['num_features'])
        self.lengthscale = float(npzfile['lengthscale'])
        self.input_dim = int(npzfile['input_dim'])
        self.output_dim = int(npzfile['output_dim'])

        assert any(
            [self.omegas is not None, self.weights is not None, self.phi is not None, self.num_features is not None, self.lengthscale is not None, self.input_dim is not None,
             self.output_dim is not None])

        self.rff_scaling = torch.sqrt(torch.tensor(2.0 / self.num_features, dtype=torch.float64))
        npzfile.close()

    def save_problem(self, filepath: str, lb: float, ub: float):
        """
        Save the problem to a file

        Parameters:
            filepath (str): path to the file where the problem will be saved
            lb (float): lower bounds of the problem for reference when generating the problem
            ub (float): upper bounds of the problem for reference when generating the problem

        Returns:
            None
        """
        assert any(
            [self.omegas is not None, self.weights is not None, self.phi is not None, self.num_features is not None, self.lengthscale is not None, self.input_dim is not None,
             self.output_dim is not None])

        os.makedirs(filepath, exist_ok=True)
        np.savez(f'{filepath}/problem.npz', omegas=self.omegas, weights=self.weights, phi=self.phi,
                 num_features=self.num_features, lengthscale=self.lengthscale, input_dim=self.input_dim, 
                 output_dim=self.output_dim, lb=lb, ub=ub)