import torch
import matplotlib.pyplot as plt
from Experiments.BO_run_Experiments.Warping_Experiment.pfn_evaluate import eval_pfn
from Utils import output_standardise
from Experiments.BO_run_Experiments.Warping_Experiment.Aquisition_sampling import generate_sobol_points
from tqdm import tqdm
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior

def plot_distribution(data_dict, title="2D Regret Curve", y_label="Regret"):
    # Create subplot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Iterate over data
    for label, tensor_data in data_dict.items():
        
        # Ensure the tensor is float for torch.quantile calculation
        tensor_data = tensor_data.float()
        
        # Extract the number of iterations (N) from the tensor shape
        N = tensor_data.shape[1]
        iterations = torch.arange(N)
        
        # Calculate the mathematical boundaries along the repeats axis (dim=0)
        q1 = torch.quantile(tensor_data, 0.25, dim=0)
        median = torch.quantile(tensor_data, 0.50, dim=0)
        q3 = torch.quantile(tensor_data, 0.75, dim=0)
        
        # Plot the median
        line, = ax.plot(iterations, median, label=label, linewidth=2)
        
        # Plot the shaded region representing the Interquartile Range
        ax.fill_between(
            iterations, 
            q1, 
            q3, 
            color=line.get_color(), 
            alpha=0.5 # 20% opacity for the shaded variance region
        )
        
    # Formatting the axes and plot
    ax.set_xlabel("Iteration")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()

def plot_GP_variance_surface(kernel, train_x, train_y, bounds, device='cpu'):
    """
    Evaluates 1000 points across a 2D space sequentially using eval_pfn.
    Plots the coordinates at (x_1, x_2, \mu) with point color mapped to \sigma^2.
    """
    
    # Move data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    bounds = bounds.to(device)
    
    if bounds.shape[1] != 2:
        raise ValueError(f"Plotting requires exactly 2 input dimensions. Received bounds with dim {bounds.shape[1]}.")
    
    # create eval grid
    n_samples = 1000
    x_queries = generate_sobol_points(bounds, n_samples, seed=42, device=device)
    
    mu_list = []
    var_list = []

    D = train_x.shape[-1]
    matern_32 = MaternKernel(
        nu=1.5, 
        ard_num_dims=D,
        lengthscale_prior=GammaPrior(3.0, 6.0)
    )
    custom_covar = ScaleKernel(
        matern_32,
        outputscale_prior=GammaPrior(2.0, 0.15)
    )
    
    # Setup surrogate model
    model = SingleTaskGP(
        train_x, 
        train_y,
        outcome_transform=Standardize(m=train_y.shape[-1]),
        covar_module=custom_covar
    )

    # Tune Hyperparams to maximise MLL
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    model.eval()
    model.likelihood.eval()
    
    # evaluate each point
    for i in tqdm(range(n_samples), desc="Evaluating PFN sequentially"):
        x_q = x_queries[i : i + 1]
        
        with torch.no_grad():
            posterior_latent = model.posterior(x_q, observation_noise=False)
            mu_actual = posterior_latent.mean
            var_actual = posterior_latent.variance
        
        mu_list.append(mu_actual.item())
        var_list.append(var_actual.item())
        
    mu_arr = torch.tensor(mu_list)
    var_arr = torch.tensor(var_list)
    
    # Offload to cpu
    x1 = x_queries[:, 0].cpu().numpy()
    x2 = x_queries[:, 1].cpu().numpy()
    mu_np = mu_arr.numpy()
    var_np = var_arr.numpy()
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        x1, x2, mu_np, 
        c=var_np, 
        cmap='inferno', 
        marker='o', 
        alpha=0.85,
        s=30,
        edgecolors='none'
    )
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Predictive Variance ($\sigma^2$)', rotation=270, labelpad=20)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Predictive Mean ($\mu$)')
    ax.set_title('PFN Predictive Landscape: Expected Value Surface with Variance Temperature Map')
    
    train_x1 = train_x[:, 0].cpu().numpy()
    train_x2 = train_x[:, 1].cpu().numpy()
    train_y_np = train_y.cpu().numpy().flatten()
    
    ax.scatter(
        train_x1, train_x2, train_y_np, 
        color='green', 
        edgecolor='black', 
        s=120, 
        label='Evaluated Points',
        marker='o',
        depthshade=False,
        zorder=10
    )
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_pfn_variance_surface(pfn, train_x, train_y, bounds, device='cpu'):
    """
    Evaluates 1000 points across a 2D space sequentially using eval_pfn.
    Plots the coordinates at (x_1, x_2, \mu) with point color mapped to \sigma^2.
    """
    
    # Move data
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    bounds = bounds.to(device)
    pfn.model.to(device)
    
    if bounds.shape[1] != 2:
        raise ValueError(f"Plotting requires exactly 2 input dimensions. Received bounds with dim {bounds.shape[1]}.")
    
    # standardise
    train_y_scaled, mu_y, std_y = output_standardise(train_y)
    
    # create eval grid
    n_samples = 1000
    x_queries = generate_sobol_points(bounds, n_samples, seed=42, device=device)
    
    mu_list = []
    var_list = []
    
    pfn.model.eval()
    
    # evaluate each point
    for i in tqdm(range(n_samples), desc="Evaluating PFN sequentially"):
        x_q = x_queries[i]
        
        with torch.no_grad():
            mu_pred_scaled, var_pred_scaled = eval_pfn(pfn, train_x, train_y_scaled, x_q)
        
        # unscale output
        mu_actual = mu_pred_scaled * std_y.squeeze() + mu_y.squeeze()
        var_actual = var_pred_scaled * (std_y.squeeze() ** 2)
        
        mu_list.append(mu_actual.item())
        var_list.append(var_actual.item())
        
    mu_arr = torch.tensor(mu_list)
    var_arr = torch.tensor(var_list)
    
    # Offload to cpu
    x1 = x_queries[:, 0].cpu().numpy()
    x2 = x_queries[:, 1].cpu().numpy()
    mu_np = mu_arr.numpy()
    var_np = var_arr.numpy()
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        x1, x2, mu_np, 
        c=var_np, 
        cmap='inferno', 
        marker='o', 
        alpha=0.85,
        s=30,
        edgecolors='none'
    )
    
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, shrink=0.7)
    cbar.set_label('Predictive Variance ($\sigma^2$)', rotation=270, labelpad=20)
    
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('Predictive Mean ($\mu$)')
    ax.set_title('PFN Predictive Landscape: Expected Value Surface with Variance Temperature Map')
    
    train_x1 = train_x[:, 0].cpu().numpy()
    train_x2 = train_x[:, 1].cpu().numpy()
    train_y_np = train_y.cpu().numpy().flatten()
    
    ax.scatter(
        train_x1, train_x2, train_y_np, 
        color='green', 
        edgecolor='black', 
        s=120, 
        label='Evaluated Points',
        marker='o',
        depthshade=False,
        zorder=10
    )
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def get_training_data(data_dict, test_idx, dim_idx, method_idx, repeat_idx, iteration):
    """
    Reconstructs the full x_train and y_train tensors from the saved experimental data.
    
    Parameters:
        data_dict (dict): The loaded dictionary from 'experimental_results.pt'
        dim_idx (int): Index of test (0 for Matern12, 1 for Matern 52, 2 for RBF)
        dim_idx (int): Index of the dimension (0 for 2D, 1 for 5D, 2 for 10D)
        method_idx (int): Index of the method (0 for GP, 1 for PFN, 2 for Random)
        repeat_idx (int): Index of the repeat (0 to 20)
        
    Returns:
        x_train (torch.Tensor): The fully reconstructed input space training points.
        y_train (torch.Tensor): The fully reconstructed objective space training points.
    """
    
    # Extract Initial Points
    x_init = data_dict["x_init"][dim_idx][test_idx, method_idx, repeat_idx, :, :]
    
    # Shape: (5 * x_dim, n_fns)
    y_init = data_dict["y_init"][dim_idx][test_idx, method_idx, repeat_idx, :, :]
    
    # Get queried points
    x_query = data_dict["x_query"][dim_idx][test_idx, method_idx, repeat_idx, :iteration, :]
    y_true = data_dict["y_true"][dim_idx][test_idx, method_idx, repeat_idx, :iteration, :]
    
    # Concatenating to get full tensor
    x_train = torch.cat([x_init, x_query], dim=0)
    y_train = torch.cat([y_init, y_true], dim=0)
    
    return x_train, y_train