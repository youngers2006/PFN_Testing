import torch
import gpytorch
import botorch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll, ExactMarginalLogLikelihood
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from tqdm import tqdm
from RFF import RFFSampler

def Experiment_GP(rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Record initial data
    x_init = x_train
    y_init = y_train

    # Initial storage
    x_query_arr = torch.zeros()
    y_true_arr = torch.zeros()
    y_best_arr = torch.zeros()
    mu_arr = torch.zeros()
    var_arr = torch.zeros()
    alpha_arr = torch.zeros()
    R_arr = torch.zeros()

    # BO test
    for i in tqdm(range(N_iters)):
        # Setup surrogate model
        model = SingleTaskGP(x_train, y_train)

        # Tune Hyperparams to maximise MLL
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise Acquisition function (need to change to output mu and var)
        f_best = y_train.max()
        qEI = qLogExpectedImprovement(model, f_best)
        candidate, acq_value = optimize_acqf_discrete(
            acq_function=qEI, 
            q=1,
            choices=sobol_acq_points
        )
        next_x = candidate.to(dtype=torch.float64).detach()

        # Evaluate and add the new point to the training set
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i] = 
        y_true_arr[i] =
        y_best_arr[i] =
        mu_arr[i] = 
        var_arr[i] = 
        alpha_arr[i] = 
        R_arr[i] = 

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr

def Experiment_PFN(rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):

    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Record initial data
    x_init = x_train
    y_init = y_train

    # Initial storage
    x_query_arr = torch.zeros()
    y_true_arr = torch.zeros()
    y_best_arr = torch.zeros()
    mu_arr = torch.zeros()
    var_arr = torch.zeros()
    alpha_arr = torch.zeros()
    R_arr = torch.zeros()

    # BO test
    for i in tqdm(range(N_iters)):
        # Setup surrogate model
        model = SingleTaskGP(x_train, y_train)

        # Tune Hyperparams to maximise MLL
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise Acquisition function (need to change to output mu and var)
        f_best = y_train.max()
        qEI = qLogExpectedImprovement(model, f_best)
        candidate, acq_value = optimize_acqf_discrete(
            acq_function=qEI, 
            q=1,
            choices=sobol_acq_points
        )
        next_x = candidate.to(dtype=torch.float64).detach()

        # Evaluate and add the new point to the training set
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i] = 
        y_true_arr[i] =
        y_best_arr[i] =
        mu_arr[i] = 
        var_arr[i] = 
        alpha_arr[i] = 
        R_arr[i] = 

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr

def Experiment_HEBO(rff_sampler: RFFSampler, x_train, N_iters, sobol_acq_points):
    
    # Compute values of sample initial points
    y_train = rff_sampler.sample(x_train)

    # Record initial data
    x_init = x_train
    y_init = y_train

    # Initial storage
    x_query_arr = torch.zeros()
    y_true_arr = torch.zeros()
    y_best_arr = torch.zeros()
    mu_arr = torch.zeros()
    var_arr = torch.zeros()
    alpha_arr = torch.zeros()
    R_arr = torch.zeros()

    # BO test
    for i in tqdm(range(N_iters)):
        # Setup surrogate model
        model = SingleTaskGP(x_train, y_train)

        # Tune Hyperparams to maximise MLL
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Maximise Acquisition function (need to change to output mu and var)
        f_best = y_train.max()
        qEI = qLogExpectedImprovement(model, f_best)
        candidate, acq_value = optimize_acqf_discrete(
            acq_function=qEI, 
            q=1,
            choices=sobol_acq_points
        )
        next_x = candidate.to(dtype=torch.float64).detach()

        # Evaluate and add the new point to the training set
        next_y = rff_sampler.sample(next_x)
        x_train = torch.cat([x_train, next_x])
        y_train = torch.cat([y_train, next_y])

        # Record data
        x_query_arr[i] = 
        y_true_arr[i] =
        y_best_arr[i] =
        mu_arr[i] = 
        var_arr[i] = 
        alpha_arr[i] = 
        R_arr[i] = 

    return x_query_arr, x_init, y_true_arr, y_init, y_best_arr, mu_arr, var_arr, alpha_arr, R_arr