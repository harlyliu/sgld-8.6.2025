import torch
from torch.distributions import Gamma


def sample_inverse_gamma(shape_param, scale_param, size=1):
    """
    Sample from an Inverse-Gamma distribution.

    Args:
        shape_param (float or torch.Tensor): Shape parameter α (must be positive).
        scale_param (float or torch.Tensor): Scale parameter β (must be positive).
        size (int or tuple): Number of samples or shape of the output tensor.
    Returns:
        torch.Tensor: Samples from the Inverse-Gamma distribution, shape determined by size.
    """
    gamma_dist = Gamma(shape_param, scale_param)
    gamma_samples = gamma_dist.sample((size,))
    inverse_gamma_samples = 1.0 / gamma_samples
    return inverse_gamma_samples


def sample_sigma_squared(X, y, model, a=2.0, b=50.0, device='cpu'):
    """
    Sample σ^2 from an Inverse-Gamma distribution using equation (20).

    Args:
        X (torch.Tensor): Design matrix of shape (n, p) containing predictors.
        y (torch.Tensor): Target vector of shape (n,) or (n, 1).
        model (model.LinearRegression): The Bayesian linear regression model.
        a (float): Shape parameter a for σ^2 (default=2.0).
        b (float): Scale parameter b for σ^2 (default=50.0).
        device (str): Device to place the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Sampled σ^2 from the Inverse-Gamma distribution.
    """
    X = X.to(device)
    if y.dim() == 1:
        y = y.unsqueeze(1).to(device)
    else:
        y = y.to(device)

    model = model.to(device)

    # Number of observations (n)
    n = y.shape[0]

    # Compute predictions (y_pred) using the model
    y_pred = model(X)
    residuals = y - y_pred  # Shape: (n, 1)
    residual_squared_sum = torch.sum(residuals**2) / 2  # (y - Xβ)^T (y - Xβ) / 2

    # New shape and scale parameters for σ^2 (equation 20)
    new_shape_sigma = torch.tensor(a, dtype=torch.float32, device=device).clone().detach() + n / 2
    new_scale_sigma = torch.tensor(b, dtype=torch.float32, device=device).clone().detach() + residual_squared_sum

    # Sample σ^2 from Inverse-Gamma(new_shape_sigma, new_scale_sigma)
    sigma_squared = sample_inverse_gamma(new_shape_sigma, new_scale_sigma, size=1).squeeze()
    # print(f"Batch residual sum of squares / 2: {residual_squared_sum.item()}")
    # print(f"shape: {new_shape_sigma.item()}, scale: {new_scale_sigma.item()} a={a} n={n}")
    # print(f"Sampled σ^2: {sigma_squared.item()}")

    return sigma_squared


def sample_sigma_beta_squared(model, a_beta=2.0, b_beta=8.0, device='cpu'):
    """
    Sample σβ^2 from an Inverse-Gamma distribution using equation (21), with data dependency via β^T β.

    Args:
        model (model.LinearRegression): The Bayesian linear regression model.
        a_beta (float): Shape parameter aβ for σβ^2 (default=2.0).
        b_beta (float): Scale parameter bβ for σβ^2 (default=8.0).
        device (str): Device to place the tensor on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Sampled σβ^2 from the Inverse-Gamma distribution.
    """
    model = model.to(device)

    # Number of features (p) including bias
    p = sum(p.numel() for p in model.parameters())  # Total number of parameters (weights + bias)

    # Compute β^T β / 2 from current model parameters
    beta = torch.cat([p.flatten() for p in model.parameters()]).to(device)  # Flatten all parameters into a vector
    beta_squared_sum = torch.sum(beta**2) / 2  # β^T β / 2
    # print(f"β^T β / 2: {beta_squared_sum.item()}")

    # New shape and scale parameters for σβ^2 (equation 21)
    new_shape_beta = torch.tensor(a_beta, dtype=torch.float32, device=device).clone().detach() + p / 2
    new_scale_beta = torch.tensor(b_beta, dtype=torch.float32, device=device).clone().detach() + beta_squared_sum

    # Sample σβ^2 from Inverse-Gamma(new_shape_beta, new_scale_beta)
    sigma_beta_squared = sample_inverse_gamma(new_shape_beta, new_scale_beta, size=1).squeeze()

    # Optional: Print diagnostics for debugging
    # print(f"β^T β / 2: {beta_squared_sum.item()}")
    # print(f"Shape: {new_shape_beta.item()}, Scale: {new_scale_beta.item()}")
    # print(f"Sampled σβ^2: {sigma_theta_squared.item()}")

    return sigma_beta_squared


def sample_variances(X, y, model, a=2.0, b=50.0, a_beta=2.0, b_beta=8.0, device='cpu'):
    """
    Sample σ^2 and σβ^2 from Inverse-Gamma distributions using equations (20) and (21) for updates during SGLD.

    Args:
        X (torch.Tensor): Design matrix of shape (n, p) containing predictors.
        y (torch.Tensor): Target vector of shape (n,) or (n, 1).
        model (model.LinearRegression): The Bayesian linear regression model.
        a (float): Shape parameter a for σ^2 (default=2.0).
        b (float): Scale parameter b for σ^2 (default=50.0).
        a_beta (float): Shape parameter aβ for σβ^2 (default=2.0).
        b_beta (float): Scale parameter bβ for σβ^2 (default=8.0).
        device (str): Device to place the tensor on ('cpu' or 'cuda').

    Returns:
        tuple: (sigma_squared, sigma_theta_squared) – Samples from Inverse-Gamma distributions.
    """
    sigma_squared = sample_sigma_squared(X, y, model, a, b, device)
    sigma_beta_squared = sample_sigma_beta_squared(model, a_beta, b_beta, device)
    return sigma_squared, sigma_beta_squared
