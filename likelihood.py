import torch


def get_model_likelihood_gradient(model, X, y, sigma):
    """
    Compute the gradient of the log-likelihood with respect to model parameters.

    Args:
        model (model.LinearRegression): The Bayesian linear regression model.
        X (torch.Tensor): Design matrix of shape (batch_size, p).
        y (torch.Tensor): Target vector of shape (batch_size,) or (batch_size, 1).
        sigma (float or torch.Tensor): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: Gradient of the log-likelihood with respect to model parameters.
    """
    if y.dim() == 1:
        y = y.unsqueeze(1)

    device = X.device if X.device.type != 'cpu' else 'cpu'
    X = X.to(device)
    y = y.to(device)
    model = model.to(device)
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, dtype=X.dtype, device=device).clone().detach()
    else:
        sigma = sigma.to(device).clone().detach()

    for param in model.parameters():
        param.requires_grad_(True)

    log_likelihood = gaussian_likelihood_log_prob(y, X, model, sigma)
    log_likelihood.backward()

    total_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    for param in model.parameters():
        param.grad.zero_()

    return total_grad


def gaussian_likelihood_log_prob(y, X, model, sigma, device='cpu'):
    """
    Compute the log-likelihood of y under a Gaussian likelihood model p(y; model, σ^2, X).

    Args:
        y (torch.Tensor): Target vector of shape (n,) or (n, 1).
        X (torch.Tensor): Design matrix of shape (n, p) containing predictors.
        model (model.LinearRegression): The Bayesian linear regression model.
        sigma (float or torch.Tensor): Standard deviation of the Gaussian noise (default=1.0).
        device:
    Returns:
        torch.Tensor: Log-likelihood of y given the model, σ^2, and X.
    """
    if y.dim() == 1:
        y = y.unsqueeze(1)

    X = X.to(device)
    model = model.to(device)
    if isinstance(sigma, (int, float)):
        sigma = torch.tensor(sigma, dtype=X.dtype, device=device).clone().detach()
    else:
        sigma = sigma.to(device).clone().detach()
    y = y.to(device)

    y_pred = model(X)  # Shape: (n, 1)
    residuals = y - y_pred  # Shape: (n, 1)

    n = y.shape[0]
    residual_squared_sum = torch.sum(residuals**2)

    # The following implementation has to use torch.log because the value is extremely small when n is big
    # and the calculation without log is very unstable
    log_norm = -(n / 2) * torch.log(2 * torch.pi * sigma**2)
    quadratic_term = -(1 / (2 * sigma**2)) * residual_squared_sum
    log_likelihood = log_norm + quadratic_term
    # print(f'log_likelihood={log_likelihood} log_norm={log_norm} quadratic_term={quadratic_term}')
    return log_likelihood

