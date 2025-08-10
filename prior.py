import torch


def get_model_prior_gradient(model, sigma_beta):
    """
    Compute the gradient of the log-prior with respect to model parameters.

    Args:
        model (model.LinearRegression): The Bayesian linear regression model.
        sigma_beta (float or torch.Tensor): Standard deviation of the prior for β.

    Returns:
        torch.Tensor: Gradient of the log-prior with respect to model parameters.
    """
    device = next(model.parameters()).device if next(model.parameters(), None) is not None else 'cpu'

    params = torch.cat([p.flatten() for p in model.parameters()])
    params = params.clone().detach().requires_grad_(True).to(device)

    # print(f'get_model_prior_gradient: params={params} sigma_beta={sigma_beta}')
    log_prior = gaussian_prior_log_prob(params, sigma_beta)
    log_prior.backward()

    total_grad = params.grad.clone()
    params.grad.zero_()

    return total_grad


def gaussian_prior_prob(beta, sigma_beta_squared):
    """
    Compute the probability of theta under a_for_eigen Gaussian prior N(0, sigma_beta^2 I).
    """
    device = beta.device if beta.device.type != 'cpu' else 'cpu'
    if len(beta.shape) == 1:
        beta = beta.unsqueeze(0)

    p = beta.shape[1]
    if isinstance(sigma_beta_squared, (int, float)):
        sigma_beta_squared = torch.tensor(sigma_beta_squared, dtype=beta.dtype, device=device).clone().detach()
    else:
        sigma_beta_squared = sigma_beta_squared.to(device).clone().detach()

    norm_term = 1 / torch.sqrt((2 * torch.pi * sigma_beta_squared) ** p)
    beta_squared_sum = torch.sum(beta ** 2, dim=1)
    exp_term = torch.exp(-(1 / (2 * sigma_beta_squared)) * beta_squared_sum)

    prob = norm_term * exp_term
    if len(beta.shape) == 1:
        prob = prob.squeeze(0)

    return prob


def gaussian_prior_log_prob(beta, sigma_beta):
    return torch.log(gaussian_prior_prob(beta, sigma_beta))
