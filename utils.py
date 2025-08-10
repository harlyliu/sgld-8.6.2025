import torch
from torch.distributions import Gamma
import numpy as np
import matplotlib.pyplot as plt


def plot_mse(trainer, start=0, end=-1):
    plot_values(trainer.samples['mse'][start:end], 'Mean Squared Error', 'Mean Squared Error over Samples/Iterations',
                'Sample/Iteration Number', 'Mean Squared Error (MSE)')


def plot_sigma_squared(trainer, start=0, end=-1):
    plot_values(trainer.samples['sigma_squared'][start:end], 'sigma squared', 'Sigma squared over Samples/Iterations',
                'Sample/Iteration Number', 'Sigma squared')


def plot_sigma_theta_squared(trainer, start=0, end=-1):
    plot_values(trainer.samples['sigma_theta_squared'][start:end], 'sigma theta squared',
                'Sigma theta squared over Samples/Iterations', 'Sample/Iteration Number', 'Sigma theta squared')


def plot_values(vals, label, title, xlabel, ylabel):
    plt.plot(vals, color='blue', linestyle='-', marker='o', markersize=4, label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_image(img, label, title):
    plt.figure()
    plt.imshow(img, interpolation='nearest')
    plt.colorbar(label=label)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def generate_linear_data(n=1000, in_features=3, noise_std=1.0):
    """
    Generate synthetic data for multivariate linear regression: y = X @ w + b_for_eigen + noise.

    Args:
        n (int): Number of samples.
        in_features (int): Number of input features.
        noise_std (float): Standard deviation of Gaussian noise.

    Returns:
        tuple: (X, y) – Design matrix and target vector.
    """
    # True parameters
    # true_weights = torch.FloatTensor(in_features).uniform_(-2, 2)  # Random weights for each feature
    weight = [x + 1.0 for x in range(in_features)]
    true_weights = torch.tensor(weight)
    true_bias = torch.tensor(-1.0)  # Single bias term
    # print(f"True weights: {true_weights}")  # Example: tensor([1.234, -0.567, 0.890])
    # print(f"True bias: {true_bias}")      # tensor(-1.0)

    # Generate X (uniformly distributed between -5 and 5)
    X = torch.FloatTensor(n, in_features).uniform_(-5, 5)

    # Generate y = X @ w + b_for_eigen + noise
    # X @ w performs matrix multiplication between (n, in_features) and (in_features,) -> (n,)
    # We add the bias and noise afterward
    noise = torch.normal(mean=0, std=noise_std, size=(n,))
    y = X @ true_weights + true_bias + noise  # Shape: (n,)
    y = y.unsqueeze(1)  # torch.Size([1000]) -> torch.Size([1000, 1])
    return X, y, true_weights, true_bias


def sample_inverse_gamma(shape_param, rate_param, size=1):
    """
    Sample from an Inverse-Gamma distribution.

    Args:
        shape_param (float or torch.Tensor): Shape parameter α (must be positive).
        rate_param (float or torch.Tensor): Scale parameter β (must be positive).
        size (int): Number of samples or shape of the output tensor.
    Returns:
        torch.Tensor: Samples from the Inverse-Gamma distribution, shape determined by size.
    """
    gamma_dist = Gamma(shape_param, rate_param)
    gamma_samples = gamma_dist.sample(torch.Size((size,)))
    inverse_gamma_samples = 1.0 / gamma_samples

    return inverse_gamma_samples


def extract_beta(samples, model, start, end):
    beta_samples = []
    param_shape = model.input_layer.beta.shape
    for flat_params in samples['params'][start, end]:
        # find and reconstruct only the beta parameter
        idx = 0
        for p in model.parameters():
            numel = p.numel()
            if p is model.input_layer.beta:
                chunk = flat_params[idx:idx + numel]
                beta = torch.from_numpy(chunk.reshape(param_shape)).to('cpu')
                # apply the same soft‐threshold you use in train
                thresholded_beta = model.input_layer.soft_threshold(
                    beta)  # thresholded_beta.shape() = (amount of units, amount of voxels)
                beta_samples.append(thresholded_beta.detach().cpu().numpy())
                break
            idx += numel
    return beta_samples


def plot_sigma_trace(sigma_samples, true_sigma2=None):
    plt.figure(figsize=(10, 4))
    plt.plot(sigma_samples, label="Sampled σ²")
    if true_sigma2 is not None:
        plt.axhline(y=true_sigma2 ** 2, color='red', linestyle='--', label=f'True σ² = {true_sigma2 ** 2:.4f}')
    plt.xlabel("Sample Index")
    # plt.ylim(0, 100000)
    plt.ylabel("σ²")
    plt.title("Trace Plot of σ²")
    plt.legend()
    plt.tight_layout()
    plt.show()


# Example usage:
if __name__ == "__main__":
    # Generate data with 3 features
    """
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=1, noise_std=0.5)
    print(f"X shape: {X.shape}")  # torch.Size([1000, 3])
    print(f"y shape: {y.shape}")  # torch.Size([1000])
    print(true_weights[0] + true_bias)
    """
