import random

import numpy as np
import torch

from GP_comp.GP import generate_grids
from SGLD_v7 import SgldBayesianRegression as V7, select_significant_voxels
from model import STGPNeuralNetwork
from simulate_single_modality import simulate_data
from utils import generate_linear_data

# Set random seed for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def test_stgp_linear(in_feature):
    # Step 1: Define parameters
    fully_connected_layers = [10, 50, 1]
    poly_degree = 22
    a = 0.01
    b = 1.0
    dimensions = 1
    nu_tilde = None
    nu = 0.5

    a_theta = 2.0
    b_theta = 1.0
    a_lambda = 2.0
    b_lambda = 1.0
    device = 'cpu'
    # Step 2: Generate synthetic data
    X, y, true_weights, true_bias = generate_linear_data(n=1000, in_features=in_feature, noise_std=1.0)
    print(f"Shape of X: {X.shape}")

    # Step 3: Generate spatial grid
    grids = generate_grids(dimensions=dimensions, num_grids=in_feature, grids_lim=(-1, 1))
    inputs = torch.ones(in_feature)
    expected_y = sum(true_weights) + true_bias

    # Step 4: Build model
    model = STGPNeuralNetwork(
        in_feature=in_feature,
        grids=grids,
        fully_connected_layers=fully_connected_layers,
        poly_degree=poly_degree,
        a=a,
        b=b,
        dimensions=dimensions,
        nu=nu,
        nu_tilde=nu_tilde,
        a_theta=a_theta,
        b_theta=b_theta,
        a_lambda=a_lambda,
        b_lambda=b_lambda,
        device=device
    )

    # Step 5: Train with SGLD
    trainer = V7(
        a=a,
        b=b,
        a_theta=a_theta,
        b_theta=b_theta,
        step_size=0.0005,
        num_epochs=300,  # fix this back to 300 later
        burn_in_epochs=100,
        batch_size=100,
        device=device,
        model=model
    )
    trainer.train(X, y)

    print(f"true_weight={true_weights} true_bias={true_bias}")
    print(f"X={inputs} Y(predicted)={trainer.predict(inputs, gamma=None)} Y(expected)={expected_y}")


def test_significant_voxels():
    beta_samples = []
    for i in range(10):
        sample = np.zeros((4, 5))
        if i >= 3:
            sample[:, 1] = 1.0
            sample[:, 3] = 1.0
            sample[0, 1] = 0.0
            sample[0, 0] = 1.0
        if i >= 8:
            sample[0, 0] = 0.0
        beta_samples.append(sample)

    mask, p_hat, delta, r = select_significant_voxels(beta_samples, gamma=0.50)
    print("p_hat:", p_hat)
    print("delta:", delta)
    print("r:", r)
    print("mask:", mask)
    assert(mask.tolist() == [True, True, False, True, False])


def test_stgp_2D(in_feature):
    # Step 1: Define parameters
    fully_connected_layers = [10, 50, 1]
    poly_degree = 22
    a = 0.01
    b = 1.0
    dimensions = 2
    nu_tilde = 5 # recommended value is 5
    nu = None

    a_theta = 2.0
    b_theta = 1.0
    a_lambda = 2.0
    b_lambda = 1.0
    device = 'cpu'
    gamma = 0.05
    # Step 2: Generate synthetic data
    # simulate a batch of images
    dim = in_feature           # e.g. 16 for a 16×16 grid → V=256
    r2 = 0.8                  # signal-to-noise ratio
    n = 1000                  # number of images/samples
    v_list, true_beta, img, Y, true_sigma = simulate_data(
        n=n, r2=r2, dim=dim, random_seed=42)
    # torch tensors
    X = torch.from_numpy(img).float().to(device)    # shape [n, V]
    y = torch.from_numpy(Y).float().to(device)      # shape [n]
    grids = v_list                                  # shape [V, 2]
    # pick one of your simulated images
    idx = 0
    inputs = X[idx : idx+1]       # shape [1, V]
    expected_y = Y[idx]
    print(f"Simulated {n} images of {dim}×{dim} = {img.shape[1]} pixels")

    # Step 4: Build model
    model = STGPNeuralNetwork(
        in_feature=in_feature,
        grids=grids,
        fully_connected_layers=fully_connected_layers,
        poly_degree=poly_degree,
        a=a,
        b=b,
        dimensions=dimensions,
        nu=nu,
        nu_tilde=nu_tilde,
        a_theta=a_theta,
        b_theta=b_theta,
        a_lambda=a_lambda,
        b_lambda=b_lambda,
        device=device
    )

    # Step 5: Train with SGLD
    trainer = V7(
        a=a,
        b=b,
        a_theta=a_theta,
        b_theta=b_theta,
        step_size=0.00005,
        num_epochs=300,  # fix this back to 300 later
        burn_in_epochs=100,
        batch_size=100,
        device=device,
        model=model
    )
    indices = torch.randperm(X.size(0))
    X, y = X[indices], y[indices]
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train with MSE tracking every 10 epochs
    trainer.train(X_train, y_train, X_test, y_test, mse_eval_interval=10, true_sigma2=true_sigma)

    print(f"Y(predicted)={trainer.predict(inputs, gamma=gamma)} Y(expected)={expected_y}")


# Generate data and run SGLD
if __name__ == "__main__":
    # test_stgp_linear(in_feature=6)
    test_stgp_2D(5)
    # test_significant_voxels()
