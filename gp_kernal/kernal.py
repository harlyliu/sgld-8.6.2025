import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def test():
    # Data
    X = np.array([1, 3, 5, 7]).reshape(-1, 1)
    y = np.array([20, 22, 21, 19])

    # Define RBF kernel: ConstantKernel * RBF allows variance scaling
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=0.1, n_restarts_optimizer=10)

    # Fit the model
    gp.fit(X, y)

    # Predict at x* = 4
    x_star = np.array([[4]])
    mu_star, sigma_star = gp.predict(x_star, return_std=True)

    print(f"Predicted temperature: {mu_star[0]:.2f} ± {sigma_star[0]:.2f} °C")


if __name__ == '__main__':
    test()
