import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SGLD_v7 import sample_inverse_gamma
from GP_comp.GP import gp_eigen_value, gp_eigen_funcs_fast


class SpatialSTGPInputLayer(nn.Module):
    def __init__(self, in_feature, num_of_units_in_top_layer_of_fully_connected_layers, grids, poly_degree=10, a=0.01, b=1.0, dimensions=2,
                 nu=0.5, nu_tilde=5, a_theta=2.0, b_theta=1.0, a_lambda=2.0, b_lambda=1.0, device='cpu'):
        """
        :param num_of_units_in_top_layer_of_fully_connected_layers: number of neurons in this layer of the neural network
        :param grids: tensor that serves as a skeleton for the image, tensor of coordinates
        :param poly_degree: the degree to which the eigen functions and eigen values must be calculated
        :param a: left bound of spatial domain(used for eigen)
        :param b: right bound of spatial domain(used for eigen)
        :param dimensions: amount of dimensions in GP(used for eigen)
        :param nu: soft threshold(determines sparsity)
        :param device: the device the neural network is on
        """

        super().__init__()
        self.in_feature = in_feature
        self.device = device
        self.num_of_units_in_top_layer_of_fully_connected_layers = num_of_units_in_top_layer_of_fully_connected_layers

        self.a_lambda = a_lambda
        self.b_lambda = b_lambda
        self.sigma_lambda_squared = sample_inverse_gamma(self.a_lambda, self.b_lambda, size=1)

        # Initialize grids and other parameters
        self.grids = torch.tensor(grids.copy(), dtype=torch.float32).to(device)
        self.poly_degree = poly_degree
        self.a = a
        self.b = b

        self.dimensions = dimensions

        # note: nu is the thing that looks like v but isn't. nu is inversely related to variance(sigma_lambda squared
        # the larger the variance, the lower the threshold. when variance is small, higher threshold, greater sparsity
        # used to normalize thresholding relative to variance. + 1e-8 prevents division by 0 error.
        # This prevents division by zero or numerical instability if sigma_lambda_squared is very small.
        # function in the line above equation 33. nu~ =v/sigma_lambda
        if dimensions == 1:
            self.nu_tilde = torch.abs(nu / (torch.sqrt(self.sigma_lambda_squared) + 1e-8))
        elif dimensions == 2:
            self.nu_tilde = nu_tilde

        eigenvalues_np = gp_eigen_value(poly_degree, a, b, dimensions)
        self.K = len(eigenvalues_np)
        self.eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)  # shape (K,)

        if isinstance(grids, torch.Tensor):
            grids_np = grids.cpu().numpy()
        else:
            grids_np = grids.copy()

        eigenfuncs_np = gp_eigen_funcs_fast(grids_np, poly_degree, a, b, orth=True)
        self.eigenfuncs = torch.tensor(eigenfuncs_np, dtype=torch.float32, device=device)  # shape (K, V)

        # self.eigenfuncs = self.eigenfuncs.T  # now (V, K)

        self.Cu = self.sample_Cu()  # Cu in equation 34
        beta = torch.matmul(self.Cu, self.eigenfuncs.T).detach()
        self.beta = nn.Parameter(beta)
        # print(beta.shape)
        # ksi is the bias term for the input layer in equation 31. ksi is the pronunciation of the greek letter
        # self.ksi is a vector, the size is num_of_units_in_top_layer_of_fully_connected_layers
        self.ksi = nn.Parameter(torch.zeros(num_of_units_in_top_layer_of_fully_connected_layers, device=device))
        self.initializeKsi(a_theta, b_theta)

    def initializeKsi(self, a_theta=2.0, b_theta=1.0):
        """
        Initialize ksi vector using an inverse-gamma prior, matching FC layer bias init.
        Draws sigma_theta_squared ~ InvGamma(a_theta, b_theta), then sets
        ksi.data ~ Normal(0, sigma_theta).
        """
        # --- Sample sigma_theta_squared ---
        sigma_theta_squared = sample_inverse_gamma(a_theta, b_theta)
        sigma_theta = math.sqrt(sigma_theta_squared)

        # --- Initialize ksi values ---
        self.ksi.data = torch.normal(
            mean=0.0,
            std=sigma_theta,
            size=self.ksi.size(),
            device=self.device
        )

    def sample_sigma_lambda_squared(self):
        with torch.no_grad():
            total_entries = self.Cu.numel()
            squared_norm = torch.sum(self.Cu ** 2)
            new_a_lambda = self.a_lambda + total_entries / 2
            new_b_lambda = self.b_lambda + squared_norm / 2
            self.sigma_lambda_squared = sample_inverse_gamma(new_a_lambda, new_b_lambda, size=1)
            return self.sigma_lambda_squared

    def sample_Cu(self):
        """
        Resample Cu ∼ N(0, σ²_lambda ⋅ Λ) after sigma_lambda_squared is updated.
        """
        std_dev = torch.sqrt(self.eigenvalues)  # eq 34, std_dev.shape = (self.K, )
        # std_dev = torch.sqrt(self.sigma_lambda_squared * self.eigenvalues) # eq 33
        Cu = torch.randn(self.num_of_units_in_top_layer_of_fully_connected_layers, self.K, device=self.device) * std_dev  # Cu in equation 34
        self.beta = torch.matmul(Cu, self.eigenfuncs.T)
        return Cu

    def soft_threshold(self, x):
        magnitude = torch.abs(x) - self.nu_tilde
        # 4) zero out negatives
        thresholded = torch.relu(magnitude)
        # 5) restore sign
        return thresholded * torch.sign(x)

    def forward(self, X):
        """

        :param X: represents the image, intensity of each voxel
        :return: the input for the fully connected hidden layers
        """
        # function 34
        # with torch.no_grad():
        #    self.beta.data.copy_( self.soft_threshold(self.beta) )
        beta = math.sqrt(self.sigma_lambda_squared) * self.soft_threshold(self.beta) # eq 36
        # beta = self.soft_threshold(self.beta) # eq 35
        # z = torch.matmul(X, self.beta.T) + self.ksi  # (B, num_of_units_in_top_layer_of_fully_connected_layers)
        z = torch.matmul(X, beta.T) + self.ksi # If you want linear to work for sure, use this line
        activated = F.relu(z)
        return activated
