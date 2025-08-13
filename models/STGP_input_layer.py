import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SGLD_v7 import sample_inverse_gamma
from GP_comp.GP import gp_eigen_value, gp_eigen_funcs_fast


class SpatialSTGPInputLayer(nn.Module):
    def __init__(self,
                 in_feature,
                 num_of_units_in_top_layer_of_fully_connected_layers,
                 grids,
                 poly_degree_for_eigen=10,
                 a_for_eigen=1.0,
                 b_for_eigen=2.0,
                 dimensions=2,
                 nu_tilde=0.1,
                 a_theta=2.0,
                 b_theta=1.0,
                 a_lambda=2.0,
                 b_lambda=1.0,
                 device='cpu'):
        """
        :param num_of_units_in_top_layer_of_fully_connected_layers: number of neurons in this layer of the neural network
        :param grids: tensor that serves as a_for_eigen skeleton for the image, tensor of coordinates
        :param poly_degree_for_eigen: the degree to which the eigen functions and eigen values must be calculated
        :param a_for_eigen: left bound of spatial domain(used for eigen)
        :param b_for_eigen: right bound of spatial domain(used for eigen)
        :param dimensions: amount of dimensions in GP(used for eigen)
        :param device: the device the neural network is on
        """

        super().__init__()
        self.in_feature = in_feature
        self.device = device
        self.num_of_units_in_top_layer_of_fully_connected_layers = num_of_units_in_top_layer_of_fully_connected_layers

        self.a_lambda = a_lambda
        self.b_lambda = b_lambda

        # Initialize grids and values for eigen
        self.grids = torch.tensor(grids.copy(), dtype=torch.float32).to(device)
        self.poly_degree_for_eigen = poly_degree_for_eigen
        self.a_for_eigen = a_for_eigen
        self.b_for_eigen = b_for_eigen



        eigenvalues_np = gp_eigen_value(poly_degree_for_eigen, a_for_eigen, b_for_eigen, dimensions)
        self.K = len(eigenvalues_np)
        self.eigenvalues = torch.tensor(eigenvalues_np, dtype=torch.float32, device=device)  # shape (K,)

        if isinstance(grids, torch.Tensor):
            grids_np = grids.cpu().numpy()
        else:
            grids_np = grids.copy()

        eigenfuncs_np = gp_eigen_funcs_fast(grids_np, poly_degree_for_eigen, a_for_eigen, b_for_eigen, orth=True)
        self.eigenfuncs = torch.tensor(eigenfuncs_np, dtype=torch.float32, device=device)  # shape (K, V)

        # Equation 36
        Cu = self._sample_Cu_for_prior()  # Cu in equation 36
        self.Cu = nn.Parameter(Cu)
        self.TOTAL_ENTRIES_OF_CU = self.Cu.numel()

        self.sigma_lambda_squared = None
        self.sample_and_set_sigma_lambda_squared()

        # note: nu is the thing that looks like v but isn't. nu is inversely related to variance(sigma_lambda squared
        # the larger the variance, the lower the threshold. when variance is small, higher threshold, greater sparsity
        # used to normalize thresholding relative to variance. + 1e-8 prevents division by 0 error.
        # This prevents division by zero or numerical instability if sigma_lambda_squared is very small.
        # function in the line above equation 33. nu~ =v/sigma_lambda
        self.NU_TILDE = nu_tilde
        self.nu = None
        self.update_and_set_nu()

        self.beta = None
        self.calculate_and_set_beta()

        # print(beta.shape)
        # ksi is the bias term for the input layer in equation 31. ksi is the pronunciation of the greek letter
        # self.ksi is a_for_eigen vector, the size is num_of_units_in_top_layer_of_fully_connected_layers
        self.ksi = nn.Parameter(torch.zeros(num_of_units_in_top_layer_of_fully_connected_layers, device=device))
        self._initialize_ksi(a_theta, b_theta)

    def calculate_and_set_beta(self):
        beta_before_threshold = torch.matmul(self.Cu, self.eigenfuncs.T)
        self.beta = math.sqrt(self.sigma_lambda_squared) * self.soft_threshold(beta_before_threshold)
        return self.beta

    def _initialize_ksi(self, a_theta=2.0, b_theta=1.0):
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

    def sample_and_set_sigma_lambda_squared(self):
        with torch.no_grad():
            new_a_lambda = self.a_lambda + self.TOTAL_ENTRIES_OF_CU / 2.0
            new_b_lambda = self.b_lambda + torch.sum(self.Cu ** 2) / 2.0
            self.sigma_lambda_squared = sample_inverse_gamma(new_a_lambda, new_b_lambda, size=1)
            return self.sigma_lambda_squared

    def update_and_set_nu(self):
        with torch.no_grad():
            self.nu = self.NU_TILDE * torch.sqrt(self.sigma_lambda_squared)
            return self.nu

    def _sample_Cu_for_prior(self):
        """
        Resample Cu ∼ N(0, Λ) after sigma_lambda_squared is updated.
        """
        std_dev = torch.sqrt(self.eigenvalues)  # eq 36, std_dev.shape = (self.K, )
        Cu = torch.randn(self.num_of_units_in_top_layer_of_fully_connected_layers, self.K, device=self.device) * std_dev  # Cu in equation 36
        return Cu

    def soft_threshold(self, x):
        magnitude = torch.abs(x) - self.nu
        thresholded = torch.relu(magnitude)
        return thresholded * torch.sign(x)

    def forward(self, X):
        """

        :param X: represents the image, intensity of each voxel
        :return: the input for the fully connected hidden layers
        """
        self.calculate_and_set_beta()
        z = torch.matmul(X, self.beta.T) + self.ksi
        activated = F.relu(z)
        return activated
