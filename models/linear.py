import torch
import torch.nn as nn
from utils import sample_inverse_gamma
import math


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features, a_beta, b_beta):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        sigma_theta_squared = sample_inverse_gamma(a_beta, b_beta, size=1)
        self._initialize_beta(sigma_theta_squared)

    def forward(self, x):
        return self.linear(x)

    def _initialize_beta(self, sigma_theta_squared, device='cpu'):
        sigma_theta = math.sqrt(sigma_theta_squared)
        self.linear.weight.data = torch.normal(mean=0, std=sigma_theta, size=self.linear.weight.size(), device=device,
                                               requires_grad=True)
        if self.linear.bias is not None:
            self.linear.bias.data = torch.normal(mean=0, std=sigma_theta, size=self.linear.bias.size(), device=device,
                                                 requires_grad=True)

    def get_beta(self):
        return None

    def calculate_and_set_beta(self):
        return None

    def get_sigma_lambda_squared(self):
        return None

    def sample_sigma_lambda_squared(self):
        return None

    def calculate_and_set_nu(self):
        return None
