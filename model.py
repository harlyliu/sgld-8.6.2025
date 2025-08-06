import torch
import torch.nn as nn
from STGP_input_layer import SpatialSTGPInputLayer
from SGLD_v7 import sample_inverse_gamma
import math


class LinearRegression(nn.Module):
    def __init__(self, in_features, out_features, a_beta, b_beta):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        sigma_beta_squared = sample_inverse_gamma(a_beta, b_beta, size=1)
        self._initialize_beta(sigma_beta_squared)

    def forward(self, x):
        return self.linear(x)

    def _initialize_beta(self, sigma_beta_squared, device='cpu'):
        sigma_beta = math.sqrt(sigma_beta_squared)
        self.linear.weight.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.weight.size(), device=device,
                                               requires_grad=True)
        if self.linear.bias is not None:
            self.linear.bias.data = torch.normal(mean=0, std=sigma_beta, size=self.linear.bias.size(), device=device,
                                                 requires_grad=True)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_unit_list=(20, 30, 1), a_theta=2.0, b_theta=1.0):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_unit_list = hidden_unit_list

        layers = []
        current_input_size = input_size

        for i, units in enumerate(hidden_unit_list):
            linear_layer = nn.Linear(current_input_size, units)
            layers.append(linear_layer)

            if i < len(hidden_unit_list) - 1:
                layers.append(nn.ReLU())

            current_input_size = units

        self.sequential = nn.Sequential(*layers)

        sigma_theta_squared = sample_inverse_gamma(a_theta, b_theta, size=1)
        self._initialize(sigma_theta_squared)

    def _initialize(self, sigma_theta_squared):
        sigma_theta = math.sqrt(sigma_theta_squared)
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0, std=sigma_theta)
                if m.bias is not None:
                    nn.init.normal_(m.bias.data, mean=0, std=sigma_theta)

    def forward(self, x):
        return self.sequential(x)


class STGPNeuralNetwork(nn.Module):
    def __init__(
            self,
            in_feature,
            grids,
            fully_connected_layers,
            poly_degree=10,
            a=0.01,
            b=1.0,
            dimensions=2,
            nu=0.1,
            nu_tilde=5,
            a_theta=2.0,
            b_theta=1.0,
            a_lambda=2.0,
            b_lambda=1.0,
            device='cpu'
    ):
        """
        Combines a fixed STGP input transform with a standard FC network.
        """
        super().__init__()
        self.device = device
        self.input_layer = SpatialSTGPInputLayer(
            in_feature=in_feature,
            num_of_units_in_top_layer_of_fully_connected_layers=fully_connected_layers[0],
            grids=grids,
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
        # Build the rest using NeuralNetwork
        self.fc = NeuralNetwork(
            input_size=fully_connected_layers[0],
            hidden_unit_list=tuple(fully_connected_layers[1:]),
            a_theta=a_theta,
            b_theta=b_theta
        ).to(device)

    def forward(self, X):
        z = self.input_layer(X)  # applies fixed theta & ksi, plus ReLU
        return self.fc(z)
