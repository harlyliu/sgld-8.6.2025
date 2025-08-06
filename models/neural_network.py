import torch.nn as nn
from utils import sample_inverse_gamma
import math


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=3, hidden_unit_list=(20, 30, 1), a_theta=1.0, b_theta=2.0):
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

    def get_beta(self):
        return None

    def get_sigma_lambda_squared(self):
        return None
