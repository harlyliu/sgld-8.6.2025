import torch
import torch.nn as nn
from STGP_input_layer import SpatialSTGPInputLayer
from utils import sample_inverse_gamma
from models.neural_network import NeuralNetwork

class STGPNeuralNetwork(nn.Module):
    def __init__(
            self,
            in_feature,
            grids,
            fully_connected_layers,
            poly_degree=22,
            a=0.01,
            b=1.0,
            dimensions=2,
            nu=0.5,
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

    def get_beta(self):
        return self.input_layer.beta

    def get_sigma_lambda_squared(self):
        return self.input_layer.sigma_lambda_squared
