import torch.nn as nn

from models.STGP_input_layer import SpatialSTGPInputLayer
from models.neural_network import NeuralNetwork


class STGPNeuralNetwork(nn.Module):
    def __init__(
            self,
            in_feature,
            grids,
            fully_connected_layers,
            dimensions=2,
            a_theta=2.0,
            b_theta=1.0,
            device='cpu'
    ):
        """
        Combines a_for_eigen fixed STGP input transform with a_for_eigen standard FC network.
        """
        super().__init__()
        self.device = device
        self.input_layer = SpatialSTGPInputLayer(
            in_feature=in_feature,
            num_of_units_in_top_layer_of_fully_connected_layers=fully_connected_layers[0],
            grids=grids,
            dimensions=dimensions,
            a_theta=a_theta,
            b_theta=b_theta
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

    def sample_sigma_lambda_squared(self):
        return self.input_layer.sample_and_set_sigma_lambda_squared()

    def calculate_and_set_nu(self):
        return self.input_layer.update_and_set_nu()

