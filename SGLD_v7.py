import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import time
from utils import sample_inverse_gamma
from sgld_utils import select_significant_voxels, implement_mask


class SgldBayesianRegression:
    """
    Stochastic Gradient Langevin Dynamics (SGLD) for Bayesian Linear Regression.
    """

    def __init__(
            self,
            model,
            step_size: float,
            num_epochs: int,
            burn_in_epochs: int,
            batch_size: int,  # number of samples in a_for_eigen single batch
            a: float = 1.0,  # shape of inverse gamma prior of sigma^2
            b: float = 2.0,  # rate of inverse gamma prior of sigma^2
            a_theta: float = 2.0,  # shape of inverse gamma prior of sigma^2_beta
            b_theta: float = 2.0,  # rate of inverse gamma prior of sigma^2_beta
            device: str = 'cpu'
    ):
        """
        Initializes the SgldBayesianRegression model.

        Args:
            a: Shape parameter for the inverse gamma prior of sigma^2.
            b: Rate parameter for the inverse gamma prior of sigma^2.
            a_theta: Shape parameter for the inverse gamma prior of sigma^2_theta.
            b_theta: Rate parameter for the inverse gamma prior of sigma^2_theta.
            step_size: Step size for the SGLD algorithm.
            num_epochs: Total number of epochs.
            burn_in_epochs: Number of burn-in epochs.
            batch_size: Number of samples in each batch.
            device: Device to use for computation (e.g., 'cpu', 'cuda').
        """
        self.device = device
        self.a = torch.tensor(a, dtype=torch.float32, device=self.device)
        self.b = torch.tensor(b, dtype=torch.float32, device=self.device)
        self.a_theta = torch.tensor(a_theta, dtype=torch.float32, device=self.device)
        self.b_theta = torch.tensor(b_theta, dtype=torch.float32, device=self.device)
        self.step_size = step_size
        self.num_epochs = num_epochs
        self.burn_in_epochs = burn_in_epochs
        self.batch_size = batch_size
        self.model = model
        self.TOTAL_SAMPLE_SIZE = None
        self.samples = {'params': [], 'sigma_squared': [], 'sigma_theta_squared': [], 'sigma_lambda_squared': [], 'nu': [], 'beta': [], 'mse': []}

    def train(self, X_train, y_train):
        self.TOTAL_SAMPLE_SIZE = len(X_train)
        X = X_train.to(self.device)
        y = y_train.to(self.device)
        with torch.no_grad():
            residual_squared_sum = self._calculate_and_set_residual_squared_sum(X, y)
            sigma_squared = self._sample_sigma_squared(y.shape[0], residual_squared_sum)
            sigma_theta_squared = self._sample_sigma_theta_squared()

        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        start_time = time.time()
        #print(self.model.input_layer.beta)
        for epoch in range(self.num_epochs):
            # print(f'epoch={epoch}')
            if epoch % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.num_epochs}")
                print(f"time elapsed {time.time()-start_time} seconds")
            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                self.model.zero_grad()
                # print(f'train:: batch_idx={batch_idx} X_batch.shape={X_batch.shape}, y_batch.shape={y_batch.shape}')
                # total_grad is 1D vector, concatenated by all parameters in the model, which is a_for_eigen common practice
                residual_squared_sum = self._calculate_and_set_residual_squared_sum(X_batch, y_batch)
                total_grad = self._calculate_total_grad_and_distribute_gradient_to_each_param(y_batch, sigma_squared, sigma_theta_squared, residual_squared_sum)

                # updating the params after gradient has been distributed above
                with torch.no_grad():
                    # for name, param in self.model.named_parameters():
                    #     print(f"Layer: {name}")
                    #     print(f"Shape: {param.shape}")
                    #     print(f"Values:\n{param.data}")
                    #     print("-" * 30)
                    param_list = [p for p in self.model.parameters()]
                    for i, param in enumerate(param_list):
                        # print(f'train:: i={i} param={param}')
                        start_idx = sum(p.numel() for p in param_list[:i])
                        end_idx = start_idx + param.numel()
                        # Extract the corresponding gradient slice and reshape it to match param's shape
                        grad_slice = total_grad[start_idx:end_idx].reshape(param.shape)

                        # Generate noise with the same shape as param
                        param_noise = torch.randn_like(param) * torch.sqrt(torch.tensor(2.0 * self.step_size, device=self.device))

                        # Update param in-place
                        param.add_(self.step_size * grad_slice + param_noise)

                    sigma_squared = self._sample_sigma_squared(y_batch.shape[0], residual_squared_sum)
                    sigma_theta_squared = self._sample_sigma_theta_squared()
                    sigma_lambda_squared = self.model.sample_sigma_lambda_squared()
                    beta = self.model.get_beta()
                    nu = self.model.calculate_and_set_nu()
                    params_flat = torch.cat([p.detach().flatten() for p in self.model.parameters()]).cpu().numpy()

                    mse = residual_squared_sum / y_batch.shape[0]
                    self.samples['mse'].append(mse)
                    self.samples['params'].append(params_flat)
                    self.samples['sigma_squared'].append(sigma_squared)
                    self.samples['sigma_theta_squared'].append(sigma_theta_squared)
                    self.samples['sigma_lambda_squared'].append(sigma_lambda_squared)
                    self.samples['nu'].append(nu)
                    self.samples['beta'].append(beta)

    def _log_prob_of_prior(self, sigma_theta_squared):
        """
        Compute the probability of theta under a_for_eigen Gaussian prior N(0, sigma_beta^2 I). eq 25 and 26
        """
        # print(f'theta.shape={theta.shape} sigma_theta_squared={sigma_theta_squared} beta.shape={beta} sigma_lambda_squared={sigma_lambda_squared}')
        # if len(theta.shape) == 1:
        #     theta = theta.unsqueeze(0)  # torch.Size([2]) --> torch.Size([1, 2])
        # print(f'theta={theta}')
        # print(f'_log_prob_of_prior::type(sigma_lambda_squared)={type(sigma_lambda_squared)} {sigma_lambda_squared.shape} type(sigma_theta_squared)={type(sigma_theta_squared)} {sigma_theta_squared.shape}')

        theta = torch.cat([p.flatten() for p in self.model.parameters()]).to(self.device)
        p = theta.shape[0]
        theta_squared_sum = torch.sum(theta ** 2)
        log_norm_theta = -(p / 2) * torch.log(2 * torch.pi * torch.tensor(sigma_theta_squared))
        # print(f'_log_prob_of_prior::type(beta.numel())={type(beta.numel())} type(p)={type(p)}')
        # print(f'_log_prob_of_prior:: log_norm_theta={log_norm_theta } theta_squared_sum={theta_squared_sum}')
        # exit()
        quadratic_term_theta = -(1 / (2 * sigma_theta_squared)) * theta_squared_sum
        # print(f'_log_prob_of_prior::quadratic_term_theta={quadratic_term_theta.shape}{quadratic_term_theta.size()}')
        ans = log_norm_theta + quadratic_term_theta
        return ans

    def _calculate_and_set_residual_squared_sum(self, X, y):
        y_pred = self.model(X)
        residuals = y - y_pred
        return torch.sum(residuals**2)

    def _calculate_total_grad_and_distribute_gradient_to_each_param(self, y_batch, sigma_squared, sigma_theta_squared, residual_squared_sum):
        # calculate the total_grad and update parameters
        likelihood_grad = self._get_gradient_of_log_prob_likelihood(y_batch, sigma_squared, residual_squared_sum)
        likelihood_grad_scaled = (self.TOTAL_SAMPLE_SIZE / self.batch_size) * likelihood_grad
        prior_grad = self._get_gradient_of_log_prob_prior(sigma_theta_squared)
        total_grad = likelihood_grad_scaled + prior_grad
        return total_grad

    def _sample_sigma_squared(self, n, residual_squared_sum):
        """
        Sample σ^2 from an Inverse-Gamma distribution using equation (20).

        Args:
            n int : number of observations
            residual_squared_sum float: residual squared sum
        Returns:
            torch.Tensor: Sampled σ^2 from the Inverse-Gamma distribution.
        """
        with torch.no_grad():
            # Compute predictions (y_pred) using the model

            # print(f'y={y} y_pred={y_pred} residuals={residuals} residual_squared_sum={residual_squared_sum}')

            # New shape and scale parameters for σ^2 (equation 20)
            new_a = self.a + n / 2.0
            new_b = self.b + residual_squared_sum / 2.0

            # Sample σ^2 from Inverse-Gamma(new_shape_sigma, new_scale_sigma)
            sigma_squared = sample_inverse_gamma(new_a, new_b, size=1).squeeze()
            return sigma_squared.item()

    def _sample_sigma_theta_squared(self):
        """
        Sample σβ^2 from an Inverse-Gamma distribution using equation (21), with data dependency via β^T β.
        Returns:
            torch.Tensor: Sampled σβ^2 from the Inverse-Gamma distribution.
        """
        with torch.no_grad():
            # Number of features (p) including bias
            p = sum(p.numel() for p in self.model.parameters())  # Total number of parameters (weights + bias)

            # Compute β^T β / 2 from current model parameters
            theta = torch.cat([p.flatten() for p in self.model.parameters()]).to(self.device)  # Flatten all parameters into a_for_eigen vector
            theta_squared_sum = torch.sum(theta**2)  # β^T β
            # New shape and scale parameters for σβ^2 (equation 21)
            new_a_theta = self.a_theta + p / 2.0
            new_b_theta = self.b_theta + theta_squared_sum / 2.0

            # Sample σβ^2 from Inverse-Gamma(new_shape_beta, new_scale_beta)
            sigma_theta_squared = sample_inverse_gamma(new_a_theta, new_b_theta, size=1).squeeze()

            # Optional: Print diagnostics for debugging
            # print(f"β^T β / 2: {theta_squared_sum.item()}")
            # print(f"Shape: {new_shape_beta.item()}, Scale: {new_scale_beta.item()}")
            # print(f"Sampled σβ^2: {sigma_theta_squared.item()}")

            return sigma_theta_squared.item()

    def predict(self, X, start=0, end=-1, gamma=None):
        if gamma is not None:
            implement_mask(self.samples, gamma, self.model)
        with torch.no_grad():
            param_list = [p for p in self.model.parameters()]
            # For each sample, make new prediction and then average predictions.(Monte Carlo)
            all_predictions = []
            for sample_params in self.samples['params'][start:end]:
                start_idx = 0
                for i, param in enumerate(param_list):
                    end_idx = start_idx + param.numel()
                    param_slice = torch.tensor(sample_params[start_idx:end_idx], device=X.device).reshape(param.shape)
                    param.set_(param_slice)
                    start_idx = end_idx

                prediction = self.model(X)
                all_predictions.append(prediction.cpu().numpy())

            mean_prediction = np.mean(all_predictions, axis=0)
            variance_prediction = np.std(all_predictions, axis=0)
            print(f'predict (sample_avg)::variance_prediction={variance_prediction}')
            return torch.tensor(mean_prediction, device=X.device)

    def _get_gradient_of_log_prob_likelihood(self, y, sigma_squared, residual_squared_sum):
        """
        Compute the gradient of the log-likelihood with respect to model parameters.

        Args:
            y (torch.Tensor): Target vector of shape (batch_size,) or (batch_size, 1).
            sigma_squared (float or torch.Tensor): Standard deviation squared of the Gaussian noise.

        Returns:
            torch.Tensor: Gradient of the log-likelihood with respect to model parameters.
        """
        log_likelihood = self._log_prob_of_likelihood(y, sigma_squared, residual_squared_sum)
        log_likelihood.backward()
        # for name, param in self.model.named_parameters():
        #     print(f"Name: {name}")
        #     print(f"Shape: {param.shape}")
        #     print(f"Requires Grad: {param.requires_grad}")
        #     print(f"param: {param}")
        #     print(f"grad: {param.grad}")
        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.requires_grad])
        return total_grad

    def _get_gradient_of_log_prob_prior(self, sigma_theta_squared):
        """
        Compute the gradient of the log-prior with respect to model parameters.

        Args:
            sigma_theta_squared (float or torch.Tensor): Standard deviation squared of the prior for β.

        Returns:
            torch.Tensor: Gradient of the log-prior with respect to model parameters (flattened).
        """
        log_prior = self._log_prob_of_prior(sigma_theta_squared)
        log_prior.backward()
        total_grad = torch.cat([p.grad.flatten() for p in self.model.parameters() if p.requires_grad])
        return total_grad

    def _log_prob_of_likelihood(self, y, sigma_square, residual_squared_sum):
        """
        Compute the log-likelihood of y under a_for_eigen Gaussian likelihood model p(y; model, σ^2, X). eq 23

        Args:
            y (torch.Tensor): Target vector of shape (n,) or (n, 1).
            sigma_square (float or torch.Tensor): Standard deviation squared of the Gaussian noise (default=1.0).
        Returns:
            torch.Tensor: torch.Size([]), a_for_eigen real number.
        """
        with torch.no_grad():
            y = y.to(self.device)
            n = y.shape[0]

        # The following implementation has to use torch.log because the value is extremely small when n is big
        # and the calculation without log is very unstable
        log_norm = -(n / 2) * torch.log(2 * torch.pi * torch.tensor(sigma_square))
        quadratic_term = -(1 / (2 * sigma_square)) * residual_squared_sum
        log_likelihood = log_norm + quadratic_term
        # print(f'log_likelihood={log_likelihood} log_norm={log_norm} quadratic_term={quadratic_term}')
        # print('_log_prob_of_likelihood::residual_squared_sum', residual_squared_sum.shape)
        # print('_log_prob_of_likelihood::quadratic_term', quadratic_term.shape)
        # print(f'_log_prob_of_likelihood::log_likelihood={log_likelihood}', log_likelihood.shape)
        return log_likelihood
