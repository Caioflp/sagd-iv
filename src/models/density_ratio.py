"""Implements density ratio estimation using the uLSIF algorithm, described in
chapter 6 of the book 'Density Ratio Estimation in Machine Learning', by
Sugiyama et al. The basis functions are gaussian kernels.

"""
import abc
import logging
from typing import List

import numpy as np
import torch
from scipy.spatial import distance_matrix
from scipy.special import erf
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold

from src.models import MLP, DefaultCNN, ResidualCNN
from src.models.utils import ensure_two_dimensional, EarlyStopper


logger = logging.getLogger("src.models.density_ratio")


class DensityRatio(BaseEstimator):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ):
        pass


class AnalyticalDensityRatio(DensityRatio):
    """ Analytical expression for density ratio used in continuous response experiments.
    """
    def __init__(self):
        super().__init__()

    def __call__(self, w):
        return self.predict(w)

    def predict(self, w):
        x = w[:, 0]
        z = w[:, 1:]
        sigma = np.sqrt(1.1)
        numerator = 3*np.sqrt(2)/sigma * np.exp(
            -1/(2*sigma**2) * (x - z[:, 0])**2
        )
        b = (x + 3)/(sigma * np.sqrt(2))
        a = (x - 3)/(sigma * np.sqrt(2))
        denominator = np.sqrt(np.pi)/2 * (
            erf(b) - erf(a)
        )
        return numerator / denominator

    def fit(self, numerator_samples, denominator_samples):
        pass


class KernelDensityRatio(DensityRatio):
    def __init__(
        self,
        regularization: str = "rkhs",
        max_support_points: int = 1000,
    ):
        super().__init__()
        self.regularization = regularization
        self.regularization_weight = 1
        self.max_support_points = max_support_points
        self.dim = None
        self.theta = None
        self.support_points = None
        self.fitted = False

    def __call__(self, w):
        return self.predict(w)

    def predict(self, w):
        assert self.fitted
        if len(w.shape) == 1:
            assert w.shape[0] == self.dim
            w = w.reshape(1, self.dim)
        elif len(w.shape) == 2:
            assert w.shape[1] == self.dim
        else:
            raise ValueError

        return np.maximum((self.kernel(w, self.support_points) @ self.theta).ravel(), 0)
        # return (self.kernel(w, self.support_points) @ self.theta).ravel()


    def kernel(self, w_1: np.ndarray, w_2: np.ndarray):
        """Kernel used for fiting.

        Computes the gramiam matrix of the kernel with respect to both vectors.

        Parameters
        ----------
        w_1: np.ndarray
            Array with shape (n_1, dim)
        w_2: np.ndarray
            Array with shape (n_2, dim)

        Returns
        -------
            Array with shape (n_1, n_2)

        """
        assert len(w_1.shape) == len(w_2.shape) == 2
        assert w_1.shape[1] == w_2.shape[1]
        squared_distances = distance_matrix(w_1, w_2)**2
        return np.exp(- self.lengthscale * squared_distances)


    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
        find_regularization_weight = True,
    ):
        """Fits estimator to provided samples.

        Uses the uLSIF algorithm, with gaussian kernel, to estimate the ratio
        of the densities of the distributions from the numerator and the
        denominator.
        Assumes that an equal number of samples from the numerator and the
        denominator was given.

        Parameters
        ----------
        numerator_samples: np.ndarray
            Array with shape `(n_samples, dim)`.
        denominator_samples: np.ndarray
            Array with shape `(n_samples, dim)`.
            of the estimator.
        max_support_points: int
            Maximum number of numerator samples to use as support points.
            Using too many has severe impacts on computation time during
            training and inference.

        """
        msg_n_samples = (
            "Provide the same number of samples for numerator and " +
            "denominator."
        )
        condition_n_samples = (
            numerator_samples.shape[0] == denominator_samples.shape[0]
        )
        assert condition_n_samples, msg_n_samples

        msg_dim = (
            "The dimension of numerator samples and denominator samples " +
            "must match."
        )

        numerator_dim = numerator_samples.shape[1]
        denominator_dim = denominator_samples.shape[1]
        condition_dim = (numerator_dim == denominator_dim)
        assert condition_dim, msg_dim

        assert self.regularization in ["l2", "rkhs"], "Unknown regularization"

        if find_regularization_weight:
            best_weight_density_ratio, best_loss_density_ratio = \
                    self.find_best_regularization_weight(
                        numerator_samples,
                        denominator_samples,
                        weights=[10**k for k in range(-3, 3)],
                        max_iter=3,
                    )
            logger.debug(
                f"Best density ratio loss: {best_loss_density_ratio}, " +
                f"with weight {best_weight_density_ratio}"
            )

        median = np.quantile(
            np.ravel(distance_matrix(numerator_samples, numerator_samples)),
            .5
        )
        self.lengthscale = 1 / median**2

        n_samples, self.dim = numerator_samples.shape
        if self.max_support_points <= n_samples:
            self.support_points = numerator_samples[:self.max_support_points]
        else:
            self.support_points = numerator_samples
        n_support_points = self.support_points.shape[0]

        K = self.kernel(self.support_points, numerator_samples)
        h_hat = np.mean(K, axis=1, keepdims=True)
        cross_K = self.kernel(self.support_points, denominator_samples)
        H_hat = cross_K @ cross_K.T / n_samples

        if self.regularization == "l2":
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * np.eye(n_support_points), h_hat
            )
        elif self.regularization == "rkhs":
            self.theta = np.linalg.solve(
                H_hat + self.regularization_weight * self.kernel(self.support_points, self.support_points),
                h_hat
            )
        self.fitted = True

    def compute_loss(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ) -> float:
        assert numerator_samples.shape == denominator_samples.shape
        assert self.fitted
        loss = np.mean(
            np.square(self.predict(denominator_samples))/2
            - self.predict(numerator_samples)
        )
        return loss

    def find_best_regularization_weight(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
        n_splits: int = 5,
        weights: list = [10**(-i) for i in range(-2, 3)],
        base_offset = None,
        current_iter = 0,
        max_iter = 2,
    ) -> float:
        """Uses K-Fold cross validation to choose regularization weight.

        Uses a recursion mechanism to better choose the regularization weight.

        """
        assert numerator_samples.shape == denominator_samples.shape
        Kf = KFold(n_splits=n_splits)
        fold_losses_by_weight = {weight: np.empty(n_splits) for weight in weights}
        for weight in weights:
            for fold, (train_idx, test_idx) in enumerate(Kf.split(numerator_samples)):
                numerator_train = numerator_samples[train_idx]
                denominator_train = denominator_samples[train_idx]
                numerator_test = numerator_samples[test_idx]
                denominator_test = denominator_samples[test_idx]
                self.regularization_weight = weight
                self.fit(numerator_train, denominator_train, find_regularization_weight=False)
                loss = self.compute_loss(numerator_test, denominator_test)
                fold_losses_by_weight[weight][fold] = loss
        cv_loss_by_weight = {
            weight: np.mean(losses)
            for weight, losses in fold_losses_by_weight.items()
        }
        best_weight = min(cv_loss_by_weight, key=cv_loss_by_weight.get)
        if current_iter == max_iter:
            best_weight_loss = cv_loss_by_weight[best_weight]
            self.regularization_weight = best_weight
            return best_weight, best_weight_loss
        elif current_iter == 0:
            log_10 = lambda x: np.log(x)/np.log(10)
            base_offset = np.power(10, np.floor(log_10(best_weight)) - 1)
            new_weights = [best_weight + k*base_offset for k in range(-5, 6)]
            return self.find_best_regularization_weight(
                numerator_samples,
                denominator_samples,
                n_splits,
                new_weights,
                base_offset,
                current_iter+1,
                max_iter,
            )
        else:
            new_base_offset = base_offset / 10
            new_weights = [best_weight + k * new_base_offset for k in range(-5, 6)]
            return self.find_best_regularization_weight(
                numerator_samples,
                denominator_samples,
                n_splits,
                new_weights,
                new_base_offset,
                current_iter+1,
                max_iter,
            )

class DeepDensityRatio(DensityRatio):
    def __init__(
        self,
        inner_layers_sizes: List = [16],
        activation: str = "tanh",
        batch_size: int = 128,
        n_epochs: int = 100,
        val_split: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        dropout_rate: float = 0.2,
        early_stopper: EarlyStopper = EarlyStopper()
    ):
        super().__init__()
        self.model = MLP(
            inner_layers_sizes=inner_layers_sizes,
            activation=activation,
            droput_rate=dropout_rate,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.early_stopper = early_stopper

    def compute_loss(
        self,
        output_on_numerator_samples: torch.Tensor,
        output_on_denominator_samples: torch.Tensor,
    ):
        loss = (
            0.5*output_on_denominator_samples.square().mean()
            - output_on_numerator_samples.mean()
        )
        return loss

    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ):
        numerator_samples = ensure_two_dimensional(numerator_samples)
        denominator_samples = ensure_two_dimensional(denominator_samples)
        numerator_samples = torch.from_numpy(numerator_samples).to(self.device, dtype=torch.float32)
        denominator_samples = torch.from_numpy(denominator_samples).to(self.device, dtype=torch.float32)
        n_samples = numerator_samples.shape[0]
        n_val_samples = int(n_samples*self.val_split)
        n_train_samples = n_samples - n_val_samples
        n_batches = n_train_samples // self.batch_size
        numerator_samples_val = numerator_samples[:n_val_samples]
        denominator_samples_val = denominator_samples[:n_val_samples]
        numerator_samples_train = numerator_samples[n_val_samples:]
        denominator_samples_train = denominator_samples[n_val_samples:]
        dataset_train = torch.utils.data.TensorDataset(numerator_samples_train, denominator_samples_train)
        dataset_val = torch.utils.data.TensorDataset(numerator_samples_val, denominator_samples_val)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=n_val_samples, shuffle=False)

        best_val_loss = 1E6
        best_val_loss_weights = {}
        for epoch_index in range(self.n_epochs):

            # Training
            self.model.train(True)
            running_loss = 0
            for i, data in enumerate(loader_train):
                batch_numerator_samples, batch_denominator_samples = data
                self.optimizer.zero_grad()

                outputs_numerator = self.model(batch_numerator_samples)
                outputs_denominator = self.model(batch_denominator_samples)
                loss = self.compute_loss(outputs_numerator, outputs_denominator)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if (i == 0) or ((i+1) % n_batches//4) == 0:
                    logger.debug(f" Epoch {epoch_index+1},  batch {i+1}, current batch loss: {loss.item()}")
            avg_loss = running_loss / (i + 1)

            # Validation
            self.model.eval()
            running_loss = 0
            for i, data in enumerate(loader_val):
                batch_numerator_samples, batch_denominator_samples = data
                outputs_numerator = self.model(batch_numerator_samples)
                outputs_denominator = self.model(batch_denominator_samples)
                loss = self.compute_loss(outputs_numerator, outputs_denominator)
                running_loss += loss
            val_loss = running_loss / (i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_weights = self.model.state_dict()
            logger.debug(f"Val loss {val_loss:1.2e}, train loss {avg_loss:1.2e}")

            if self.early_stopper.early_stop(val_loss):
                logger.info(f"Stopping early. Epoch {epoch_index} out of {self.n_epochs}")
                break
        logger.info(f"Fitted density ratio model. Best val loss: {best_val_loss:1.2e}")
        self.model.load_state_dict(best_val_loss_weights)

    def predict(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        inputs = torch.from_numpy(inputs).to(self.device, dtype=torch.float32)
        output = self.model(inputs).cpu().detach().numpy().ravel()
        if inputs.ndim == 1:
            output = output[0]
        return output


class DeepDensityRatioHighDim(DensityRatio):
    def __init__(
        self,
        batch_size: int = 128,
        n_epochs: int = 100,
        val_split: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        early_stopper: EarlyStopper = EarlyStopper()
    ):
        super().__init__()
        self.model = ResidualCNN()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.early_stopper = early_stopper

    def compute_loss(
        self,
        output_on_numerator_samples: torch.Tensor,
        output_on_denominator_samples: torch.Tensor,
    ):
        loss = (
            0.5*output_on_denominator_samples.square().mean()
            - output_on_numerator_samples.mean()
        )
        return loss

    def fit(
        self,
        numerator_samples: np.ndarray,
        denominator_samples: np.ndarray,
    ):
        numerator_samples = ensure_two_dimensional(numerator_samples)
        denominator_samples = ensure_two_dimensional(denominator_samples)
        numerator_samples = torch.from_numpy(numerator_samples).to(self.device, dtype=torch.float32)
        denominator_samples = torch.from_numpy(denominator_samples).to(self.device, dtype=torch.float32)
        n_samples = numerator_samples.shape[0]
        n_val_samples = int(n_samples*self.val_split)
        n_train_samples = n_samples - n_val_samples
        n_batches = n_train_samples // self.batch_size
        numerator_samples_val = numerator_samples[:n_val_samples]
        denominator_samples_val = denominator_samples[:n_val_samples]
        numerator_samples_train = numerator_samples[n_val_samples:]
        denominator_samples_train = denominator_samples[n_val_samples:]
        dataset_train = torch.utils.data.TensorDataset(numerator_samples_train, denominator_samples_train)
        dataset_val = torch.utils.data.TensorDataset(numerator_samples_val, denominator_samples_val)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=n_val_samples, shuffle=False)

        best_val_loss = 1E6
        best_val_loss_weights = {}
        for epoch_index in range(self.n_epochs):

            # Training
            self.model.train(True)
            running_loss = 0
            for i, data in enumerate(loader_train):
                batch_numerator_samples, batch_denominator_samples = data
                self.optimizer.zero_grad()

                outputs_numerator = self.model(batch_numerator_samples)
                outputs_denominator = self.model(batch_denominator_samples)
                loss = self.compute_loss(outputs_numerator, outputs_denominator)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if (i == 0) or ((i+1) % n_batches//4) == 0:
                    logger.debug(f" Epoch {epoch_index+1},  batch {i+1}, current batch loss: {loss.item()}")
            avg_loss = running_loss / (i + 1)

            # Validation
            self.model.eval()
            running_loss = 0
            for i, data in enumerate(loader_val):
                batch_numerator_samples, batch_denominator_samples = data
                outputs_numerator = self.model(batch_numerator_samples)
                outputs_denominator = self.model(batch_denominator_samples)
                loss = self.compute_loss(outputs_numerator, outputs_denominator)
                running_loss += loss
            val_loss = running_loss / (i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_weights = self.model.state_dict()
            logger.debug(f"Val loss {val_loss:1.2e}, train loss {avg_loss:1.2e}")

            if self.early_stopper.early_stop(val_loss):
                logger.info(f"Stopping early. Epoch {epoch_index} out of {self.n_epochs}")
                break
        logger.info(f"Fitted density ratio model. Best val loss: {best_val_loss:1.2e}")
        self.model.load_state_dict(best_val_loss_weights)

    def predict(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        inputs = torch.from_numpy(inputs).to(self.device, dtype=torch.float32)
        output = self.model(inputs).cpu().detach().numpy().ravel()
        if inputs.ndim == 1:
            output = output[0]
        return output