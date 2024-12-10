""" Implements KIV algorithm for IV regression.
"""
import logging
from typing import Literal, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.data.utils import KIVDataset
from src.models.utils import ensure_two_dimensional


logger = logging.getLogger("src.models.kiv")


class KIV(BaseEstimator):
    def __init__(self):
        self.lenghtscale_z = None
        self.lenghtscale_x = None
        self.lambda_ = None
        self.W = None
        self.alpha = None

    def kernel_z(self, z_1: np.ndarray, z_2: np.ndarray):
        """Kernel used for H_Z.

        Computes the gramiam matrix of the kernel with respect to both vectors.

        Parameters
        ----------
        z_1: np.ndarray
            Array with shape (n_1, dim)
        z_2: np.ndarray
            Array with shape (n_2, dim)

        Returns
        -------
            Array with shape (n_1, n_2)

        """
        assert len(z_1.shape) == len(z_2.shape) == 2
        assert z_1.shape[1] == z_2.shape[1]
        squared_distances = distance_matrix(z_1, z_2)**2
        return np.exp(- self.lengthscale_z * squared_distances)

    def kernel_x(self, x_1: np.ndarray, x_2: np.ndarray):
        """Kernel used for H_X.

        Computes the gramiam matrix of the kernel with respect to both vectors.

        Parameters
        ----------
        x_1: np.ndarray
            Array with shape (n_1, dim)
        x_2: np.ndarray
            Array with shape (n_2, dim)

        Returns
        -------
            Array with shape (n_1, n_2)

        """
        assert len(x_1.shape) == len(x_2.shape) == 2
        assert x_1.shape[1] == x_2.shape[1]
        squared_distances = distance_matrix(x_1, x_2)**2
        return np.exp(- self.lengthscale_x * squared_distances)

    def find_and_set_best_lengthscales(self, X, Z):
        median_x = np.quantile(
            np.ravel(distance_matrix(X, X)),
            .5
        )
        median_z = np.quantile(
            np.ravel(distance_matrix(Z, Z)),
            .5
        )
        self.lengthscale_x = 1 / median_x**2
        self.lengthscale_z = 1 / median_z**2

    def find_and_set_best_regularization_weights(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
        X_tilde: np.ndarray,
        Z_tilde: np.ndarray,
        Y_tilde: np.ndarray,
        n_splits: int = 5,
        weights: list = [10**(-i) for i in range(-2, 3)],
    ):
        best_lambda, best_lambda_loss = self.find_and_set_best_lambda(
            X, Z, X_tilde, Z_tilde,
            weights=weights,
        )
        best_xi, best_xi_loss = self.find_and_set_best_xi(
            X, Z, Y, X_tilde, Z_tilde, Y_tilde,
            weights=weights,
        )
        return best_lambda, best_lambda_loss, best_xi, best_xi_loss

    def compute_loss_lambda(self, X_train, Z_train, X_test, Z_test) -> float:
        assert Z_test.shape[0] == X_test.shape[0]
        assert X_train.shape[0] == Z_train.shape[0]
        n_samples_test = Z_test.shape[0]
        n_samples_train = Z_train.shape[0]
        # Training
        self.find_and_set_best_lengthscales(X_train, Z_train)
        inv_regularized_gramian = (
            self.kernel_z(Z_train, Z_train)
            + n_samples_train*self.lambda_*np.eye(n_samples_train)
        )
        # Computing test loss
        gamma = np.linalg.solve(
            inv_regularized_gramian,
            self.kernel_z(Z_train, Z_test),
        )
        loss = np.trace(
            self.kernel_x(X_test, X_test)
            - 2 * self.kernel_x(X_test, X_train) @ gamma
            + gamma.T @ self.kernel_x(X_train, X_train) @ gamma
        ) / n_samples_test
        return loss

    def find_and_set_best_lambda(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        X_test: np.ndarray,
        Z_test: np.ndarray,
        weights: list = [10**(-i) for i in range(-2, 3)],
        base_offset = None,
        current_iter = 0,
        max_iter = 2,
    ):
        assert X_train.shape[0] == Z_train.shape[0]
        assert X_test.shape[0] == Z_test.shape[0]
        losses_by_weight = {weight: np.inf for weight in weights}
        for weight in weights:
            self.lambda_ = weight
            loss = self.compute_loss_lambda(
                X_train, Z_train, X_test, Z_test
            )
            losses_by_weight[weight] = loss
        best_weight = min(losses_by_weight, key=losses_by_weight.get)
        if current_iter == max_iter:
            best_weight_loss = losses_by_weight[best_weight]
            self.lambda_ = best_weight
            return best_weight, best_weight_loss
        elif current_iter == 0:
            # Find order of magnitude of best weight and guess more weights
            # based on that
            log_10 = lambda x: np.log(x)/np.log(10)
            base_offset = np.power(10, np.floor(log_10(best_weight)) - 1)
            new_weights = [best_weight + k*base_offset for k in range(-5, 6)]
            return self.find_and_set_best_lambda(
                X_train, Z_train,
                X_test, Z_test,
                new_weights,
                base_offset,
                current_iter+1,
                max_iter,
            )
        else:
            new_base_offset = base_offset / 10
            new_weights = [best_weight + k*new_base_offset for k in range(-5, 6)]
            return self.find_and_set_best_lambda(
                X_train, Z_train,
                X_test, Z_test,
                new_weights,
                new_base_offset,
                current_iter+1,
                max_iter
            )

    def compute_loss_xi(
        self,
        X_train,
        Z_train,
        Z_tilde_train,
        Y_tilde_train,
        X_test,
        Y_test,
    ):
        assert X_test.shape[0] == Y_test.shape[0]
        Z_whole = np.concatenate([Z_train, Z_tilde_train], axis=0)
        self.find_and_set_best_lengthscales(X_train, Z_whole)
        n_samples_first_stage = X_train.shape[0]
        n_samples_second_stage = Z_tilde_train.shape[0]
        W = (
            self.kernel_x(X_train, X_train)
            @ np.linalg.solve(
                self.kernel_x(Z_train, Z_train) + n_samples_first_stage*self.lambda_*np.eye(n_samples_first_stage),
                self.kernel_z(Z_train, Z_tilde_train)
            )
        )
        WW_m_xi_K_XX = (
            W@W.T
            + n_samples_second_stage*self.xi*self.kernel_x(X_train, X_train)
        )
        alpha = np.linalg.solve(WW_m_xi_K_XX, W@Y_tilde_train)
        h_hat = (alpha.T@self.kernel_x(X_train, X_test)).flatten()
        loss = np.sum((h_hat - Y_test)**2) / n_samples_second_stage
        return loss

    def find_and_set_best_xi(
        self,
        X: np.ndarray,
        Z: np.ndarray,
        Y: np.ndarray,
        X_tilde: np.ndarray,
        Z_tilde: np.ndarray,
        Y_tilde: np.ndarray,
        weights: list = [10**(-i) for i in range(-2, 3)],
        base_offset = None,
        current_iter = 0,
        max_iter = 2,
    ):
        losses_by_weight = {weight: np.inf for weight in weights}
        for weight in weights:
            self.xi = weight
            loss = self.compute_loss_xi(
                X, Z, Z_tilde, Y_tilde, X, Y,
            )
            losses_by_weight[weight] = loss
        best_weight = min(losses_by_weight, key=losses_by_weight.get)
        if current_iter == max_iter:
            best_weight_loss = losses_by_weight[best_weight]
            self.xi = best_weight
            return best_weight, best_weight_loss
        elif current_iter == 0:
            # Find order of magnitude of best weight and guess more weights
            # based on that
            log_10 = lambda x: np.log(x)/np.log(10)
            base_offset = np.power(10, np.floor(log_10(best_weight)) - 1)
            new_weights = [best_weight + k*base_offset for k in range(-5, 6)]
            return self.find_and_set_best_xi(
                X,
                Z,
                Y,
                X_tilde,
                Z_tilde,
                Y_tilde,
                new_weights,
                base_offset,
                current_iter+1,
                max_iter,
            )
        else:
            new_base_offset = base_offset / 10
            new_weights = [best_weight + k*new_base_offset for k in range(-5, 6)]
            return self.find_and_set_best_xi(
                X,
                Z,
                Y,
                X_tilde,
                Z_tilde,
                Y_tilde,
                new_weights,
                new_base_offset,
                current_iter+1,
                max_iter
            )

    def fit(self, dataset: KIVDataset) -> None:
        X, Z, Y, X_tilde, Z_tilde, Y_tilde = (
            dataset.X, dataset.Z, dataset.Y,
            dataset.X_tilde, dataset.Z_tilde, dataset.Y_tilde,
        )
        X = ensure_two_dimensional(X)
        X_tilde = ensure_two_dimensional(X_tilde)
        Z = ensure_two_dimensional(Z)
        Z_tilde = ensure_two_dimensional(Z_tilde)
        assert X.shape[0] == Z.shape[0] == Y.shape[0]
        assert Z_tilde.shape[0] == Y_tilde.shape[0] == X_tilde.shape[0]
        assert Z.shape[1] == Z_tilde.shape[1] 
        assert X.shape[1] == X_tilde.shape[1]
        n = X.shape[0]
        m = Z_tilde.shape[0]

        lambda_, lambda_loss, xi, xi_loss = \
                self.find_and_set_best_regularization_weights(
                    X, Z, Y, X_tilde, Z_tilde, Y_tilde,
                )
        logger.debug(f"Best lambda: {lambda_}")
        logger.debug(f"With loss: {lambda_loss:1.2e}")
        logger.debug(f"Best Xi: {xi}")
        logger.debug(f"With loss: {xi_loss:1.2e}")

        self.find_and_set_best_lengthscales(
            np.concatenate([X, X_tilde], axis=0),
            np.concatenate([Z, Z_tilde], axis=0)
        )

        self.W = self.kernel_x(X, X) @ np.linalg.solve(
            self.kernel_z(Z, Z) + n*self.lambda_*np.eye(n),
            self.kernel_z(Z, Z_tilde)
        )
        self.alpha = np.linalg.solve(
            self.W@self.W.T + m*self.xi*self.kernel_x(X, X),
            self.W
        ) @ Y_tilde
        # Needed for predict
        self.X_train = X

    def predict(self, X):
        X = ensure_two_dimensional(X)
        return (self.alpha@self.kernel_x(self.X_train, X)).flatten()
