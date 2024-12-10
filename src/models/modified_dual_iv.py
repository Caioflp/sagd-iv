""" Implements a modified DualIV algorithm for IV regression

We substitute Y in the risk definition by E[Y|Z]. In this way, the dual function only depends on Z.

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


logger = logging.getLogger("src.models.modified_dual_iv")


class ModifiedDualIV(BaseEstimator):
    def __init__(self):
        self.lengthscale_x = None
        self.lengthscale_z = None
        self.lambda_1 = None
        self.lambda_2 = None
        self.K = None
        self. L = None
        self.M = None
        self.beta = None
    
    def kernel_z(self, z_1: np.ndarray, z_2: np.ndarray):
        """Kernel used for \mathcal{U}.

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
        assert len(z_1.shape) == len(z_2.shape) == 2
        assert z_1.shape[1] == z_2.shape[1]
        squared_distances = distance_matrix(z_1, z_2)**2
        return np.exp(- self.lengthscale_z * squared_distances)

    def kernel_x(self, x_1: np.ndarray, x_2: np.ndarray):
        """Kernel used for \mathcal{F}.

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

    def compute_loss_lambdas(self, X, Z, Y, Z_val):
        lambda_ = 1E-15
        n = X.shape[0]

        Y = ensure_two_dimensional(Y)

        self.find_and_set_best_lengthscales(X, Z)

        K = self.kernel_x(X, X)
        L = self.kernel_z(Z, Z)
        M = K @ np.linalg.solve(L + n * self.lambda_1 * np.eye(n), L)
        beta = np.linalg.solve(M @ K + n * self.lambda_2 * K, M @ Y)

        alfa = np.linalg.solve(L + n * lambda_ * np.eye(n), K @ beta - Y)
        L_tilde = self.kernel_z(Z, Z_val)

        loss = np.mean((alfa.T @ L_tilde)**2)
        return loss

    def find_and_set_best_lambdas(self, X, Z, Y, Z_val):
        values = np.array([10**(-i) for i in range(1, 10)])
        loss_by_lambda_pair = {}
        for lambda_1 in values:
            for lambda_2 in values:
                self.lambda_1 = lambda_1
                self.lambda_2 = lambda_2
                loss_by_lambda_pair[(lambda_1, lambda_2)] = self.compute_loss_lambdas(X, Z, Y, Z_val)
        self.lambda_1, self.lambda_2 = min(loss_by_lambda_pair, key=loss_by_lambda_pair.get)
        loss = loss_by_lambda_pair[(self.lambda_1, self.lambda_2)]
        return self.lambda_1, self.lambda_2, loss

    def fit(self, X, Z, Y, X_val, Z_val, Y_val):
        # Y = ensure_two_dimensional(Y)
        # Y_val = ensure_two_dimensional(Y)
        X = ensure_two_dimensional(X)
        X_val = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)
        Z_val = ensure_two_dimensional(Z)

        n = X.shape[0]

        lambda_1, lambda_2, loss =  self.find_and_set_best_lambdas(X, Z, Y, Z_val)
        logger.debug(f"Best lambda_1: {lambda_1}")
        logger.debug(f"Best lambda_2: {lambda_2}")
        logger.debug(f"With loss: {loss:1.2e}")

        self.find_and_set_best_lengthscales(
            np.concatenate([X, X_val], axis=0),
            np.concatenate([Z, Z_val], axis=0),
        )

        K = self.kernel_x(X, X)
        L = self.kernel_z(Z, Z)
        M = K @ np.linalg.solve(L + n * self.lambda_1 * np.eye(n), L)
        self.beta = np.linalg.solve(M @ K + n * self.lambda_2 * K, M @ Y)
        self.X_train = X
    
    def predict(self, X):
        X = ensure_two_dimensional(X)
        return (self.beta@self.kernel_x(self.X_train, X)).flatten()