""" Implements DualIV algorithm for IV regression

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


logger = logging.getLogger("src.models.dual_iv")


class DualIV(BaseEstimator):
    def __init__(self):
        self.lengthscale_x = None
        self.lengthscale_w = None
        self.lambda_1 = None
        self.lambda_2 = None
        self.K = None
        self. L = None
        self.M = None
        self.beta = None
    
    def kernel_w(self, w_1: np.ndarray, w_2: np.ndarray):
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
        assert len(w_1.shape) == len(w_2.shape) == 2
        assert w_1.shape[1] == w_2.shape[1]
        squared_distances = distance_matrix(w_1, w_2)**2
        return np.exp(- self.lengthscale_w * squared_distances)

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

    def find_and_set_best_lengthscales(self, X, Z, Y):
        Y = ensure_two_dimensional(Y)
        W = np.concatenate([Y, Z], axis=1)
        median_x = np.quantile(
            np.ravel(distance_matrix(X, X)),
            .5
        )
        median_w = np.quantile(
            np.ravel(distance_matrix(W, W)),
            .5
        )
        self.lengthscale_x = 1 / median_x**2
        self.lengthscale_w = 1 / median_w**2

    def compute_loss_lambdas(self, X, Z, Y, X_val, Z_val, Y_val):
        lambda_ = 1E-7
        n = X.shape[0]

        Y = ensure_two_dimensional(Y)
        W = np.concatenate([Y, Z], axis=1)
        Y_val = ensure_two_dimensional(Y_val)
        W_val = np.concatenate([Y_val, Z_val], axis=1)

        self.find_and_set_best_lengthscales(X, Z, Y)

        K = self.kernel_x(X, X)
        L = self.kernel_w(W, W)
        M = K @ np.linalg.solve(L + n * self.lambda_1 * np.eye(n), L)
        beta = np.linalg.solve(M @ K + n * self.lambda_2 * K, M @ Y)

        alfa = np.linalg.solve(L + n * lambda_ * np.eye(n), K @ beta - Y)
        L_tilde = self.kernel_w(W, W_val)

        loss = np.mean((alfa.T @ L_tilde)**2)
        return loss

    def find_and_set_best_lambdas(self, X, Z, Y, X_val, Z_val, Y_val):
        values = np.array([10**(-i) for i in range(1, 10)])
        loss_by_lambda_pair = {}
        for lambda_1 in values:
            for lambda_2 in values:
                self.lambda_1 = lambda_1
                self.lambda_2 = lambda_2
                loss_by_lambda_pair[(lambda_1, lambda_2)] = self.compute_loss_lambdas(X, Z, Y, X_val, Z_val, Y_val)
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

        lambda_1, lambda_2, loss =  self.find_and_set_best_lambdas(X, Z, Y, X_val, Z_val, Y_val)
        logger.debug(f"Best lambda_1: {lambda_1}")
        logger.debug(f"Best lambda_2: {lambda_2}")
        logger.debug(f"With loss: {loss:1.2e}")

        self.find_and_set_best_lengthscales(
            np.concatenate([X, X_val], axis=0),
            np.concatenate([Z, Z_val], axis=0),
            np.concatenate([Y, Y_val], axis=0)
        )

        W = np.concatenate([Y[:, np.newaxis], Z], axis=1)
        K = self.kernel_x(X, X)
        L = self.kernel_w(W, W)
        M = K @ np.linalg.solve(L + n * self.lambda_1 * np.eye(n), L)
        self.beta = np.linalg.solve(M @ K + n * self.lambda_2 * K, M @ Y)
        self.X_train = X
    
    def predict(self, X):
        X = ensure_two_dimensional(X)
        return (self.beta@self.kernel_x(self.X_train, X)).flatten()