""" Implements Two Stages Least Squares
"""
import logging

import numpy as np
from sklearn.base import BaseEstimator


logger = logging.getLogger("src.models.kiv")


class TSLS(BaseEstimator):
    def __init__(self):
        self.beta = None

    def fit(self, X, Z, Y):
        Xt_Z = X.T @ Z
        Zt_X = Z.T @ X
        Zt_Z = Z.T @ Z
        Zt_Y = Z.T @ Y
        self.beta = np.linalg.solve(
            Xt_Z @ np.linalg.solve(Zt_Z, Zt_X),
            Xt_Z @ np.linalg.solve(Zt_Z, Zt_Y)
        ).flatten()
        
    def predict(self, X):
        return X @ self.beta
