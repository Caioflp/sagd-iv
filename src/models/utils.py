"""Model related utilities.

"""
import logging
import abc
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neighbors import KernelDensity as KDE


logger = logging.getLogger("src.models.utils")


class Loss(abc.ABC):
    """ Base class for pointwise loss function.
    """
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        pass

    @abc.abstractmethod
    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        pass

class QuadraticLoss(Loss):
    """ Quadratic loss.

    .. math::
        \ell (y, y') = 1/2 * (y - y')^2
    """
    def __init__(self):
        super().__init__()
    
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return 0.5 * (y - y_prime)**2

    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return y_prime - y


class BCELogisticLoss(Loss):
    """ BCE + logistic function loss

    .. math::
        \ell (y, y') = BCE(y, 1 - logistic(-y'))
    """
    def __init__(self, scale: float = 1):
        super().__init__()
        self.scale = scale

    def logistic(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x/self.scale))

    def BCE(
        self,
        p: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        return - ( p * np.log(q) + (1 - p) * np.log(q) )
    
    def __call__(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return self.BCE(y, self.logistic(y_prime))

    def derivative_second_argument(
        self,
        y: np.ndarray,
        y_prime: np.ndarray,
    ) -> np.ndarray:
        assert y.shape == y_prime.shape
        return 1/self.scale * (self.logistic(y_prime) - y)


def ensure_two_dimensional(arr: np.ndarray):
    """Ensures passed array has two dimensions
    
    Parameters
    ----------
    arr : np.ndarray
        array of shape (n_samples,) or (n_samples, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples, n_features)

    Raises
    ------
    ValueError if the array has more than two dimensions.

    """
    if len(arr.shape) == 1:
        return arr.reshape(-1, 1)
    elif len(arr.shape) == 2:
        return arr
    else:
        raise ValueError


def truncate(arr: np.ndarray, M: float):
    """Truncate the values of `arr` inside [-M, M]
    """
    return np.minimum(np.maximum(arr, 0), M) - np.minimum(np.maximum(-arr, 0), M)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False