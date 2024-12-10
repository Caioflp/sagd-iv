"""This module implements an estimator for the conditional expectation
operator from L2(X) to L2(Z). The implementation is based on the first stage of
the Kernel Instrumental Variable algorithm, and relies on kernel methods.

"""
import logging

import numpy as np

from scipy.spatial import distance_matrix
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold


logger = logging.getLogger("src.models.conditional_mean_operator")


class ConditionalMeanOperator(BaseEstimator):
    def __init__(self):
        self.lengthscale_z = None
        self.lengthscale_x = None
        self.regularization_weight = None
        self.kernel_gramian_regularized = None
        self.loop_weights = None
        self.n_samples = None
        self.loop_fitted = None
        self.z_samples = None
        self.x_samples = None


    def kernel_z(self, z_1: np.ndarray, z_2: np.ndarray):
        """Kernel used for fiting.

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
        """Kernel used for computing cv scores.

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

    def loop_fit(self, z_train: np.ndarray, z_loop: np.ndarray):
        assert len(z_train.shape) == len(z_loop.shape) == 2
        assert z_train.shape[1] == z_loop.shape[1]
        self.n_samples = z_train.shape[0]

        median_z = np.quantile(
            np.ravel(distance_matrix(z_train, z_train)),
            .5
        )
        self.lengthscale_z = 1 / median_z

        self.loop_weights = np.linalg.solve(
            (
                self.kernel_z(z_train, z_train)
                + self.regularization_weight*np.eye(self.n_samples)
            ),
            self.kernel_z(z_train, z_loop)

        )
        self.loop_fitted = True

    def fit(self, z_samples: np.ndarray, x_samples: np.ndarray):
        assert z_samples.shape[0] == x_samples.shape[0]
        self.x_samples = x_samples
        self.z_samples = z_samples
        self.n_samples = z_samples.shape[0]

        median_z = np.quantile(
            np.ravel(distance_matrix(z_samples, z_samples)),
            .5
        )
        self.lengthscale_z = 1 / median_z**2
        median_x = np.quantile(
            np.ravel(distance_matrix(x_samples, x_samples)),
            .5
        )
        # logger.debug(f"Median X distance: {median_x:1.2e}")
        # If we are computing E[Y|Z] and Y is binary, then the median is
        # 0. In that case, set the lenghtscale to 1/2
        if np.isclose(median_x, 0):
            self.lengthscale_x = .5
        else:
            self.lengthscale_x = 1 / median_x**2

        self.kernel_gramian_regularized = (
            self.kernel_z(z_samples, z_samples)
            + self.regularization_weight*np.eye(self.n_samples)
        )

    def loop_predict(self, f_samples: np.ndarray, it: int):
        """Predict method for use within the loop in the main model.
        `it` is the iteration number, starting from 0.

        f_samples must be one dimensional

        """
        assert len(f_samples.shape) == 1
        assert f_samples.size == self.n_samples
        output = self.loop_weights[:, it] @ f_samples
        return output

    def predict(self, f_samples: np.ndarray, z: np.ndarray):
        assert len(z.shape) == 2
        assert len(f_samples.shape) == 1
        assert f_samples.size == self.n_samples
        weights = np.linalg.solve(
            self.kernel_gramian_regularized,
            self.kernel_z(self.z_samples, z)
        )
        # weights' shape is (f_samples.shape[0], z.shape[0])
        result = weights.T @ f_samples
        if result.size == 1:
            return result[0]
        else:
            return result

    def compute_loss(
        self,
        z_samples: np.ndarray,
        x_samples: np.ndarray,
    ) -> float:
        assert z_samples.shape[0] == x_samples.shape[0]
        n_samples = z_samples.shape[0]
        gamma = np.linalg.solve(
            self.kernel_gramian_regularized,
            self.kernel_z(self.z_samples, z_samples),
        )
        loss = np.trace(
            self.kernel_x(x_samples, x_samples)
            - 2 * self.kernel_x(x_samples, self.x_samples) @ gamma
            + gamma.T @ self.kernel_x(self.x_samples, self.x_samples) @ gamma
        ) / n_samples
        return loss

    def find_best_regularization_weight(
        self,
        z_samples: np.ndarray,
        x_samples: np.ndarray,
        n_splits: int = 5,
        weights: list = [10**(-i) for i in range(-2, 3)],
        base_offset = None,
        offset_range = list(range(-2, 3)),
        current_iter = 0,
        max_iter = 2,
    ) -> float:
        """Uses K-Fold cross validation to choose regularization weight.

        Uses a recursion mechanism to better choose the regularization weight.

        """
        assert x_samples.shape[0] == z_samples.shape[0]
        Kf = KFold(n_splits=n_splits)
        fold_losses_by_weight = {weight: np.empty(n_splits) for weight in weights}
        for weight in weights:
            for fold, (train_idx, test_idx) in enumerate(Kf.split(z_samples)):
                z_train = z_samples[train_idx]
                x_train = x_samples[train_idx]
                z_test = z_samples[test_idx]
                x_test = x_samples[test_idx]
                self.regularization_weight = weight
                self.fit(z_train, x_train)
                loss = self.compute_loss(z_test, x_test)
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
            # Find order of magnitude of best weight and guess more weights
            # based on that
            log_10 = lambda x: np.log(x)/np.log(10)
            base_offset = np.power(10, np.floor(log_10(best_weight)) - 1)
            new_weights = [best_weight + k*base_offset for k in offset_range]
            return self.find_best_regularization_weight(
                z_samples,
                x_samples,
                n_splits,
                new_weights,
                base_offset,
                current_iter+1,
                max_iter,
            )
        else:
            new_base_offset = base_offset / 10
            new_weights = [best_weight + k*new_base_offset for k in offset_range]
            return self.find_best_regularization_weight(
                z_samples,
                x_samples,
                n_splits,
                new_weights,
                new_base_offset,
                current_iter+1,
                max_iter
            )
