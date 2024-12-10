"""Implements the algorithm 'Stochastic Approximate Gradient Descent IV'.

"""
import logging
from time import time
from typing import Literal, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from tqdm import tqdm

from src.data.utils import SAGDIVDataset
from src.models import (
    DensityRatio,
    KernelDensityRatio,
    ConditionalMeanOperator,
    MeanRegressionYZ,
    OperatorRegressionYZ,
)
from src.models.utils import (
    Loss,
    QuadraticLoss,
    ensure_two_dimensional,
    truncate,
)


logger = logging.getLogger("src.models.sagdiv")


class SAGDIV(BaseEstimator):
    """Regressor based on a variant of stochastic gradient descent.

    """
    def __init__(
        self,
        lr: Union[Literal["inv_sqrt", "inv_n_samples"], float] = "inv_n_samples",
        loss: Loss = QuadraticLoss(),
        mean_regressor_yz: MeanRegressionYZ = OperatorRegressionYZ(),
        density_ratio_model: DensityRatio = KernelDensityRatio(regularization="rkhs"),
        initial_value: float = 0,
        warm_up_duration: int = 100,
        bound: float = 10,
        update_scheme: Literal["sgd", "nesterov"] = "sgd",
    ):
        self.lr = lr
        self.loss = loss
        self.mean_regressor_yz = mean_regressor_yz
        self.density_ratio_model = density_ratio_model
        self.initial_value = initial_value
        self.warm_up_duration = warm_up_duration
        self.bound = bound
        self.update_scheme = update_scheme
        self.is_fitted = False
        self.fit_dataset_name = None
        self.lr_func = None
        self.conditional_mean_model_xz = None
        self.loss_derivative_array = None
        self.sequence_of_estimates = None
        self.Z_loop = None

    def fit_density_ratio_model(
        self,
        X: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        """ Fits density ratio model.

        """
        start = time()
        joint_samples = np.concatenate([X, Z], axis=1)
        independent_samples = np.concatenate([X, np.roll(Z, 2, axis=0)], axis=1)
        self.density_ratio_model.fit(joint_samples, independent_samples)
        end = time()
        logger.info("Density ratio model fitted.")
        logger.debug(f"Time to fit density ratio model: {end-start:1.2e}s")

    def fit_conditional_mean_xz(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        X: np.ndarray,
    ) -> None:
        """ Fits models for estimating the conditional expectation operators of
        X conditioned on Z.

        """
        self.conditional_mean_model_xz = ConditionalMeanOperator()
        n_samples = X.shape[0]
        # maximum iteration for regularization scheme
        log_10 = lambda x: np.log(x)/np.log(10)
        max_iter = max(1, 4 - np.floor(log_10(n_samples)))
        best_weight_xz, best_loss_xz = \
                self.conditional_mean_model_xz.find_best_regularization_weight(Z, X)
        logger.debug(
            f"Best conditional mean XZ loss: {best_loss_xz}, with weight "
            + f"{best_weight_xz}"
        )
        self.conditional_mean_model_xz.loop_fit(Z, Z_loop)
        logger.info("Conditional Mean Operator of X|Z fitted.")

    def fit(self, dataset: SAGDIVDataset) -> None:
        """Fits model to dataset.

        Parameters
        ----------
        dataset: SAGDIVDataset
            Dataset containing X, Z and Y.

        """
        fit_start = time()

        self.fit_dataset_name = dataset.name

        X, Z, Y, Z_loop = dataset.X, dataset.Z, dataset.Y, dataset.Z_loop
        X = ensure_two_dimensional(X)
        Z = ensure_two_dimensional(Z)
        Z_loop = ensure_two_dimensional(Z_loop)

        n_samples = X.shape[0]
        n_iter = Z_loop.shape[0]
        dim_x = X.shape[1]
        dim_z = Z.shape[1]

        if isinstance(self.lr, str):
            lr_dict = {
                "inv_n_samples": lambda i: 1/np.sqrt(n_samples),
                "inv_sqrt": lambda i: 1/np.sqrt(i)
            }
            self.lr_func = lr_dict[self.lr]
        elif isinstance(self.lr, float):
            self.lr_func = lambda i: self.lr

        # Create array which will store the gradient descent path of estimates
        # `n_iter+1` is due to the first estimate, which is the null function
        estimates = np.zeros((n_iter+1, n_samples), dtype=float)
        estimates[0] = self.initial_value

        self.fit_density_ratio_model(X, Z)
        # density_ratios = self.compute_density_ratios(X, Z_loop)

        self.fit_conditional_mean_xz(Z, Z_loop, X)

        self.mean_regressor_yz.fit(Z, Z_loop, Y)

        # Create array for storing \partial_{2} \ell values for later
        # calls to `predict`
        self.loss_derivative_array = np.empty(n_iter, dtype=np.float64)

        if self.update_scheme == "nesterov":
            momentum = 1/n_samples
            phi_current = estimates[0]

        execution_times = {
            "computing conditional expectations": [],
            "computing pointwise loss gradient": [],
            "computing ratio of densities": [],
            "computing functional gradient": [],
            "computing projected gradient descent step": [],
        }

        for i in tqdm(range(n_iter)):
            start = time()
            # Project current estimate on Z, i.e., compute E [h_{i-1}(X) | Z]
            projected_current_estimate = \
                    self.conditional_mean_model_xz.loop_predict(estimates[i], i)
            # Project Y on current Z
            # This looks weird, but it is necessary to pass those four arguments because there are two possible
            # options for the E[Y|Z] regressor. The logistic regression one only needs Z_loop[i], but the operator
            # regression one only needs Y, i and the for_loop boolean.
            # So we pass all of them and each one uses what it needs.
            projected_y = self.mean_regressor_yz.predict(Z_loop[i], Y, i, for_loop=True)
            end = time()
            execution_times["computing conditional expectations"].append(end-start)

            start = time()
            loss_derivative = self.loss.derivative_second_argument(
                projected_y,
                projected_current_estimate,
            )
            end = time()
            self.loss_derivative_array[i] = loss_derivative
            execution_times["computing pointwise loss gradient"].append(end-start)

            start = time()
            # ratio_of_densities = density_ratios[i]
            joint_x_and_z_i = np.concatenate(
                [X, np.full((n_samples, dim_z), Z_loop[i])],
                axis=1
            )
            ratio_of_densities = self.density_ratio_model.predict(joint_x_and_z_i)
            end = time()
            execution_times["computing ratio of densities"].append(end-start)

            # Compute the stochastic estimates for the functional loss gradient
            start = time()
            stochastic_approximate_gradient = (
                ratio_of_densities * loss_derivative
            )
            end = time()
            execution_times["computing functional gradient"].append(end-start)

            # Take one step in the negative gradient direction
            start = time()
            sagd_update = (
                estimates[i] - self.lr_func(i+1)*stochastic_approximate_gradient
            )
            if self.update_scheme == "nesterov":
                phi_next = sagd_update
                estimates[i+1] = truncate(
                    phi_next + momentum*(phi_next - phi_current),
                    self.bound
                )
                phi_current = phi_next
            else:
                estimates[i+1] = truncate(sagd_update, self.bound)
            end = time()
            execution_times["computing projected gradient descent step"].append(end-start)

        for action, times in execution_times.items():
            mean = np.mean(times)
            std = np.std(times)
            logger.debug(f"Average time spent {action}: {mean:1.2e}s±{std:1.2e}s")

        # Save Z_loop values for predict method
        self.Z_loop = Z_loop
        self.is_fitted = True
        fit_end = time()
        logger.debug(f"Time spent fitting SAGD-IV model: {fit_end-fit_start:1.2e}s")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.is_fitted
        # Compute all necessary density ratios
        X = ensure_two_dimensional(X)
        n_samples = X.shape[0]
        n_iter = self.Z_loop.shape[0]
        dim_z = self.Z_loop.shape[1]
        estimates = np.zeros((n_iter+1, X.shape[0]), dtype=np.float64)
        estimates[0] = self.initial_value
        if self.update_scheme == "nesterov":
            momentum = 1/n_iter
            phi_current = estimates[0]
        for i in tqdm(range(n_iter)):
            joint_x_and_z_i = np.concatenate(
                [X, np.full((n_samples, dim_z), self.Z_loop[i])],
                axis=1
            )
            density_ratio = self.density_ratio_model.predict(joint_x_and_z_i)
            stochastic_approximate_gradient = self.loss_derivative_array[i] * density_ratio
            sagd_update = (
                    estimates[i]
                    - self.lr_func(i+1)*stochastic_approximate_gradient
            )
            if self.update_scheme == "nesterov":
                phi_next = sagd_update
                estimates[i+1] = truncate(
                    phi_next + momentum*(phi_next - phi_current),
                    self.bound
                )
                phi_current = phi_next
            else:
                estimates[i+1] = truncate(sagd_update, self.bound)
        return np.mean(estimates[self.warm_up_duration:], axis=0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.predict(X)
