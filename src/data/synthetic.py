""" Synthetic data generation.

"""
import logging
import random
from dataclasses import dataclass
from typing import Literal
from src.data.utils import SAGDIVDataset, KIVDataset

import numpy as np
import scipy
import torch
from torchvision import datasets, transforms


logger = logging.getLogger(__name__)

def make_benchmark_dataset(
    n_fit_samples: int = 500, 
    n_test_samples: int = 500,
    scenario: Literal["sin", "step", "abs", "linear"] = "sin",
    strong_instrument: bool = False,
    small_noise: bool = False,
):
    scenario_h_star_dict = {
        "sin": np.sin,
        "step": lambda x: x >= 0,
        "abs": np.abs,
        "linear": lambda x: x,
    }
    assert scenario in ["sin", "step", "abs", "linear"]
    h_star = scenario_h_star_dict[scenario]

    rng = np.random.default_rng()

    if small_noise:
        scale_param = 0.1
    else:
        scale_param = np.sqrt(0.1)

    Z_fit = rng.uniform(low=-3, high=3, size=(n_fit_samples, 2))
    eps = rng.normal(loc=0, scale=1, size=n_fit_samples)
    gamma = rng.normal(loc=0, scale=scale_param, size=n_fit_samples)
    delta = rng.normal(loc=0, scale=scale_param, size=n_fit_samples)
    if strong_instrument:
        X_fit = Z_fit[:, 0] + Z_fit[:, 1] + eps + gamma
    else:
        X_fit = Z_fit[:, 0] + eps + gamma
    h_star_fit = h_star(X_fit)
    Y_fit = h_star_fit + eps + delta

    if strong_instrument:
        X_test = (
            rng.uniform(low=-3, high=3, size=n_test_samples)
            + rng.uniform(low=-3, high=3, size=n_test_samples)
            + rng.normal(loc=0, scale=1, size=n_test_samples)
            + rng.normal(loc=0, scale=scale_param, size=n_test_samples)
        )
    else:
        X_test = (
            rng.uniform(low=-3, high=3, size=n_test_samples)
            + rng.normal(loc=0, scale=1, size=n_test_samples)
            + rng.normal(loc=0, scale=scale_param, size=n_test_samples)
        )
    h_star_test = h_star(X_test)

    X_fit = X_fit.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    dataset = {
        "X_fit": X_fit, "Z_fit": Z_fit, "Y_fit": Y_fit,
        "h_star_fit": h_star_fit, "X_test": X_test,
        "h_star_test": h_star_test
    }
    return dataset


def make_binary_benchmark_dataset(
    n_fit_samples: int = 500, 
    n_test_samples: int = 500,
    scenario: Literal["sin", "linear"] = "sin",
    scale=np.sqrt(0.1),
):
    scenario_h_star_dict = {
        "sin": np.sin,
        "linear": lambda x: x,
    }
    assert scenario in ["sin", "linear"]
    h_star = scenario_h_star_dict[scenario]
    conditional_mean_dict = {
        "sin": lambda Z: np.sin(Z[:, 0])*np.pi*scale*np.exp(-(0.1)/2)/np.sinh(np.pi*scale),
        "linear": lambda Z: Z[:, 0]
    }
    h_star = scenario_h_star_dict[scenario]
    conditional_mean_given = conditional_mean_dict[scenario]

    rng = np.random.default_rng()

    Z_fit = rng.uniform(low=-3, high=3, size=(n_fit_samples, 2))
    eta = rng.logistic(loc=0, scale=scale, size=n_fit_samples)
    gamma = rng.normal(loc=0, scale=np.sqrt(0.1), size=n_fit_samples)
    X_fit = Z_fit[:, 0] + eta + gamma
    h_star_fit = h_star(X_fit)
    Y_fit = (conditional_mean_given(Z_fit) + eta > 0).astype(float)
    X_test = (
        rng.uniform(low=-3, high=3, size=n_test_samples)
        + rng.logistic(loc=0, scale=scale, size=n_test_samples)
        + rng.normal(loc=0, scale=np.sqrt(0.1), size=n_test_samples)
    )
    h_star_test = h_star(X_test)
    X_fit = X_fit.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    dataset = {
        "X_fit": X_fit, "Z_fit": Z_fit, "Y_fit": Y_fit,
        "h_star_fit": h_star_fit, "X_test": X_test,
        "h_star_test": h_star_test
    }
    return dataset


if __name__ == "__main__":
    print(make_benchmark_dataset(high_dimensional_Z=True)["Z_fit"].shape)