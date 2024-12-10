"""Script to compare SAGDIV with KIV, DeepGMM and DeepIV

"""
import logging
import os
from pathlib import Path
from typing import Tuple, Dict

import econml
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import torch
from econml.iv.nnet import DeepIV
from tensorflow import keras

from src.data.synthetic import make_benchmark_dataset, make_binary_benchmark_dataset
from src.data.utils import SAGDIVDataset, KIVDataset
from src.DeepGMM.scenarios.abstract_scenario import AbstractScenario
from src.DeepGMM.methods.toy_model_selection_method import ToyModelSelectionMethod as DeepGMM
from src.models import (
    SAGDIV, KIV, DeepDensityRatio, DeepRegressionYZ, EarlyStopper
)
from src.models.utils import BCELogisticLoss
from src.scripts.utils import experiment


logger = logging.getLogger("src.scripts.benchmarks")
COLOR_PER_MODEL = {
    "DeepGMM": "pink",
    "KIV": "orange",
    "DeepIV": "violet",
    # "SAGD-IV": "lightblue",
    "Kernel SAGD-IV": "darkcyan",
    "Deep SAGD-IV": "dodgerblue",
    "Kernel": "darkcyan",
    "Deep": "dodgerblue",
    "Binary SAGD-IV": "seagreen",
}


cm = 1/2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 10,
    "figure.figsize": (22*cm, 10*cm),
})


def train_eval_store_kernel_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
        scale: float,
):
    """ SAGD-IV using RKHS algorithms for \hat{Phi} and \hat{r} evaluation function.

        Let N denote the number of triplets (X, Y, Z) that will be used to fit r, Phi and P.
        We want the number of Z samples used in the loop to be 2*N.
        Hence, we have
            n_rv_samples = 3*N + 2*N = 5*N
        and N = n_rv_samples // 5

    """

    n_samples = n_rv_samples // 5
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    test_x = data["X_test"]

    train_loop_z = data["Z_fit"][n_samples:n_samples + 2*n_samples]

    model = SAGDIV(
        lr="inv_n_samples",
        loss=BCELogisticLoss(scale=scale),
        initial_value=0,
        warm_up_duration=100,
        bound=10,
    )

    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_binary_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
        scale: float,
):
    """ SAGD-IV using deep learning for computing \hat{r} and RKHS for computing \hat{\Phi}.

        Let N denote the number of triplets (X, Y, Z) that will be used to fit r, Phi and P.
        We want the number of Z samples used in the loop to be 2*N.
        Hence, we have
            n_rv_samples = 3*N + 2*N = 5*N
        and N = n_rv_samples // 5

    """

    n_samples = n_rv_samples // 5
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    test_x = data["X_test"]

    train_loop_z = data["Z_fit"][n_samples:n_samples + 2*n_samples]
    mean_regressor_yz = DeepRegressionYZ(
        inner_layers_sizes=[64, 32],
        activation="sigmoid",
        batch_size=128,
        n_epochs=int(1.5*1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.001,
        dropout_rate=0,
        early_stopper=EarlyStopper(patience=10, min_delta=0.3),
        loss_func=torch.nn.BCELoss(),
        activate_last_layer=True,
    )
    model = SAGDIV(
        lr="inv_n_samples",
        loss=BCELogisticLoss(scale=scale),
        mean_regressor_yz=mean_regressor_yz,
        initial_value=0,
        warm_up_duration=100,
        bound=10,
    )

    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
        scale: float,
):
    """ SAGD-IV using deep learning algorithms for \hat{Phi} and \hat{r} evaluation function.

        Let N denote the number of triplets (X, Y, Z) that will be used to fit r, Phi and P.
        We want the number of Z samples used in the loop to be 2*N.
        Hence, we have
            n_rv_samples = 3*N + 2*N = 5*N
        and N = n_rv_samples // 5

    """

    n_samples = n_rv_samples // 5
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    test_x = data["X_test"]

    train_loop_z = data["Z_fit"][n_samples:n_samples + 2*n_samples]
    mean_regressor_yz = DeepRegressionYZ(
        inner_layers_sizes=[64, 32],
        activation="sigmoid",
        batch_size=128,
        n_epochs=int(1.5*1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.001,
        dropout_rate=0,
        early_stopper=EarlyStopper(patience=10, min_delta=0.3),
        loss_func=torch.nn.BCELoss(),
        activate_last_layer=True,
    )
    density_ratio_model = DeepDensityRatio(
        inner_layers_sizes=[64, 32],
        activation="sigmoid",
        batch_size=128,
        n_epochs=int(1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.001,
        dropout_rate=0.01,
        early_stopper=EarlyStopper(patience=10, min_delta=0.5),
    )
    model = SAGDIV(
        lr="inv_n_samples",
        loss=BCELogisticLoss(scale=scale),
        mean_regressor_yz=mean_regressor_yz,
        density_ratio_model=density_ratio_model,
        initial_value=0,
        warm_up_duration=100,
        bound=10,
    )

    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store(model_name: str, *args):
    model_eval_function_dict = {
        "Kernel SAGD-IV": train_eval_store_kernel_sagd_iv,
        "Deep SAGD-IV": train_eval_store_deep_sagd_iv,
        "Binary SAGD-IV": train_eval_store_binary_sagd_iv,
    }
    return model_eval_function_dict[model_name](*args)


def eval_models_accross_scenarios(
    scenarios: list = ["sin", "linear"],
    model_name_list: list = ["Deep SAGD-IV", "Kernel SAGD-IV", "Binary SAGD-IV"],
    n_runs: int = 20,
    scale: float = np.sqrt(0.1),
    n_triplet_samples: int = 5000,
    n_rv_samples_for_fit: int = 3000,
    n_test_samples: int = 1000,
    generate_new_data: bool = False,
    retrain: bool = False,
):
    """ Evaluates each model `n_runs` times in each scenario.

    Since different algorithms require different proportions of X, Z and Y samples,
    our experiment assumes that each model has access to the same total amount of random
    variable samples, but those might be distributed differently accross models.
    For example, if all models should be fitted using 3000 r.v. variables, then DeepGMM
    may use 800 (X, Y, Z) samples for training and another 200 (X, Y, Z) samples for validation,
    while SAGD-IV might use 600 (X, Y, Z) samples for fitting the preliminary r, P and Phi
    estimators and another 1200 Z samples for the SAGD loop.

    Parameters
    ----------
    scenarios: list
        All scenarios which should be evaluated.
    n_runs: int
        Number times to evaluate each model in each scenario.
    n_triplet_samples: int
        Amount of (X, Y, Z) triplet samples to draw. It might be the case that not all of them
        will be actually used. We just want to make sure every algorithm gets what it needs.
    n_rv_samples_for_fitting: int
        Amount of random variable samples each model is allowed to use during the fitting process.
    n_test_samples: int
        Number of X samples in which the models will be evaluated.
    strong_instrument: bool
        Determines if X = Z_1 or X = Z_1 + Z_2, with the second option resulting in a stronger
        instrumental variable.


    Experiment structure:

    for each scenario:
        for each run:
            generate and save new data
            for each method:
                train method
                store predictions on x_test
                compute test mse
        store test mse data accross runs for each model

    """

    message = f"""
    This function was called with:
    \tgenerate_new_data = {generate_new_data}
    \tretrain = {retrain}
    \tmodel_name_list = {model_name_list}
    \tscenarios = {scenarios}
    """
    if generate_new_data:
        message += "\nThis will DELETE all stored synthetic data."
    if retrain:
        message += f"\nThis will RETRAIN the selected models and RECOMPUTE their predictions."
    message += "\nContinue? (y/n) "
    permission = input(message)
    if permission != "y":
        raise Exception("Experiment terminated.")
    model_mse_dict = {name: np.empty(n_runs, dtype=float) for name in model_name_list}
    for scenario in scenarios:
        scenario_dir = Path(scenario)
        scenario_dir.mkdir(exist_ok=True)
        logger.info(f"Entering scenario: {scenario.upper()}")
        for run_number in range(n_runs):
            logger.info(f"Starting run number {run_number}")
            run_dir = scenario_dir / ("run_" + str(run_number))
            run_dir.mkdir(exist_ok=True)
            if generate_new_data:
                data = make_binary_benchmark_dataset(
                    n_triplet_samples,
                    n_test_samples,
                    scenario,
                    scale,
                )
                logger.info(f"Generated {scenario.upper()} scenario benchmark data.")
                np.savez(run_dir / "data.npz", **data)
            else:
                data = np.load(run_dir / "data.npz")
                logger.info(f"Loaded {scenario.upper()} scenario existing benchmark data.")
            for model_name in model_name_list:
                model_file = run_dir / (model_name.lower().replace(" ", "_") + ".npz")
                if retrain:
                    h_hat = train_eval_store(model_name, data, n_rv_samples_for_fit, model_file, scale)
                else:
                    h_hat = np.load(model_file)["h_hat_test"]
                mse = np.mean(np.square(data["h_star_test"] - h_hat))
                model_mse_dict[model_name][run_number] = mse
            logger.info("Evaluated all models")
        report = f"MSE results for scenario {scenario.upper()}:\n"
        for model_name in model_name_list:
            mse_array = model_mse_dict[model_name]
            mean, std = np.mean(mse_array), np.std(mse_array)
            report += f"{model_name}: {mean:1.2e} ± {std:1.2e}\n" 
            np.save(scenario_dir / (model_name.lower().replace(" ", "_") + "_mse_array" + ".npy"), mse_array)
        logger.info(report)


def plot_MSEs(
    scenarios: list = ["sin", "linear"],
    model_name_list: list = ["Deep SAGD-IV", "Kernel SAGD-IV", "Binary SAGD-IV"],
):
    flierprops = dict(
        marker='o', markersize=3,
        linestyle='none', markeredgecolor='k',
    )
    n_scenarios = len(scenarios)
    fig, axs = plt.subplots(
        1, 2, sharey=True, sharex=True,
        figsize=(9*cm, 5*cm),
    )
    axs = axs.flatten()
    for i, scenario in enumerate(scenarios):
        scenario_dir = Path(scenario)
        mse_arrays = {}
        for model_name in model_name_list:
            file_name = model_name.lower().replace(" ", "_") + "_mse_array" + ".npy"
            mse_arrays[model_name.split(" ")[0]] = np.load(scenario_dir / file_name)
        mse_arrays = {k: np.log(mse_arrays[k])/np.log(10) for k in mse_arrays}# if k in model_name_list}
        plot = axs[i].boxplot(mse_arrays.values(), labels=mse_arrays.keys(), patch_artist=True, flierprops=flierprops)
        for patch, model_name in zip(plot['boxes'], model_name_list):
            patch.set(facecolor=COLOR_PER_MODEL[model_name])
        for line, model_name in zip(plot['medians'], model_name_list):
            line.set(color="black")
        axs[i].set_title(scenario.title())
    # fig.tight_layout()
    fig.text(0.5, -0.1, "Model", ha="center")
    fig.text(-0.05, 0.5, "Out of sample log-MSE", va="center", rotation="vertical")
    # fig.autofmt_xdate()
    fig.savefig("mse_binary.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("mse_binary.png", bbox_inches="tight", dpi=300)


def plot_graphs(
    n_runs: int = 20,
    scenarios: list = ["sin", "linear"],
    model_name_list: list = ["Deep SADG-IV", "Kernel SAGD-IV", "Binary SAGD-IV"],
    linewidth: int = 1,
    fraction_of_data_to_plot: float = 0.3,
):
    n_scenarios = len(scenarios)
    n_models = len(model_name_list)
    # Choose a random run
    random_run = np.random.choice(n_runs)

    fig, axs = plt.subplots(
        n_scenarios,
        n_models+1,
        sharey="row",
        sharex=True,
        figsize=(16*cm, 6*cm)
        )
    # fig.tight_layout()
    for i, scenario in enumerate(scenarios):
        run_dir = Path(scenario + "/" + f"run_{random_run}")
        data_file = run_dir / "data.npz"
        data = np.load(data_file)
        n_samples_plot = int(len(data["X_fit"])*fraction_of_data_to_plot)
        X = data["X_fit"][:n_samples_plot].flatten()
        Y = data["Y_fit"][:n_samples_plot]
        # Plot the data
        axs[i, 0].scatter(
            X, Y, c="g", s=.5, alpha=0.5,
        )
        # Sorting is necessary to make line plots
        sort_idx = np.argsort(X)
        sorted_x = X[sort_idx]
        sorted_h_star = data["h_star_fit"][:n_samples_plot][sort_idx]
        axs[i, 0].plot(
            sorted_x,
            sorted_h_star,
            c="k",
            linewidth=linewidth,
        )
        for j, model_name in enumerate(model_name_list, start=1):
            # Plot model estimates
            axs[i, j].plot(
                sorted_x,
                sorted_h_star,
                c="k",
                linewidth=linewidth,
            )
            X_test = data["X_test"].flatten()
            sorted_idx = np.argsort(X_test)
            sorted_x_test = X_test[sorted_idx]
            model_file = run_dir / (model_name.lower().replace(" ", "_") + ".npz")
            h_hat_test = np.load(model_file)["h_hat_test"]
            sorted_h_hat = h_hat_test[sorted_idx]
            axs[i, j].plot(
                sorted_x_test,
                sorted_h_hat,
                c="b",
                linewidth=linewidth,
            )
    cols = ["Data"] + model_name_list
    rows = [scenario.title() for scenario in scenarios]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(row)#, rotation=0)#, size='large')
    fig.savefig("graph_plots_binary.pdf", dpi=300)
    fig.savefig("graph_plots_binary.png", dpi=300)


@experiment("benchmark-binary-response", benchmark=True)
def benchmark_binary_response(
    n_runs=20,
    model_name_list=["Deep SAGD-IV", "Kernel SAGD-IV", "Binary SAGD-IV"],
    scenarios=["linear", "sin"],
    generate_new_data=False,
    plot=False,
    run_eval=False,
    retrain=False,
):
    if run_eval:
        eval_models_accross_scenarios( 
            scenarios=scenarios,
            model_name_list=model_name_list,
            n_runs=n_runs,
            n_triplet_samples=5000,
            n_rv_samples_for_fit=3000,
            n_test_samples=1000,
            generate_new_data=generate_new_data,
            retrain=retrain,
        )
    if plot:
        plot_MSEs(scenarios=scenarios, model_name_list=model_name_list)
        plot_graphs(n_runs=n_runs, scenarios=scenarios, model_name_list=model_name_list)


if __name__ == "__main__":
    model_names = [
        "Kernel SAGD-IV",
        "Deep SAGD-IV",
    ]
    benchmark_binary_response(run_eval=True, model_name_list=model_names, generate_new_data=True, retrain=True, plot=True)