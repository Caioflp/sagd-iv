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
from icecream import ic
from tensorflow import keras

from src.data.synthetic import make_benchmark_dataset
from src.data.utils import SAGDIVDataset, KIVDataset
from src.DeepGMM.scenarios.abstract_scenario import AbstractScenario
from src.DeepGMM.methods.toy_model_selection_method import ToyModelSelectionMethod as DeepGMM
from src.models import (
    SAGDIV, KIV, DeepDensityRatio, DeepRegressionYZ, EarlyStopper,
    AnalyticalDensityRatio, TSLS, DualIV, ModifiedDualIV
)
from src.scripts.utils import experiment


logger = logging.getLogger("src.scripts.benchmarks")


MODEL_NAMES = ["DeepGMM", "KIV", "DeepIV", "Kernel SAGD-IV", "Deep SAGD-IV", "TSLS"]
COLOR_PER_MODEL = {
    "DeepGMM": "pink",
    "KIV": "orange",
    "Dual IV": "firebrick",
    "Modified Dual IV": "saddlebrown",
    "DeepIV": "violet",
    "Kernel SAGD-IV": "darkcyan",
    "Kernel SAGD-IV true Phi": "fuchsia",
    "Deep SAGD-IV": "dodgerblue",
    "Deep SAGD-IV true Phi": "crimson",
    "TSLS": "olivedrab",
}



cm = 1/2.54
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8,
    "figure.figsize": (32*cm, 10*cm),
})


def train_eval_store_modified_dual_iv(
    data: Dict,
    n_rv_samples: int,
    model_file: Path,
):
    """ DualIV splits the training data in half for validation.

    """
    n_samples = n_rv_samples // 3 // 2
    x_train = data["X_fit"][:n_samples]
    z_train = data["Z_fit"][:n_samples]
    y_train = data["Y_fit"][:n_samples]

    x_val = data["X_fit"][n_samples:2*n_samples]
    z_val = data["Z_fit"][n_samples:2*n_samples]
    y_val = data["Y_fit"][n_samples:2*n_samples]

    x_test = data["X_test"]

    model = ModifiedDualIV()
    model.fit(x_train, z_train, y_train, x_val, z_val, y_val)
    h_hat_test = model.predict(x_test)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_dual_iv(
    data: Dict,
    n_rv_samples: int,
    model_file: Path,
):
    """ DualIV splits the training data in half for validation.

    """
    n_samples = n_rv_samples // 3 // 2
    x_train = data["X_fit"][:n_samples]
    z_train = data["Z_fit"][:n_samples]
    y_train = data["Y_fit"][:n_samples]

    x_val = data["X_fit"][n_samples:2*n_samples]
    z_val = data["Z_fit"][n_samples:2*n_samples]
    y_val = data["Y_fit"][n_samples:2*n_samples]

    x_test = data["X_test"]

    model = DualIV()
    model.fit(x_train, z_train, y_train, x_val, z_val, y_val)
    h_hat_test = model.predict(x_test)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_sagd_iv_true_Phi(
    data: Dict,
    n_rv_samples: int,
    model_file: Path,
):
    """ SAGD-IV using deep learning algorithms for \hat{Phi} and \hat{r} evaluation function.

        This version uses the analytical ratio of densities (Phi) from the DGP.

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
        activation="relu",
        batch_size=512,
        n_epochs=int(1.5*1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.003,
        dropout_rate=0,
        early_stopper=EarlyStopper(patience=10, min_delta=0.3),
    )
    density_ratio_model = AnalyticalDensityRatio()
    model = SAGDIV(
        lr="inv_n_samples",
        warm_up_duration=100,
        bound=10,
        mean_regressor_yz=mean_regressor_yz,
        density_ratio_model=density_ratio_model,
    )

    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_kernel_sagd_iv_true_Phi(
    data: Dict,
    n_rv_samples: int,
    model_file: Path,
):
    """ SAGD-IV using kernel algorithms for \hat{Phi} and \hat{r} evaluation function.

        This version uses the analytical ratio of densities (Phi) from the DGP.

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

    density_ratio_model = AnalyticalDensityRatio()

    model = SAGDIV(
        lr="inv_n_samples",
        warm_up_duration=100,
        bound=10,
        density_ratio_model=density_ratio_model,
    )
    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_tsls(
    data: Dict,
    n_rv_samples: int,
    model_file: Path,
):
    n_samples = n_rv_samples // 3

    train_x = data["X_fit"][:n_samples]
    train_z = data["Z_fit"][:n_samples]
    train_y = data["Y_fit"][:n_samples]

    test_x = data["X_test"]

    model = TSLS()
    model.fit(train_x, train_z, train_y)
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_gmm(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ DeepGMM evaluation function.

        We adopt a 80/20% train/validation split, as opposed to the 50/50% split used in the original article
    """
    enable_cuda = torch.cuda.is_available()
    n_samples = n_rv_samples // 10 // 3

    train_x = torch.as_tensor(data["X_fit"][:8*n_samples]).double()
    train_z = torch.as_tensor(data["Z_fit"][:8*n_samples]).double()
    train_y = torch.as_tensor(data["Y_fit"][:8*n_samples]).double()

    test_x = torch.as_tensor(data["X_test"]).double()

    val_x = torch.as_tensor(data["X_fit"][8*n_samples:10*n_samples]).double()
    val_z = torch.as_tensor(data["Z_fit"][8*n_samples:10*n_samples]).double()
    val_y = torch.as_tensor(data["Y_fit"][8*n_samples:10*n_samples]).double()

    if enable_cuda:
        train_x = train_x.cuda()
        train_z = train_z.cuda()
        train_y = train_y.cuda()
        val_x = val_x.cuda()
        val_z = val_z.cuda()
        val_y = val_y.cuda()
        test_x = test_x.cuda()

    method = DeepGMM(enable_cuda=enable_cuda)
    method.fit(train_x, train_z, train_y, val_x, val_z, val_y, verbose=True)
    if enable_cuda:
        h_hat_test = method.predict(test_x).cpu().detach().numpy()
    else:
        h_hat_test = method.predict(test_x).detach().numpy()
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_kernel_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ SAGD-IV using kernel algorithms for \hat{Phi} and \hat{r} evaluation function.

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

    model = SAGDIV(lr="inv_n_samples", warm_up_duration=100, bound=10)
    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_sagd_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
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
        activation="relu",
        batch_size=512//4,
        n_epochs=int(1.5*1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.003,
        dropout_rate=0.00,
        early_stopper=EarlyStopper(patience=6, min_delta=0.3),
    )
    density_ratio_model = DeepDensityRatio(
        inner_layers_sizes=[64, 32],
        activation="relu",
        batch_size=512//4,
        n_epochs=int(1.5*1E5/n_samples),
        learning_rate=0.01,
        weight_decay=0.005,
        dropout_rate=0.01,
        early_stopper=EarlyStopper(patience=15, min_delta=0.5),
    )
    model = SAGDIV(
        lr="inv_n_samples",
        warm_up_duration=100,
        bound=10,
        mean_regressor_yz=mean_regressor_yz,
        density_ratio_model=density_ratio_model,
    )

    model.fit(SAGDIVDataset(train_x, train_z, train_loop_z, train_y))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_deep_iv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ DeepIV evaluation function.

        The EconML package does not specify how to split the data, so we pass all
        samples to the fit method.

        `context` here means DeepIV's X variable, while our X variable is DeepIV's p variable.
        DeepIV requires a context variable, but this DGP does not have one.
        Hence, we will set a context variable to be constant and equal to 0.
    """
    n_samples = n_rv_samples // 3

    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 

    train_context = np.zeros((train_x.shape[0], 1), dtype=float)

    test_x = data["X_test"]

    dim_x = train_x.shape[1]
    dim_z = train_z.shape[1]
    dim_context = train_context.shape[1]

    dropout_rate = min(1000/(1000+n_samples), 0.5)

    keras_fit_options = {
        "batch_size": 100,
        "epochs": int(1.5E6/n_samples),
        "validation_split": 0.2,
        "callbacks": [keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
    }

    treatment_model = keras.Sequential([
            keras.layers.Dense(128, activation='tanh', input_shape=(dim_z+dim_context,), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(64, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.001), ),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(32, activation='tanh', kernel_regularizer=tf.keras.regularizers.L2(0.001),),
            keras.layers.Dropout(dropout_rate),
    ])

    response_model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(dim_x+dim_context,), kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.001)),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.L2(0.001)),
    ])

    model = DeepIV(
        n_components=10,
        m=lambda z, context : treatment_model(keras.layers.concatenate([z, context])),
        h=lambda x, context: response_model(keras.layers.concatenate([x, context])),
        n_samples=1,
        use_upper_bound_loss=False,
        n_gradient_samples=1,
        optimizer="adam",
        first_stage_options=keras_fit_options,
        second_stage_options=keras_fit_options,
    )
    
    # Monkey patching.
    # As of writing this code, this function on the econml.iv.nnet._deepiv module
    # needs to be updated because it calls keras.backend.logsumexp, which no longer exists.
    def fixed_mog_loss_model(n_components, d_t):
        pi = keras.layers.Input((n_components,))
        mu = keras.layers.Input((n_components, d_t))
        sig = keras.layers.Input((n_components,))
        t = keras.layers.Input((d_t,))
        # || t - mu_i || ^2
        d2 = keras.layers.Lambda(lambda d: keras.backend.sum(keras.backend.square(d), axis=-1),
                    output_shape=(n_components,))(
            keras.layers.Subtract()([keras.layers.RepeatVector(n_components)(t), mu])
        )
        # keras.layerskeras.layers = C - log(sum(pi_i/sig^d * exp(-d2/(2*sig^2))))
        # Use logsumexp for numeric stability:
        # keras.layerskeras.layers = C - log(sum(exp(-d2/(2*sig^2) + log(pi_i/sig^d))))
        # TODO: does the numeric stability actually make any difference?
        def make_logloss(d2, sig, pi):
            # return -keras.backend.logsumexp(-d2 / (2 * keras.backend.square(sig)) + keras.backend.log(pi / keras.backend.pow(sig, d_t)), axis=-1)
            return -tf.math.reduce_logsumexp(-d2 / (2 * keras.backend.square(sig)) + keras.backend.log(pi / keras.backend.pow(sig, d_t)), axis=-1)
        ll = keras.layers.Lambda(lambda dsp: make_logloss(*dsp), output_shape=(1,))([d2, sig, pi])
        m = keras.Model([pi, mu, sig, t], [ll])
        return m
    econml.iv.nnet._deepiv.mog_loss_model = fixed_mog_loss_model

    model.fit(Y=train_y, T=train_x, X=train_context, Z=train_z)
    test_context = np.zeros((test_x.shape[0], 1), dtype=float)
    h_hat_test = model.predict(test_x, test_context)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store_kiv(
        data: Dict,
        n_rv_samples: int,
        model_file: Path,
):
    """ KIV evaluation function.

        KIV needs two datasets of (X, Z, Y) samples for training and validation.
        We make each dataset have the same amount of samples.

    """
    n_samples = n_rv_samples // 3 // 2
    train_x = data["X_fit"][:n_samples] 
    train_z = data["Z_fit"][:n_samples] 
    train_y = data["Y_fit"][:n_samples] 
    train_x_tilde = data["X_fit"][n_samples:2*n_samples]
    train_z_tilde = data["Z_fit"][n_samples:2*n_samples]
    train_y_tilde = data["Y_fit"][n_samples:2*n_samples]

    test_x = data["X_test"]

    model = KIV()
    model.fit(KIVDataset(train_x, train_z, train_y, train_x_tilde, train_z_tilde, train_y_tilde))
    h_hat_test = model.predict(test_x)
    np.savez(model_file, h_hat_test=h_hat_test)
    return h_hat_test


def train_eval_store(model_name: str, *args):
    model_eval_function_dict = {
        "DeepGMM": train_eval_store_deep_gmm,
        "KIV": train_eval_store_kiv,
        "Dual IV": train_eval_store_dual_iv,
        "Modified Dual IV": train_eval_store_modified_dual_iv,
        "DeepIV": train_eval_store_deep_iv,
        "Kernel SAGD-IV": train_eval_store_kernel_sagd_iv,
        "Kernel SAGD-IV true Phi": train_eval_store_kernel_sagd_iv_true_Phi,
        "Kernel SAGD-IV true Phi": train_eval_store_kernel_sagd_iv_true_Phi,
        "Deep SAGD-IV": train_eval_store_deep_sagd_iv,
        "Deep SAGD-IV true Phi": train_eval_store_deep_sagd_iv_true_Phi,
        "TSLS": train_eval_store_tsls,
    }
    return model_eval_function_dict[model_name](*args)


def eval_models_accross_scenarios(
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = MODEL_NAMES,
    n_runs: int = 20,
    n_triplet_samples: int = 5000,
    n_rv_samples_for_fit: int = 3000,
    n_test_samples: int = 1000,
    strong_instrument: bool = False,
    small_noise: bool = False,
    generate_new_data: bool = True,
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
    n_rv_samples_for_fit: int
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
                data = make_benchmark_dataset(
                    n_triplet_samples,
                    n_test_samples,
                    scenario,
                    strong_instrument=strong_instrument,
                    small_noise=small_noise,
                )
                logger.info(f"Generated {scenario.upper()} scenario benchmark data.")
                np.savez(run_dir / "data.npz", **data)
            else:
                data = np.load(run_dir / "data.npz")
                logger.info(f"Loaded {scenario.upper()} scenario existing benchmark data.")
            for model_name in model_name_list:
                model_file = run_dir / (model_name.lower().replace(" ", "_") + ".npz")
                if retrain:
                    h_hat = train_eval_store(model_name, data, n_rv_samples_for_fit, model_file)
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
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = MODEL_NAMES,
):
    size = 12
    flierprops = dict(
        marker='o', markersize=3,
        linestyle='none', markeredgecolor='k',
    )
    n_scenarios = len(scenarios)
    fig, axs = plt.subplots(
        2, 2,
        # sharey=True,
        sharey=False,
        sharex=True,
        figsize=(15*cm, 10*cm),
    )
    axs = axs.flatten()
    for i, scenario in enumerate(scenarios):
        scenario_dir = Path(scenario)
        mse_arrays = {}
        for model_name in model_name_list:
            file_name = model_name.lower().replace(" ", "_") + "_mse_array" + ".npy"
            mse_arrays[model_name] = np.load(scenario_dir / file_name)
        mse_arrays = {k: np.log(mse_arrays[k])/np.log(10) for k in mse_arrays}
        plot = axs[i].boxplot(mse_arrays.values(), labels=mse_arrays.keys(), patch_artist=True, flierprops=flierprops)
        for patch, model_name in zip(plot['boxes'], model_name_list):
            patch.set(facecolor=COLOR_PER_MODEL[model_name])
        for line, model_name in zip(plot['medians'], model_name_list):
            line.set(color="black")
        axs[i].set_title(scenario.title(), size=size)
    # fig.tight_layout()
    # fig.text(0.5, 0.07, "Model", ha="center")
    fig.text(0.03, 0.5, "Out of sample log-MSE", va="center", rotation="vertical", size=size)
    fig.autofmt_xdate()
    fig.savefig("mse.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("mse.png", bbox_inches="tight", dpi=300)


def plot_graphs(
    n_runs: int = 20,
    scenarios: list = ["step", "sin", "abs", "linear"],
    model_name_list: list = MODEL_NAMES,
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
        figsize=(27*cm, 11*cm)
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
    fig.savefig("graph_plots.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("graph_plots.png", bbox_inches="tight", dpi=300)


@experiment("benchmark-on-deepgmm-dgp-less-data", benchmark=True)
def benchmark_on_deepgmm_dgp_with_less_data(
    n_runs=20,
    model_name_list=["DeepGMM", "KIV", "DeepIV", "Kernel SAGD-IV", "Deep SAGD-IV"],
    scenarios=["step", "abs", "linear", "sin"],
    generate_new_data=True,
    small_noise=False,
    plot=False,
    run_eval=False,
    retrain=False,
):
    if run_eval:
        eval_models_accross_scenarios( 
            scenarios=scenarios,
            model_name_list=model_name_list,
            n_runs=n_runs,
            n_triplet_samples=5000//2,
            n_rv_samples_for_fit=3000//2,
            n_test_samples=1000//2,
            generate_new_data=generate_new_data,
            small_noise=small_noise,
            retrain=retrain,
        )
    if plot:
        plot_MSEs(scenarios=scenarios, model_name_list=model_name_list)
        plot_graphs(n_runs=n_runs, scenarios=scenarios, model_name_list=model_name_list)



@experiment("benchmark-continuous-response", benchmark=True)
def benchmark_continuous_response(
    n_runs=20,
    model_name_list=["DeepGMM", "KIV", "DeepIV", "Kernel SAGD-IV", "Deep SAGD-IV"],
    scenarios=["step", "abs", "linear", "sin"],
    generate_new_data=True,
    small_noise=False,
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
            small_noise=small_noise,
            retrain=retrain,
        )
    if plot:
        plot_MSEs(scenarios=scenarios, model_name_list=model_name_list)
        plot_graphs(n_runs=n_runs, scenarios=scenarios, model_name_list=model_name_list)


if __name__ == "__main__":
    model_names = [
        "Kernel SAGD-IV",
        "Deep SAGD-IV",
        "KIV",
        "DeepGMM",
        "DeepIV",
        "TSLS",
        "Dual IV",
    ]
    benchmark_continuous_response(
        scenarios=["sin", "step", "abs", "linear"],
        n_runs=20,
        run_eval=True,
        model_name_list=model_names,
        generate_new_data=True,
        retrain=True,
        plot=True,
    )
