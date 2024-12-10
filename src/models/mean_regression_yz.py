""" This module implements estimators of E[Y|Z]
"""
import abc
import logging
from typing import List, Callable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegressionCV
from torch import nn

from src.models import ConditionalMeanOperator, MLP, DefaultCNN
from src.models.utils import ensure_two_dimensional, EarlyStopper


logger = logging.getLogger("src.models.mean_regression_yz")


class MeanRegressionYZ(abc.ABC):
    def __init__(self):
        self.is_fitted = False

    @abc.abstractmethod
    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        pass

    @abc.abstractmethod
    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        pass


class OperatorRegressionYZ(MeanRegressionYZ):
    def __init__(self):
        super().__init__()
        self.conditional_mean_operator = ConditionalMeanOperator()
    
    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        assert Y.ndim == 1
        Y = Y.reshape(-1, 1)
        best_weight, best_loss = \
                self.conditional_mean_operator.find_best_regularization_weight(Z, Y)
        logger.debug(
            f"Best conditional mean Y|Z loss: {best_loss}, with weight "
            + f"{best_weight}"
        )
        self.conditional_mean_operator.loop_fit(Z, Z_loop)
        logger.info("Conditional Mean Regressor of Y|Z fitted.")

    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        if for_loop:
            return self.conditional_mean_operator.loop_predict(Y, i)
        else:
            return self.conditional_mean_operator.predict(Y, z)


class LogisticRegressionYZ(MeanRegressionYZ):
    def __init__(self):
        self.regressor = LogisticRegressionCV(Cs=1, cv=5)

    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        self.regressor.fit(Z, Y)
        logger.info("Conditional Mean Regressor of Y|Z fitted.")

    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        if z.ndim == 1:
            z = z.reshape(1, -1)
            return self.regressor.predict_proba(z)[0, 0]
        else:
            return self.regressor.predict_proba(z)[:, 0]


class DeepRegressionYZ(MeanRegressionYZ):
    def __init__(
        self,
        inner_layers_sizes: List = [16],
        activation: str = "tanh",
        batch_size: int = 128,
        n_epochs: int = 100,
        val_split: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        dropout_rate: float = 0.2,
        activate_last_layer: bool = False,
        early_stopper: EarlyStopper = EarlyStopper(),
        loss_func: Callable = nn.MSELoss(),
    ):
        self.model = MLP(
            inner_layers_sizes=inner_layers_sizes,
            activation=activation,
            droput_rate=dropout_rate,
            activate_last_layer=activate_last_layer,
        )
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.loss_func = nn.MSELoss()
        self.early_stopper = early_stopper


    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ):
        Y = ensure_two_dimensional(Y)
        Z = ensure_two_dimensional(Z)
        Z = torch.from_numpy(Z).to(self.device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(self.device, dtype=torch.float32)
        n_val_samples = int(Z.shape[0]*self.val_split)
        n_train_samples = Z.shape[0] - n_val_samples
        n_batches = n_train_samples // self.batch_size
        Z_val = Z[:n_val_samples]
        Y_val = Y[:n_val_samples]
        Z_train = Z[n_val_samples:]
        Y_train = Y[n_val_samples:]
        dataset_train = torch.utils.data.TensorDataset(Z_train, Y_train)
        dataset_val = torch.utils.data.TensorDataset(Z_val, Y_val)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=n_val_samples, shuffle=False)

        best_val_loss = 1E6
        best_val_loss_weights = {}
        for epoch_index in range(self.n_epochs):

            # Training
            self.model.train(True)
            running_loss = 0
            for i, data in enumerate(loader_train):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if (i == 0) or ((i+1) % n_batches//4) == 0:
                    logger.debug(f" Epoch {epoch_index+1},  batch {i+1}, current batch loss: {loss.item()}")
            avg_loss = running_loss / (i + 1)

            # Validation
            self.model.eval()
            running_loss = 0
            for i, data in enumerate(loader_val):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                running_loss += loss
            val_loss = running_loss / (i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_weights = self.model.state_dict()
            logger.debug(f"Val loss {val_loss:1.2e}, train loss {avg_loss:1.2e}")

            if self.early_stopper.early_stop(val_loss):
                logger.info(f"Stopping early. Epoch {epoch_index} out of {self.n_epochs}")
                break
        logger.info(f"Fitted Y given Z regression. Best val loss: {best_val_loss:1.2e}")
        self.model.load_state_dict(best_val_loss_weights)

    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        z = torch.from_numpy(z).to(self.device, dtype=torch.float32)
        output = self.model(z).cpu().detach().numpy().ravel()
        if z.ndim == 1:
            output = output[0]
        return output


class DeepRegressionYZHighDim(MeanRegressionYZ):
    def __init__(
        self,
        batch_size: int = 128,
        n_epochs: int = 100,
        val_split: float = 0.1,
        learning_rate: float = 0.001,
        weight_decay: float = 0.001,
        early_stopper: EarlyStopper = EarlyStopper(),
        loss_func: Callable = nn.MSELoss(),
    ):
        self.model = DefaultCNN()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.val_split = val_split
        self.loss_func = nn.MSELoss()
        self.early_stopper = early_stopper


    def fit(
        self,
        Z: np.ndarray,
        Z_loop: np.ndarray,
        Y: np.ndarray,
    ):
        Y = ensure_two_dimensional(Y)
        Z = ensure_two_dimensional(Z)
        Z = torch.from_numpy(Z).to(self.device, dtype=torch.float32)
        Y = torch.from_numpy(Y).to(self.device, dtype=torch.float32)
        n_val_samples = int(Z.shape[0]*self.val_split)
        n_train_samples = Z.shape[0] - n_val_samples
        n_batches = n_train_samples // self.batch_size
        Z_val = Z[:n_val_samples]
        Y_val = Y[:n_val_samples]
        Z_train = Z[n_val_samples:]
        Y_train = Y[n_val_samples:]
        dataset_train = torch.utils.data.TensorDataset(Z_train, Y_train)
        dataset_val = torch.utils.data.TensorDataset(Z_val, Y_val)
        loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=n_val_samples, shuffle=False)

        best_val_loss = 1E6
        best_val_loss_weights = {}
        for epoch_index in range(self.n_epochs):

            # Training
            self.model.train(True)
            running_loss = 0
            for i, data in enumerate(loader_train):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()
                if (i == 0) or ((i+1) % n_batches//4) == 0:
                    logger.debug(f" Epoch {epoch_index+1},  batch {i+1}, current batch loss: {loss.item()}")
            avg_loss = running_loss / (i + 1)

            # Validation
            self.model.eval()
            running_loss = 0
            for i, data in enumerate(loader_val):
                inputs, labels = data
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)
                running_loss += loss
            val_loss = running_loss / (i + 1)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_loss_weights = self.model.state_dict()
            logger.debug(f"Val loss {val_loss:1.2e}, train loss {avg_loss:1.2e}")

            if self.early_stopper.early_stop(val_loss):
                logger.info(f"Stopping early. Epoch {epoch_index} out of {self.n_epochs}")
                break
        logger.info(f"Fitted Y given Z regression. Best val loss: {best_val_loss:1.2e}")
        self.model.load_state_dict(best_val_loss_weights)

    def predict(
        self,
        z: np.ndarray,
        Y: np.ndarray,
        i: int,
        for_loop: bool = False,
    ) -> np.ndarray:
        z = torch.from_numpy(z).to(self.device, dtype=torch.float32)
        output = self.model(z).cpu().detach().numpy().ravel()
        if z.ndim == 1:
            output = output[0]
        return output