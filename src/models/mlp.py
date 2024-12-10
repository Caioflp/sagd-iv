""" Implements simple MLP model to fit a scalar valued function.

"""
import logging
from typing import List, Callable

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        inner_layers_sizes: List,
        activation: str = "tanh",
        droput_rate: float = 0.17,
        activate_last_layer: bool = False,
    ):
        super().__init__()
        assert activation in ["tanh", "relu", "swish", "sigmoid"]
        activation_dict = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "swish": nn.SiLU,
            "sigmoid": nn.Sigmoid,
        }
        self.activation_factory = activation_dict[activation]
        self.dropout_rate = droput_rate
        self.layers = nn.ModuleList()
        for out_dim in inner_layers_sizes:
            self.layers.append(nn.LazyLinear(out_dim))
            self.layers.append(self.activation_factory())
            self.layers.append(nn.Dropout(self.dropout_rate))
        self.layers.append(nn.LazyLinear(1))
        if activate_last_layer:
            self.layers.append(self.activation_factory())

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = MLP([10, 20, 30])
    print(model(torch.ones(3)))
    for param in model.parameters():
        print(param)