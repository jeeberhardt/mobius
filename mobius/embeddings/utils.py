#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - utils
#


import re

import numpy as np
import torch


class LoRALayer(torch.nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        """
        Initializes the LoRALayer class.

        Parameters
        ----------
        in_features : int
            Input dimension.
        out_features : int
            Output dimension.
        rank : int, default : 8
            Rank of the LoRA layer.
        alpha : int, default : 16
            Scaling factor of the LoRA layer.

        """
        super().__init__()
        self.scaling = alpha
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_features, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        """
        Computes the output of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.

        """
        x = self.scaling * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, rank=4, alpha=16):
        """
        Initializes the LinearWithLoRA class.

        Parameters
        ----------
        in_features : int
            Size of each input sample.
        out_features : int
            Size of each output sample.
        bias : bool, default : True
            If set to False, the layer will not learn an additive bias.
        rank : int, default : 4
            Rank of the LoRA layer.
        alpha : int, default : 16
            Scaling factor of the LoRA layer.

        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.lora = LoRALayer(in_features, out_features, rank, alpha)

    @staticmethod
    def from_linear(linear, rank=4, alpha=16):
        """
        Initializes the LinearWithLoRA class from a linear layer.

        Parameters
        ----------
        linear : torch.nn.Linear
            Linear layer.
        rank : int, default : 4
            Rank of the LoRA layer.
        alpha : int, default : 16
            Scaling factor of the LoRA layer.

        Returns
        -------
        LinearWithLoRA
            Linear layer with LoRA.

        """
        has_bias = True if linear.bias is not None else False

        lora_linear = LinearWithLoRA(linear.in_features, linear.out_features, has_bias, rank, alpha)
        # Replace the randomly initialized linear layer with the input linear
        lora_linear.linear = linear

        return lora_linear

    def to_linear(self):
        """
        Converts the LinearWithLoRA layer to a linear layer.

        Returns
        -------
        torch.nn.Linear
            Linear layer.

        """
        linear = self.linear
        has_bias = True if self.linear.bias is not None else False

        fused_linear = torch.nn.Linear(linear.in_features, linear.out_features, bias=has_bias)
        fused_linear.weight = self.weight
        if has_bias:
            fused_linear.bias = self.bias

        return fused_linear

    @property
    def weight(self):
        """Returns the weight of the linear layer."""
        # This is a dirty hack to make it work. I do not like it. This happens when 
        # the model is looking for the weight attribute, and does not use the forward method.
        return self.linear.weight + (self.lora.scaling * (self.lora.A @ self.lora.B))

    @property
    def bias(self):
        """Returns the bias of the linear layer."""
        return self.linear.bias

    def forward(self, x):
        """
        Computes the output of the layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        output : torch.Tensor
            Output tensor.

        """
        return self.linear(x) + self.lora(x)


def replace_layers_with_lora(model, patterns, rank=4, alpha=16):
    """
    Replaces linear layers with LoRA layers.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    patterns : str or List of str
        Patterns to select the layers to replace.
    rank : int, default : 4
        Rank of the LoRA layer.
    alpha : int, default : 16
        Scaling factor of the LoRA layer.

    Raises
    ------
    AssertionError
        If the layer to replace is not a torch.nn.Linear.

    """
    for m_name, module in dict(model.named_modules()).items():
        for c_name, layer in dict(module.named_children()).items():
            if any(re.search(pattern, f'{m_name}.{c_name}') for pattern in patterns):
                msg_error = f'LoRA can only be applied to torch.nn.Linear, but {layer} is {type(layer)}.'
                assert isinstance(layer, torch.nn.Linear), msg_error
                setattr(module, c_name, LinearWithLoRA.from_linear(layer, rank, alpha))


def select_parameters(model, parameters_names):
    """
    Selects parameters from a model based on their names.

    Parameters
    ----------
    model : torch.nn.Module
        Model.
    parameter_names : str or List of str (default=None)
        Names of the parameters to select.

    Returns
    -------
    selected_parameters : List of torch.nn.Parameter
        Selected parameters.

    """
    selected_parameters = []

    if not isinstance(parameters_names, (list, tuple, np.ndarray)):
        parameters_names = np.array([parameters_names])

    for name, param in model.named_parameters():
        if any(re.search(pattern, name) for pattern in parameters_names):
            selected_parameters.append(param)

    return selected_parameters