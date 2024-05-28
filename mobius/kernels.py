#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kernels
#

import torch
import gpytorch
from grakel.kernels import Kernel


class TanimotoSimilarityKernel(gpytorch.kernels.Kernel):
    """
    A class omputing the Tanimoto similarity coefficient between input data points.

    """

    # this kernel is non-stationary
    is_stationary = False

    # this is the kernel function
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, eps=1e-6, **params):
        """
        Computes the Tanimoto similarity kernel between two input tensors.

        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor of shape `(batch_size_1, n_features)` containing `batch_size_1` data points,
            each with `n_features` features.
        x2 : torch.Tensor
            Input tensor of shape `(batch_size_2, n_features)` containing `batch_size_2` data points,
            each with `n_features` features.
        diag : bool, default : False
            If True, return the diagonal of the kernel matrix.
        last_dim_is_batch : bool, default : False
            If True, treat the last dimension of the input tensors as a batch dimension.
        eps : float, default : 1e-6
            A small constant to add to the denominator for numerical stability.

        Returns
        -------
        torch.Tensor
            A tensor of shape `(batch_size_1, batch_size_2)` representing the 
            Tanimoto similarity coefficient between each pair of data points.

        """
        if last_dim_is_batch:
            # Not tested
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        x1s = torch.sum(torch.square(x1), dim=-1)
        x2s = torch.sum(torch.square(x2), dim=-1)

        if diag:
            if x1_eq_x2:
                res = torch.ones(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device)
                return res
            else:
                product = torch.mul(x1, x2).sum(dim=1)
                denominator = torch.add(x2s, x1s) - product
        else:
            product = torch.mm(x1, x2.transpose(1, 0))
            denominator = torch.add(x2s, x1s[:, None]) - product

        res = (product + eps) / (denominator + eps)

        return res


class CosineSimilarityKernel(gpytorch.kernels.Kernel):
    """
    A class for computing Cosine similarity coefficient.

    """

    # this kernel is non-stationary
    is_stationary = False

    # this is the kernel function
    def forward(self, x1, x2, eps=1e-6, **params):
        """
        Computes the cosine similarity coefficient between two input tensors.
        
        Parameters
        ----------
        x1 : torch.Tensor
            Input tensor of shape `(batch_size_1, n_features)` containing `batch_size_1` data points,
            each with `n_features` features.
        x2 : torch.Tensor
            Input tensor of shape `(batch_size_2, n_features)` containing `batch_size_2` data points,
            each with `n_features` features.
        eps : float, default : 1e-6
            A small constant to add to the denominator for numerical stability.
            
        Returns
        -------
        torch.Tensor
            A tensor of shape `(batch_size_1, batch_size_2)` representing the 
            Tanimoto similarity coefficient between each pair of data points.

        Notes
        -----
        This function is adapted from https://stackoverflow.com/questions/50411191/

        """
        # Normalize the rows, before computing their dot products via transposition
        x1_n = x1.norm(dim=1)[:, None]
        x2_n = x2.norm(dim=1)[:, None]

        x1_norm = x1 / torch.max(x1_n, eps * torch.ones_like(x1_n))
        x2_norm = x2 / torch.max(x2_n, eps * torch.ones_like(x2_n))

        sim_mt = torch.mm(x1_norm, x2_norm.transpose(0, 1))

        return sim_mt


class GraphKernel(gpytorch.Module):
    """
    Computes the graph distances between graphs.

    """
    def __init__(self, kernel):
        """
        Initializes the GraphKernel class.

        Parameters
        ----------
        kernel : grakel.kernels.Kernel
            The kernel to compute the graph distances.
        
        """
        super().__init__()
        self._outputscales = torch.nn.Parameter(torch.tensor([1.0], dtype=float))

        if not isinstance(kernel, Kernel):
            raise ValueError("The kernel must be an instance of grakel.kernels.Kernel.")

        self._kernel = kernel

    def forward(self, x, diag=False, eps=1e-6, **params):
        """
        Computes the graph distances between input graphs.

        Parameters
        ----------
        x : list of graphs
            Input graphs obtained using the `graph_from_network function` from grakel.
        diag : bool, default : False
            If True, return the diagonal of the kernel matrix.
        eps : float, default : 1e-6
            A small constant to add for numerical stability.

        Returns
        -------
        torch.Tensor
            A tensor of shape `(batch_size, batch_size)` representing the graph distances.
        
        """
        self._kernel.fit(x)

        if diag:
            res = self._kernel.diagonal()
        else:
            res = self._kernel.transform(x)

        # Scale the covariance matrix, replace the ScaleKernel
        res = self._outputscales * torch.tensor(res).float()

        # Add some jitter along the diagonal to make computations more stable
        jitter = max(res.diag().mean() * eps, eps)
        res += torch.eye(len(x)) * jitter

        return res
