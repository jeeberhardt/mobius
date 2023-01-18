#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Kernels
#

import torch
import gpytorch


class TanimotoSimilarityKernel(gpytorch.kernels.Kernel):
    # the sequence kernel is stationary
    is_stationary = True

    # this is the kernel function
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, eps=1e-6, **params):
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
    # the sequence kernel is stationary
    is_stationary = True     

    # this is the kernel function
    def forward(self, x1, x2, eps=1e-6, **params):
        # Source: https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
        # Normalize the rows, before computing their dot products via transposition
        x1_n = x1.norm(dim=1)[:, None]
        x2_n = x2.norm(dim=1)[:, None]

        x1_norm = x1 / torch.max(x1_n, eps * torch.ones_like(x1_n))
        x2_norm = x2 / torch.max(x2_n, eps * torch.ones_like(x2_n))
        
        sim_mt = torch.mm(x1_norm, x2_norm.transpose(0, 1))
        
        return sim_mt
