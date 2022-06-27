#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - surrogate model
#

from abc import ABC, abstractmethod

import botorch
import gpytorch
import torch
from botorch.fit import fit_gpytorch_model
from sklearn.metrics import r2_score

from .kernels import TanimotoSimilarityKernel


class _SurrogateModel(ABC):

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def score(self):
        raise NotImplementedError()


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoSimilarityKernel())

        # make sure we're on the right device/dtype
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(_SurrogateModel):

    def __init__(self, kernel=None, data_transformer=None):
        self._kernel = kernel
        self._data_transformer = data_transformer
        self._likelihood = None
        self._model = None

    def fit(self, X_train, y_train):
        if self._data_transformer is not None:
            # Transform input data
            X_train = self._data_transformer.fit_transform(X_train)

        X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(y_train).float()

        # Set noise_prior to None, otherwise cannot pickle GPModel
        noise_prior = None
        #noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
        self._model = ExactGPModel(X_train, y_train, self._likelihood, self._kernel)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)
        mll.to(X_train)

        # Set model in training mode
        self._model.train()
        self._likelihood.train()

        # Train model!
        fit_gpytorch_model(mll)

    def predict(self, X_test):
        if self._model is None:
            msg = 'This Gaussian Process instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        # Set model in evaluation mode
        self._model.eval()
        self._likelihood.eval()

        if self._data_transformer is not None:
            # Transform input data
            X_test = self._data_transformer.fit_transform(X_test)

        X_test = torch.from_numpy(X_test).float()

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            predictions = self._likelihood(self._model(X_test))

        mu = predictions.mean.detach().numpy()
        sigma = predictions.stddev.detach().numpy()

        return mu, sigma

    def score(self, X_test, y_test):
        mu, _ = self.predict(X_test)
        return r2_score(y_test, mu)
