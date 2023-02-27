#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - surrogate model
#

from abc import ABC, abstractmethod

import botorch
import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from .kernels import TanimotoSimilarityKernel


class _SurrogateModel(ABC):

    @property
    @abstractmethod
    def X_train(self):
        pass

    @property
    @abstractmethod
    def X_train_original(self):
        pass

    @property
    @abstractmethod
    def y_train(self):
        pass

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

    @abstractmethod
    def score(self):
        raise NotImplementedError()


class DummyModel(_SurrogateModel):

    def __init__(self):
        self._kernel = None
        self._input_transformer = None
        self._likelihood = None
        self._model = None
        self._X_train = np.array([])
        self._X_train_original = np.array([])
        self._y_train = np.array([])

    @property
    def X_train(self):
        return self._X_train

    @property
    def X_train_original(self):
        return self._X_train_original

    @property
    def y_train(self):
        return self._y_train

    def fit(self, X_train, y_train):
        pass

    def predict(self, X_test):
        return None, None

    def score(self, X_test, y_test):
        return None


# We will use the simplest form of GP model, exact inference
class _ExactGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
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

    def __init__(self, kernel=None, input_transformer=None):
        self._kernel = kernel
        self._input_transformer = input_transformer
        self._likelihood = None
        self._model = None
        self._X_train = None
        self._X_train_original = None
        self._y_train = None

    @property
    def X_train(self):
        if self._X_train is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._X_train

    @property
    def X_train_original(self):
        if self._X_train_original is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._X_train_original

    @property
    def y_train(self):
        if self._y_train is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._y_train

    def fit(self, X_train, y_train):
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train_original = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()

        if self._input_transformer is not None:
            # Transform input data
            self._X_train = self._input_transformer.transform(self._X_train_original)
        else:
            self._X_train = self._X_train_original

        # Convert to torch tensors
        X_train = torch.from_numpy(self._X_train).float()
        y_train = torch.from_numpy(self._y_train).float()

        # Set noise_prior to None, otherwise cannot pickle GPModel
        noise_prior = None
        #noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
        self._model = _ExactGPModel(X_train, y_train, self._likelihood, self._kernel)

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
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        # Set model in evaluation mode
        self._model.eval()
        self._likelihood.eval()

        if self._input_transformer is not None:
            # Transform input data
            X_test = self._input_transformer.transform(X_test)

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


class RFModel(_SurrogateModel):
    
    def __init__(self, input_transformer=None, **kwargs):
        self._input_transformer = input_transformer
        self._model = None
        self._X_train = None
        self._X_train_original = None
        self._y_train = None
        self._kwargs = kwargs
        
        # Set default parameters for RF
        self._kwargs.setdefault('n_estimators', 500)
        self._kwargs.setdefault('max_features', 'auto')
        self._kwargs.setdefault('max_depth', 5)
        self._kwargs.setdefault('oob_score', True)
        self._kwargs.setdefault('bootstrap', True)
        self._kwargs.setdefault('max_samples', 0.8)
        
    @property
    def X_train(self):
        if self._X_train is None:
            msg = 'This RFModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._X_train

    @property
    def X_train_original(self):
        if self._X_train_original is None:
            msg = 'This RFModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._X_train_original

    @property
    def y_train(self):
        if self._y_train is None:
            msg = 'This RFModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._y_train
    
    def fit(self, X_train, y_train):
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train_original = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()

        if self._input_transformer is not None:
            # Transform input data
            self._X_train = self._input_transformer.transform(self._X_train_original)
        else:
            self._X_train = self._X_train_original
        
        self._model = RandomForestRegressor(**self._kwargs)
        self._model.fit(self._X_train, self._y_train)
    
    def predict(self, X_test):
        if self._model is None:
            msg = 'This RFModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)
        
        if self._input_transformer is not None:
            # Transform input data
            X_test = self._input_transformer.transform(X_test)
        
        mu = self._model.predict(X_test)
        # Just a test. The uncertainty estimations should not be done this way.
        # https://stats.stackexchange.com/questions/490514/is-it-valid-to-use-the-distribution-of-individual-tree-predictions-in-a-random-f
        estimations = np.stack([t.predict(X_test) for t in self._model.estimators_])
        sigma = np.std(estimations, axis=0)
        
        return mu, sigma
    
    def score(self, X_test, y_test):
        mu, _ = self.predict(X_test)
        return r2_score(y_test, mu)
