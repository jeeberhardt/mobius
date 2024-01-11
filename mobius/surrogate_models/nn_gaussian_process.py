#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor with Language model
#

from abc import ABC, abstractmethod

import botorch
import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from sklearn.metrics import r2_score
from sklearn.exceptions import NotFittedError


class _SurrogateModel(ABC):

    @property
    @abstractmethod
    def X_train(self):
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


# We will use the simplest form of GP model, exact inference
class _ExactGPLModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel, transformer):
        super(_ExactGPLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.transformer = transformer

        # So the optimizer can find the transformer parameters
        self._model = self.transformer.model

    def forward(self, x):
        x = self.transformer.embed(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPLModel(_SurrogateModel):
    """
    Class for the Gaussian Process Regressor (GPR) surrogate model with Language model.

    """

    def __init__(self, kernel, transformer, finetune_transformer=False, device=None):
        """
        Initializes the Gaussian Process Regressor (GPR surrogate model with Language model.

        Parameters
        ----------
        kernel : `gpytorch.kernels.Kernel`
            The kernel specifying the covariance function of the GPR model.
        transformer : transformer
            Language Model that transforms the input into data exploitable by the GP model.
        fit_transformer : bool, default : False
            Whether to finetune the Language Model during GP fitting.
        device : str or torch.device, default : None
            Device on which to run the GP model. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.

        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._kernel = kernel
        self._transformer = transformer
        self._finetune_transformer = finetune_transformer
        self._device = device
        self._likelihood = None
        self._model = None
        self._X_train = None
        self._y_train = None

    @property
    def X_train(self):
        """
        Returns the training dataset after transformation.

        Raises
        ------
        NotFittedError
            If the model instance is not fitted yet.

        """
        if self._X_train is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        return self._X_train

    @property
    def y_train(self):
        """
        Returns the target values.

        Raises
        ------
        NotFittedError
            If the model instance is not fitted yet.

        """
        if self._y_train is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        return self._y_train

    def fit(self, X_train, y_train, y_noise=None):
        """
        Fits the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Input training dataset.
        y_train : array-like of shape (n_samples,)
            Target values.
        y_noise : array-like of shape (n_samples,), default : None
            Noise value associated to each target value (y_train), and expressed as 
            standard deviation (sigma). Values are squared internally to obtain the variance.

        """
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()
        if y_noise is not None:
            self._y_noise = np.asarray(y_noise).copy()
            self._y_noise = self._y_noise**2

        # Tokenize sequences
        X_train = self._transformer.tokenize(self._X_train)

        # Convert to torch tensors if necessary
        if not torch.is_tensor(X_train):
            X_train = torch.from_numpy(X_train).float()
        y_train = torch.from_numpy(self._y_train).float()
        if y_noise is not None:
            y_noise = torch.from_numpy(self._y_noise).float()

        # Move tensors to device
        X_train.to(self._device)
        y_train.to(self._device)
        if y_noise is not None:
            y_noise.to(self._device)

        noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)

        if self._noise is not None:
            self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=y_noise, learn_additional_noise=True)
        else:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)

        self._model = _ExactGPLModel(X_train, y_train, self._likelihood, self._kernel, self._transformer)

        # Move model and likelihood to device
        self._model.to(self._device)
        self._likelihood.to(self._device)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Set model in training mode
        self._model.train()
        self._model.transformer.train()
        self._likelihood.train()

        # Train model!
        fit_gpytorch_model(mll)

    def predict(self, X_test, y_noise=None):
        """
        Predicts using the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Query points where the GPR is evaluated.
        y_noise : array-like of shape (n_samples,), default : None
            Noise value associated to each query point (X_test), and expressed as 
            standard deviation (sigma). Values are squared internally to obtain the variance.

        Returns
        -------
        mu : array-like of shape (n_samples,)
            Mean of predictive distribution at query points.
        sigma : array-like of shape (n_samples,)
            Standard deviation of predictive distribution at query points.

        Raises
        ------
        NotFittedError
            If the model instance is not fitted yet.

        """
        X_test = np.asarray(X_test)
        if y_noise is not None:
            y_noise = np.asarray(y_noise)
            y_noise = y_noise**2
        
        if self._model is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        # Set model in evaluation mode
        self._model.eval()
        self._model.transformer.eval()
        self._likelihood.eval()

        # Tokenize sequences
        X_test = self._transformer.tokenize(np.asarray(X_test))

        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(np.asarray(X_test)).float()
        if y_noise is not None:
            y_noise = torch.from_numpy(y_noise).float()

        # Move tensors to device
        X_test.to(self._device)
        if y_noise is not None:
            y_noise.to(self._device)

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            predictions = self._likelihood(self._model(X_test), noise=y_noise)

        mu = predictions.mean.detach().numpy()
        sigma = predictions.stddev.detach().numpy()

        return mu, sigma

    def score(self, X_test, y_test):
        """
        Returns the coefficient of determination R^2.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Query points where the GPR is evaluated.
        y_test : array-like of shape (n_samples,)
            True values of `X_test`.

        Returns
        -------
        score : float
            Coefficient of determination R^2.

        """
        mu, _ = self.predict(X_test)
        return r2_score(y_test, mu)
