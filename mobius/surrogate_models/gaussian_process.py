#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor
#

from abc import ABC, abstractmethod

import botorch
import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_model
from sklearn.metrics import r2_score


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
class _ExactGPModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(_SurrogateModel):
    """
    Class for the Gaussian Process Regressor (GPR) surrogate model.

    """

    def __init__(self, kernel, input_transformer=None):
        """
        Initializes the Gaussian Process Regressor (GPR) surrogate model.

        Parameters
        ----------
        kernel : `gpytorch.kernels.Kernel`
            The kernel specifying the covariance function of the GPR model.
        input_transformer : input transformer, default : None
            Function that transforms the input into data exploitable by the GP model.
        fit_transformer : bool, default : False
            Whether to finetune the input transformer during GP fitting.

        """
        self._kernel = kernel
        self._transformer = input_transformer
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
        RuntimeError
            If the GPModel instance is not fitted yet.

        """
        if self._X_train is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._X_train

    @property
    def y_train(self):
        """
        Returns the target values.

        Raises
        ------
        RuntimeError
            If the GPModel instance is not fitted yet.

        """
        if self._y_train is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        return self._y_train

    def fit(self, X_train, y_train):
        """
        Fits the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Input training dataset.
        y_train : array-like of shape (n_samples,)
            Target values.

        """
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()

         # Transform input data if necessary
        if self._transformer is not None:
            # No gradient computation needed, in case trasnformer is a neural network
            with torch.no_grad():
                self._X_train = self._transformer.transform(self._X_train)

        # Convert to torch tensors if necessary
        if not torch.is_tensor(self._X_train):
            self._X_train = torch.from_numpy(self._X_train).float()
        if not torch.is_tensor(self._y_train):
            self._y_train = torch.from_numpy(self._y_train).float()

        noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
        self._model = _ExactGPModel(self._X_train, self._y_train, self._likelihood, self._kernel)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Set model in training mode
        self._model.train()
        self._likelihood.train()

        # Train model!
        fit_gpytorch_model(mll)

    def predict(self, X_test):
        """
        Predicts using the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Query points where the GPR is evaluated.

        Returns
        -------
        mu : array-like of shape (n_samples,)
            Mean of predictive distribution at query points.
        sigma : array-like of shape (n_samples,)
            Standard deviation of predictive distribution at query points.

        Raises
        ------
        RuntimeError
            If instance is not fitted yet.

        """
        if self._model is None:
            msg = 'This GPModel instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise RuntimeError(msg)

        # Set model in evaluation mode
        self._model.eval()
        self._likelihood.eval()

        # Transform input data if necessary
        if self._transformer is not None:
            # No gradient computation needed, in case trasnformer is a neural network
            with torch.no_grad():
                X_test = self._transformer.transform(X_test)

        if not torch.is_tensor(X_test):
            X_test = torch.from_numpy(np.asarray(X_test)).float()

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            predictions = self._likelihood(self._model(X_test))

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
