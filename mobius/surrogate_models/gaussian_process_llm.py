#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor with pretrained protein Language Models (pLM)
#

import botorch
import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from sklearn.exceptions import NotFittedError

from .surrogate_model import _SurrogateModel
from ..utils import ProgressBar


def show_optimization_progress(parameters, OptimizationResult):
    print(f'Step: {OptimizationResult.step:04d} - Loss: {OptimizationResult.fval:.5f}')


# We will use the simplest form of GP model, exact inference
class _ExactGPLLModel(gpytorch.models.ExactGP, botorch.models.gpytorch.GPyTorchModel):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel, transformer):
        super(_ExactGPLLModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.transformer = transformer

        # So the optimizer can find the transformer parameters
        self._model = self.transformer.model

    def forward(self, tokens):
        x = self.transformer.embed(tokens)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPLLModel(_SurrogateModel):
    """
    Class for the Gaussian Process Regressor (GPR) surrogate model for pretrained protein Language Models (pLM).

    """

    def __init__(self, kernel, pretained_model, noise_prior=None, missing_values=False, device=None, show_progression=True):
        """
        Initializes the Gaussian Process Regressor (GPR) surrogate model for pretrained protein Language Models (pLM).

        Parameters
        ----------
        kernel : `gpytorch.kernels.Kernel`
            The kernel function used by the GPR model.
        pretained_model : `PreTrainedModel`
            Pretrained protein Language Model (pLM) that transforms the input into data exploitable by the GP model.
        noise_prior : `gpytorch.priors.Prior`, default : None
            Prior distribution for the noise term in the likelihood function.
        missing_values : bool, default : False
            Whether we support missing values in the input data.
        device : str or torch.device, default : None
            Device on which to run the GP model. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.
        show_progression : bool, default : True
            Whether to show the progression of the optimization.

        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if noise_prior is not None:
            if not isinstance(noise_prior, gpytorch.priors.Prior):
                raise ValueError("The noise prior must be an instance of gpytorch.priors.Prior.")

        self._kernel = kernel
        self._pretained_model = pretained_model
        self._noise_prior = noise_prior
        self._missing_values = missing_values
        self._device = device
        self._show_progression = show_progression
        self._likelihood = None
        self._model = None
        self._X_train = None
        self._y_train = None
        self._y_noise = None
    
    @property
    def device(self):
        """Returns the device on which the model is running."""
        return self._device

    def fit(self, X_train, y_train, y_noise=None):
        """
        Fits the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_train : list of polymers (if transformer defined) or array-like of shape (n_samples, n_features)
            Data to be used for training the GPR model.
        y_train : array-like of shape (n_samples,)
            Target values.
        y_noise : array-like of shape (n_samples,), default : None
            Known observation noise (variance) for each training example (y_train). If your noise 
            is expressed as standard deviation (sigma), you need to square the values to obtain 
            the variance (variance = sigma**2).

        """
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()
        if y_noise is not None:
            self._y_noise = np.asarray(y_noise).copy()
            self._y_noise = self._y_noise

        # Check that the number of polymers in X_train, y_train and y_noise are the same
        msg_error = "The number of sequences in X_train and values in y_train must be the same."
        assert self._X_train.shape[0] == self._y_train.shape[0], msg_error
        if y_noise is not None:
            msg_error = "The number of sequences in X_train and values in y_noise must be the same."
            assert self._X_train.shape[0] == self._y_noise.shape[0], msg_error

        # Tokenize sequences
        X_tokens = self._pretained_model.tokenize(self._X_train)

        # Convert to torch tensors if necessary
        if not torch.is_tensor(X_tokens):
            X_tokens = torch.from_numpy(np.asarray(X_tokens)).float()
        y_train = torch.from_numpy(self._y_train).float()
        if y_noise is not None:
            y_noise = torch.from_numpy(self._y_noise).float()

        # Move tensors to device
        X_tokens = X_tokens.to(self._device)
        y_train = y_train.to(self._device)
        if y_noise is not None:
            y_noise = y_noise.to(self._device)

        if self._missing_values:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=self._noise_prior)
        elif y_noise is not None:
            self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=y_noise, learn_additional_noise=True)
        else:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=self._noise_prior)

        self._model = _ExactGPLLModel(X_tokens, y_train, self._likelihood, self._kernel, self._pretained_model)

        # Move GP, pLM (inside GP) and likelihood to device
        self._model.to(self._device)
        self._likelihood.to(self._device)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Set model in training mode
        self._model.train()
        self._model.transformer.train()
        self._likelihood.train()

        # Train model!
        if self._show_progression:
            optimizer_kwargs={'callback': ProgressBar()}
        else:
            optimizer_kwargs=None

        fit_gpytorch_mll(mll, optimizer_kwargs=optimizer_kwargs)

    def predict(self, X_test, y_noise=None):
        """
        Predicts using the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_test : list of polymers (if input_transformer defined) or array-like of shape (n_samples, n_features)
            Data to be evaluated by the GPR model.
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
        X_tokens = self._pretained_model.tokenize(X_test)

        # Convert to torch tensors if necessary
        if not torch.is_tensor(X_tokens):
            X_tokens = torch.from_numpy(np.asarray(X_tokens)).float()
        if y_noise is not None:
            y_noise = torch.from_numpy(y_noise).float()

        # Move tensors to device
        X_tokens = X_tokens.to(self._device)
        if y_noise is not None:
            y_noise = y_noise.to(self._device)

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            if y_noise is None:
                predictions = self._likelihood(self._model(X_tokens))
            else:
                predictions = self._likelihood(self._model(X_tokens), noise=y_noise)

        mu = predictions.mean.detach().cpu().numpy()
        sigma = predictions.stddev.detach().cpu().numpy()

        return mu, sigma
