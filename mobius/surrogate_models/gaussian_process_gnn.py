#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor with Grahp Neural Network (GNN) models
#

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from sklearn.exceptions import NotFittedError

from .surrogate_model import _SurrogateModel
from .gaussian_process_graph import ExactGPGraph
from ..utils import ProgressBar


# We will use the simplest form of GP model, exact inference
class _ExactGPGNNModel(ExactGPGraph):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel, feature_extractor):
        super(_ExactGPGNNModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)
        self.feature_extractor = feature_extractor

    def forward(self, x):
        x = self.feature_extractor.forward(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class GPGNNModel(_SurrogateModel):
    """
    Class for the Gaussian Process Regressor (GPR) surrogate model for Graph Neural Network (GNN) models.

    """

    def __init__(self, kernel, model, transform, noise_prior=None, missing_values=False, device=None, show_progression=True):
        """
        Initializes the Gaussian Process Regressor (GPR) surrogate model for Graph Neural Network (GNN) models.

        Parameters
        ----------
        kernel : `grakel.kernels.Kernel`
            The kernel function used by the GPR model.
        model : `torch.nn.Module` 
            Graph Neural Network (GNN) model that transforms graphs into data exploitable by the GP model.
        transform : callable
            Function that transforms the inputs into graphs.
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
        self._transform = transform
        self._noise_prior = noise_prior
        self._missing_values = missing_values
        self._feature_extractor = model
        self._device = device
        self._show_progression = show_progression
        self._model = None
        self._likelihood = None
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

         # Transform input data
        X_train = self._transform.transform(self._X_train)

        # X_train cannot be transformed into a tensor
        # Convert to double, otherwise we get an error.
        # return torch.cholesky_solve(rhs, self.to_dense(), upper=upper)
        # RuntimeError: Expected b and A to have the same dtype, but found b of type Float and A of type Double instead.
        y_train = torch.from_numpy(self._y_train).float()
        if y_noise is not None:
            y_noise = torch.from_numpy(self._y_noise).float()

        # Move data to device
        X_train = X_train.to(self._device)
        y_train = y_train.to(self._device)
        if y_noise is not None:
            y_noise = y_noise.to(self._device)

        if self._missing_values:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=self._noise_prior)
        elif y_noise is not None:
            self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=y_noise, learn_additional_noise=True)
        else:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=self._noise_prior)

        self._model = _ExactGPGNNModel(X_train, y_train, self._likelihood, self._kernel, self._feature_extractor)

        # Move GP, GNN (inside GP) and likelihood to device
        self._model.feature_extractor.to(self._device)
        self._model.to(self._device)
        self._likelihood.to(self._device)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Set model in training mode
        self._model.feature_extractor.train()
        self._model.train()
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
        self._model.feature_extractor.eval()
        self._model.eval()
        self._likelihood.eval()

        # Transform input data
        X_test = self._transform.transform(X_test)

        # X_test cannot be transformed into a tensor
        if y_noise is not None:
            y_noise = torch.from_numpy(y_noise).float()

        # Move tensors to device
        X_test = X_test.to(self._device)
        if y_noise is not None:
            y_noise = y_noise.to(self._device)

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            if y_noise is None:
                predictions = self._likelihood(self._model(X_test))
            else:
                predictions = self._likelihood(self._model(X_test), noise=y_noise)

        mu = predictions.mean.detach().cpu().numpy()
        sigma = predictions.stddev.detach().cpu().numpy()

        return mu, sigma
