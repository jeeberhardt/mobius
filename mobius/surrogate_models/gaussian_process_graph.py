#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Gaussian Process Regressor with Graphs
#

import copy

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.utils.generic import length_safe_zip
from sklearn.exceptions import NotFittedError
from torch_geometric.data import Batch

from .surrogate_model import _SurrogateModel
from ..kernels import GraphKernel


class ExactGPGraph(ExactGP):
    """
    Unsafe version of ExactGP to handle graphs.
    """

    def __init__(self, train_inputs, train_targets, likelihood):
        # We do check the inputs here, not safe.
        train_inputs = (train_inputs,)
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        super(ExactGP, self).__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if torch.is_tensor(tri) and tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if torch.is_tensor(i) and i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            res = super(ExactGP, self).__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = super(ExactGP, self).__call__(*train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            if torch.is_tensor(train_inputs[0]):
                batch_shape = train_inputs[0].shape[:-2]
                for train_input, input in length_safe_zip(train_inputs, inputs):
                    # Make sure the batch shapes agree for training/test data
                    if batch_shape != train_input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                        train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    if batch_shape != input.shape[:-2]:
                        batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                        train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                        input = input.expand(*batch_shape, *input.shape[-2:])
                    full_inputs.append(torch.cat([train_input, input], dim=-2))
            else:
                full_inputs = copy.deepcopy(train_inputs)

                if isinstance(full_inputs[0], np.ndarray) and isinstance(inputs[0], np.ndarray):
                    full_inputs[0] = np.concatenate((full_inputs[0], inputs[0]))
                elif isinstance(full_inputs[0], list) and isinstance(inputs[0], list):
                    full_inputs[0] = full_inputs[0] + inputs[0]
                elif isinstance(full_inputs[0], Batch) and isinstance(inputs[0], Batch):
                    full_inputs[0] = full_inputs[0].to_data_list() + inputs[0].to_data_list()
                    full_inputs[0] = Batch.from_data_list(full_inputs[0])
                else:
                    raise RuntimeError(f"Unsupported input types for ExactGPGraph: {type(full_inputs[0])} and {type(inputs[0])})")

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                (
                    predictive_mean,
                    predictive_covar,
                ) = self.prediction_strategy.exact_prediction(full_mean, full_covar)

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)


# We will use the simplest form of GP model, exact inference
class _ExactGPModel(ExactGPGraph):
    # to inform GPyTorchModel API
    _num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, kernel):
        super(_ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(torch.zeros(len(x.data), 1)).float()
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGModel(_SurrogateModel):
    """
    Class for the Gaussian Process Regressor (GPR) surrogate model for Graphs.

    """

    def __init__(self, kernel, input_transformer=None, missing_values=False):
        """
        Initializes the Gaussian Process Regressor (GPR) surrogate model for Graphs.

        Parameters
        ----------
        kernel : `grakel.kernels.Kernel`
            The kernel specifying the covariance function of the GPR model.
        input_transformer : input transformer, default : None
            Function that transforms the input into a graph exploitable by the GP model.
        missing_values : bool, default : False
            Whether we support missing values in the input data.

        """
        self._kernel = GraphKernel(kernel=kernel)
        self._transformer = input_transformer
        self._missing_values = missing_values
        self._model = None
        self._likelihood = None
        self._train_x = None
        self._train_y = None

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
            Noise value associated to each target value (y_train), and expressed as 
            standard deviation (sigma). Values are squared internally to obtain the variance.

        """
        # Make sure that inputs are numpy arrays, keep a persistant copy
        self._X_train = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()
        if y_noise is not None:
            self._y_noise = np.asarray(y_noise).copy()
            self._y_noise = self._y_noise**2

        # Check that the number of polymers in X_train, y_train and y_noise are the same
        msg_error = "The number of sequences in X_train and values in y_train must be the same."
        assert self._X_train.shape[0] == self._y_train.shape[0], msg_error
        if y_noise is not None:
            msg_error = "The number of sequences in X_train and values in y_noise must be the same."
            assert self._X_train.shape[0] == self._y_noise.shape[0], msg_error

         # Transform input data if necessary
        if self._transformer is not None:
            X_train = self._transformer.transform(self._X_train)
        else:
            X_train = self._X_train

        # X_train cannot be transformed into a tensor
        # Convert to double, otherwise we get an error.
        # return torch.cholesky_solve(rhs, self.to_dense(), upper=upper)
        # RuntimeError: Expected b and A to have the same dtype, but found b of type Float and A of type Double instead.
        y_train = torch.from_numpy(self._y_train).double()
        if y_noise is not None:
            y_noise = torch.from_numpy(self._y_noise).float()

        noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)

        if self._missing_values:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihoodWithMissingObs(noise_prior=noise_prior)
        elif y_noise is not None:
            self._likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=y_noise, learn_additional_noise=True)
        else:
            self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)

        self._model = _ExactGPModel(X_train, y_train, self._likelihood, self._kernel)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, self._model)

        # Set model in training mode
        self._model.train()
        self._likelihood.train()

        # Train model!
        fit_gpytorch_mll(mll)

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
        self._likelihood.eval()

        # Transform input data if necessary
        if self._transformer is not None:
            X_test = self._transformer.transform(X_test)

        # X_test cannot be transformed into a tensor
        if y_noise is not None:
            y_noise = torch.from_numpy(y_noise).float()

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
