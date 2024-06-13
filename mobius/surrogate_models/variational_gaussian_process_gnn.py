#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Variational Gaussian Process Regressor with GNN
#

import gpytorch
import numpy as np
import torch
from sklearn.exceptions import NotFittedError
from torch.utils.data import Dataset, DataLoader

from .surrogate_model import _SurrogateModel


class PolymerDataset(Dataset):
    def __init__(self, polymers, values):
        if not isinstance(polymers, (list, tuple, np.ndarray)):
            polymers = [polymers]

        if not isinstance(values, (list, tuple, np.ndarray)):
            values = [values]

        if len(polymers) != len(values):
            raise ValueError("Polymers and values must have the same length")

        self._polymers = polymers
        self._values = values

    def __len__(self):
        return len(self._polymers)

    def __getitem__(self, idx):
        return self._polymers[idx], self._values[idx]

    
class _ApproximateGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, kernel, inducing_points):
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        

class VGPGNNModel(_SurrogateModel):
    """
    Class for the Variational Gaussian Process Regressor (VGPR) surrogate model for Graphs.

    """

    def __init__(self, kernel, model, transform, device=None):
        """
        Initializes the Variational Gaussian Process Regressor (GPR) surrogate model for Graphs.

        Parameters
        ----------
        kernel : `grakel.kernels.Kernel`
            The kernel specifying the covariance function of the GPR model.
        model : `torch.nn.Module` 
             Graph Neural Network (GNN) model that transforms inputs into data exploitable by the GP model.
        transform : callable, default : None
            Function that transforms the input into a graph features for the GNN model.
        device : str or torch.device, default : None
            Device on which to run the GNN and GP models. Per default, the device is set to 
            'cuda' if available, otherwise to 'cpu'.

        """
        self._kernel = kernel
        self._transform = transform
        self._feature_extractor = model
        self._model = None
        self._device = device
        self._likelihood = None
        self._X_train = None
        self._y_train = None
        self._X_validation = None
        self._y_validation = None

    @property
    def device(self):
        """Returns the device on which the model is running."""
        return self._device

    def fit(self, X_train, y_train, X_validation, y_validation, batch_size=100, n_epochs=5, n_inducing_points=50):
        """
        Fits the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_train : list of polymers (if transformer defined) or array-like of shape (n_samples, n_features)
            Data to be used for training the GPR model.
        y_train : array-like of shape (n_samples,)
            Target values.

        """
        self._X_train = np.asarray(X_train).copy()
        self._y_train = np.asarray(y_train).copy()
        self._X_validation = np.asarray(X_validation).copy()
        self._y_validation = np.asarray(y_validation).copy()

        # Check that the number of polymers in (X_train, y_train) and (X_validation, y_validation)
        msg_error = "The number of sequences in X_train and values in y_train must be the same."
        assert self._X_train.shape[0] == self._y_train.shape[0], msg_error
        msg_error = "The number of sequences in X_validation and values in y_validation must be the same."
        assert self._X_validation.shape[0] == self._y_validation.shape[0], msg_error

        # Create DataLoader with input data
        dataset = PolymerDataset(self._X_train, self._y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True)

        # Transform validation set in graphs, and convert y_validation to tensor
        x_validation_graphs = self._transform.transform(self._X_validation)
        y_validation = torch.from_numpy(self._y_validation).float()

        # Initialize GP model with some inducing points
        graphs = self._transform.transform(dataset[:n_inducing_points][0])
        initial_inducing_points = self._feature_extractor.forward(graphs)
        self._model = _ApproximateGPModel(self._kernel, initial_inducing_points)

        noise_prior = gpytorch.priors.NormalPrior(loc=0, scale=1)
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior)
        mll = gpytorch.mlls.VariationalELBO(self._likelihood, self._model, num_data=len(dataset))

        optimizer = torch.optim.Adam(
            [
                {"params": self._feature_extractor.parameters()},
                {"params": self._model.parameters()},
                {"params": self._likelihood.parameters()}
            ],
        lr=0.01)

        # Set models in training mode
        self._feature_extractor.train()
        self._model.train()
        self._likelihood.train()

        # Train model!
        for i in range(n_epochs):
            for x_train_batch, y_train_batch in dataloader:
                optimizer.zero_grad()
    
                # Transform input molecules in graph and then embeddings
                x_train_graphs = self._transform.transform(x_train_batch)
                x = self._feature_extractor.forward(x_train_graphs)
                    
                output = self._model(x)
                loss = -mll(output, y_train_batch)
                print('Training loss: ', loss)
    
                loss.backward()
                optimizer.step()
    
                with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
                    x = self._feature_extractor.forward(x_validation_graphs)
                    predictions = self._likelihood(self._model(x))
    
                    loss = -mll(predictions, y_validation)
                    print('Validation loss: ', loss)
    
                print("")

        self._feature_extractor.eval()
        self._model.eval()
        self._likelihood.eval()

    def predict(self, X_test):
        """
        Predicts using the Gaussian Process Regressor (GPR) model.

        Parameters
        ----------
        X_test : list of polymers (if input_transformer defined) or array-like of shape (n_samples, n_features)
            Data to be evaluated by the GPR model.

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
        if self._model is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        # Set model in evaluation mode
        self._feature_extractor.eval()
        self._model.eval()
        self._likelihood.eval()

        # Transform input data if necessary
        X_test_graphs = self._transform.transform(X_test)

        # Make predictions by feeding model through likelihood
        # Set fast_pred_var state to False, otherwise cannot pickle GPModel
        with torch.no_grad(), gpytorch.settings.fast_pred_var(state=False):
            x = self._feature_extractor.forward(X_test_graphs)
            predictions = self._likelihood(self._model(x))

        mu = predictions.mean.detach().cpu().numpy()
        sigma = predictions.stddev.detach().cpu().numpy()

        return mu, sigma
