#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Random Forest Regressor
#

from abc import ABC, abstractmethod

import numpy as np

from sklearn.ensemble import RandomForestRegressor
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


class RFModel(_SurrogateModel):
    """
    Class for the Random Forest Regressor (RFR) surrogate model.

    """
    
    def __init__(self, input_transformer=None, **kwargs):
        """
        Initialization of the Random Forest Regressor (RFR) surrogate model.

        Parameters
        ----------
        input_transformer : input_transformer, default : None
            Function that transforms the input into data exploitable by the RFR model.
        **kwargs
            All the other keyword arguments are passed on to the internal `RFR` 
            model from the scikit-learn package. The default parameters are
            `n_estimators=500`, `max_features='sqrt'`, `max_depth=None`, 
            `oob_Score=True`, `bootstrap=True` and `max_samples=None`.

        """
        self._input_transformer = input_transformer
        self._model = None
        self._X_train = None
        self._y_train = None
        self._kwargs = kwargs

        # Set default parameters for RF
        self._kwargs.setdefault('n_estimators', 500)
        self._kwargs.setdefault('max_features', 'sqrt')
        self._kwargs.setdefault('max_depth', None)
        self._kwargs.setdefault('oob_score', True)
        self._kwargs.setdefault('bootstrap', True)
        self._kwargs.setdefault('max_samples', None)

    @property
    def X_train(self):
        """
        Returns the training dataset obtained after transformation.

        Returns
        -------
        X_train : array-like of shape (n_samples, n_features)
            The training dataset obtained after transformation.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        """
        if self._X_train is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        return self._X_train

    @property
    def y_train(self):
        """
        Returns the target values.

        Returns
        -------
        y_train : array-like of shape (n_smaples, )
            The target values.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        """
        if self._y_train is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        return self._y_train

    def fit(self, X_train, y_train):
        """
        Fit Random Forest Regressor (RFR) model.

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

        if self._input_transformer is not None:
            # Transform input data
            self._X_train = self._input_transformer.transform(self._X_train)

        self._model = RandomForestRegressor(**self._kwargs)
        self._model.fit(self._X_train, self._y_train)

    def predict(self, X_test):
        """
        Predict using the Random Forest Regressor (RFR) model.

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
        NotFittedError
            If the model instance is not fitted yet.

        Notes
        -----
        The uncertainty is estimated by calcutating the standard deviations from the
        predictions of each individual trees in the RFR model. It should not be done 
        this way: https://stats.stackexchange.com/questions/490514

        """
        if self._model is None:
            msg = 'This model instance is not fitted yet. Call \'fit\' with appropriate arguments before using this estimator.'
            raise NotFittedError(msg)

        if self._input_transformer is not None:
            # Transform input data
            X_test = self._input_transformer.transform(X_test)

        mu = self._model.predict(X_test)
        estimations = np.stack([t.predict(X_test) for t in self._model.estimators_])
        sigma = np.std(estimations, axis=0)

        return mu, sigma

    def score(self, X_test, y_test):
        """
        Returns the coefficient of determination :math:`R^2`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Query points where the GPR is evaluated.
        y_test : array-like of shape (n_samples,)
            True values of `X_test`.

        Returns
        -------
        score : float
            Coefficient of determination :math:`R^2`.

        Raises
        ------
        RuntimeError
            If the instance is not fitted yet.

        """
        mu, _ = self.predict(X_test)
        return r2_score(y_test, mu)
