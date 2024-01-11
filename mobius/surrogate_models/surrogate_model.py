#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Surrogate Model
#

from abc import ABC, abstractmethod

from sklearn.metrics import r2_score
from sklearn.exceptions import NotFittedError


class _SurrogateModel(ABC):

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

    @abstractmethod
    def fit(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self):
        raise NotImplementedError()

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
