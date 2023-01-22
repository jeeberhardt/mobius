#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Acquisition function
#

from abc import ABC, abstractmethod

import numpy as np
import torch
import ray
from scipy.stats import norm

from .surrogate_model import DummyModel


@ray.remote
def parallel_acq(acquisition_function, X_test):
    return acquisition_function.forward(X_test)


class _AcquisitionFunction(ABC):

    @property
    @abstractmethod
    def surrogate_model(self):
        pass

    @property
    @abstractmethod
    def maximize(self):
        pass

    @property
    @abstractmethod
    def scaling_factor(self):
        pass

    @abstractmethod
    def forward(self):
        raise NotImplementedError()


class RandomImprovement(_AcquisitionFunction):

    def __init__(self, maximize=True):
        """
        Random acquisition function

        Arguments:
        ----------
            maximize: Indicates whether the function is to be maximised (default: False).

        """
        self._surrogate_model = DummyModel()
        self._maximize = maximize

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factory(self):
        if self.maximize:
            return -1
        else:
            return 1

    def forward(self, X_test):
        X_test = np.asarray(X_test)
        mu = np.random.uniform(low=0, high=1, size=X_test.shape[0])

        return mu


class Greedy(_AcquisitionFunction):

    def __init__(self, surrogate_model, maximize=False):
        """
        Greedy acquisition function

        Arguments:
        ----------
            surrogate_model: Surrogate model
            maximize: Indicates whether the function is to be maximised (default: False).

        """
        self._surrogate_model = surrogate_model
        self._maximize = maximize

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factor(self):
        if self._maximize:
            return -1
        else:
            return 1

    def forward(self, X_test):
        mu, _ = self._surrogate_model.predict(X_test)

        #scaling_factor = (-1) ** (not self._maximize)

        return mu


class ExpectedImprovement(_AcquisitionFunction):

    def __init__(self, surrogate_model, maximize=False, xi=0.00, eps=1e-9):
        """
        Expected improvement acquisition function.

        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

        Arguments:
        ----------
            surrogate_model: Surrogate model
            maximize: Indicates whether the function is to be maximised (default: False).
            xi: Exploitation-exploration trade-off parameter (default: 0.0)

        """
        self._surrogate_model = surrogate_model
        self._xi = xi
        self._eps = eps
        self._maximize = maximize

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factor(self):
        return -1

    def forward(self, X_test):
        if self._maximize:
            best_f = np.max(self._surrogate_model.y_train)
        else:
            best_f = np.min(self._surrogate_model.y_train)

        # calculate mean and stdev via surrogate function*
        mu, sigma = self._surrogate_model.predict(X_test)

        u = (mu - best_f - self._xi) / (sigma + self._eps)
        if not self.maximize:
            u = -u
        ucdf = norm.cdf(u)
        updf = norm.pdf(u)
        ei = sigma * (updf + u * ucdf)
        ei[sigma == 0.0] == 0.0

        return ei


class ProbabilityOfImprovement(_AcquisitionFunction):

    def __init__(self, surrogate_model, maximize=False, eps=1e-9):
        """
        Expected improvement acquisition function.

        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

        Arguments:
        ----------
            surrogate_model: Surrogate model
            maximize: Indicates whether the function is to be maximised (default: False).

        """
        self._surrogate_model = surrogate_model
        self._maximize = maximize
        self._eps = eps

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factor(self):
        return -1

    def forward(self, X_test):
        if self._maximize:
            best_f = np.max(self._surrogate_model.y_train)
        else:
            best_f = np.min(self._surrogate_model.y_train)

        # calculate mean and stdev via surrogate function
        mu, sigma = self._surrogate_model.predict(X_test)

        # calculate the probability of improvement
        u = (mu - best_f) / (sigma + self._eps)
        if not self.maximize:
            u = -u
        pi = norm.cdf(u)
        pi[sigma == 0.0] == 0.0

        return pi
