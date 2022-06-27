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


@ray.remote
def parallel_acq(acquisition_function, X_test):
    return acquisition_function.forward(X_test)


class _AcquisitionFunction(ABC):

    @abstractmethod
    def forward(self):
        raise NotImplementedError()


class RandomImprovement(_AcquisitionFunction):

    def __init__(self, model, y_train, goal='maximize'):
        """
        Random acquisition function

        Arguments:
        ----------
            model: Gaussian process model (needed for API compatibility)
            y_train: Array that contains all the observed energy interaction seed so far (needed for API compatibility)
            goal: Indicates whether the function is to be maximised or minimised.

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        if goal == 'minimize':
            self.greater_is_better = False
        else:
            self.greater_is_better = True

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self.greater_is_better)

    def forward(self, X_test):
        X_test = np.array(X_test)
        mu = self._scaling_factor * np.random.uniform(low=0, high=1, size=X_test.shape[0])

        return mu


class Greedy(_AcquisitionFunction):

    def __init__(self, model, y_train, goal='maximize', xi=0.00):
        """
        Greedy acquisition function

        Arguments:
        ----------
            model: Gaussian process model
            y_train: Array that contains all the observed energy interaction seed so far (needed for API compatibility).
            goal: Indicates whether the function is to be maximised or minimised (needed for API compatibility).

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._model = model
        self._xi = xi

        if goal == 'minimize':
            self.greater_is_better = False
            self._best_f = np.min(y_train)
        else:
            self.greater_is_better = True
            self._best_f = np.max(y_train)

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self.greater_is_better)

    def forward(self, X_test):
        mu, _ = self._model.predict(X_test)

        return mu


class ExpectedImprovement(_AcquisitionFunction):

    def __init__(self, model, y_train, goal='maximize', xi=0.00):
        """
        Expected improvement acquisition function.
    
        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
        
        Arguments:
        ----------
            model: Surrogate model
            y_train: Array that contains all the observed energy interaction seed so far
            goal: Indicates whether the function is to be maximised or minimised.
            xi: Exploitation-exploration trade-off parameter

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._model = model
        self._xi = xi

        if goal == 'minimize':
            self.greater_is_better = False
            self._best_f = np.min(y_train)
        else:
            self.greater_is_better = True
            self._best_f = np.max(y_train)

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self.greater_is_better)

    def forward(self, X_test):
        # calculate mean and stdev via surrogate function*
        mu, sigma = self._model.predict(X_test)

        # calculate the expected improvement
        Z = self._scaling_factor * (mu - self._best_f - self._xi) / (sigma + 1E-9)
        ei = self._scaling_factor * (mu - self._best_f) * norm.cdf(Z) + (sigma * norm.pdf(Z))
        ei[sigma == 0.0] == 0.0

        return -1 * ei


class ProbabilityOfImprovement(_AcquisitionFunction):

    def __init__(self, model, y_train, goal='maximize'):
        """
        Expected improvement acquisition function.
    
        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
        
        Arguments:
        ----------
            model: Surrogate model
            y_train: Array that contains all the observed energy interaction seed so far
            goal: Indicates whether the function is to be maximised or minimised.

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._model = model
        self._xi = xi

        if goal == 'minimize':
            self.greater_is_better = False
            self._best_f = np.min(y_train)
        else:
            self.greater_is_better = True
            self._best_f = np.max(y_train)

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self.greater_is_better)

    def forward(self, X_test):
        # calculate mean and stdev via surrogate function
        mu, sigma = self._model.predict(X_test)

        # calculate the probability of improvement
        Z = scaling_factor * (mu - self._best_f) / (sigma + 1E-9)
        pi = norm.cdf(Z)
        pi[sigma == 0.0] == 0.0

        return pi
