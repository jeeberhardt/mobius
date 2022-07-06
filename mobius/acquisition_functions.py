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

    @property
    @abstractmethod
    def surrogate_model(self):
        pass

    @property
    @abstractmethod
    def goal(self):
        pass

    @property
    @abstractmethod
    def greater_is_better(self):
        pass

    @abstractmethod
    def forward(self):
        raise NotImplementedError()


class RandomImprovement(_AcquisitionFunction):

    def __init__(self, goal='maximize'):
        """
        Random acquisition function

        Arguments:
        ----------
            goal: Indicates whether the function is to be maximised or minimised.

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'
        
        self._goal = goal

        if self._goal == 'minimize':
            self._greater_is_better = False
        else:
            self._greater_is_better = True

    @property
    def surrogate_model(self):
        raise NotImplementedError()

    @property
    def goal(self):
        return self._goal

    @property
    def greater_is_better(self):
        return self._greater_is_better

    def forward(self, X_test):
        X_test = np.asarray(X_test)
        mu = np.random.uniform(low=0, high=1, size=X_test.shape[0])

        return mu


class Greedy(_AcquisitionFunction):

    def __init__(self, surrogate_model, goal='maximize'):
        """
        Greedy acquisition function

        Arguments:
        ----------
            surrogate_model: Surrogate model
            goal: Indicates whether the function is to be maximised or minimised (needed for API compatibility).

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._goal = goal
        self._surrogate_model = surrogate_model

        if self._goal == 'minimize':
            self._greater_is_better = False
        else:
            self._greater_is_better = True

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def goal(self):
        return self._goal

    @property
    def greater_is_better(self):
        return self._greater_is_better

    def forward(self, X_test):
        mu, _ = self._surrogate_model.predict(X_test)

        return mu


class ExpectedImprovement(_AcquisitionFunction):

    def __init__(self, surrogate_model, goal='maximize', xi=0.00):
        """
        Expected improvement acquisition function.
    
        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
        
        Arguments:
        ----------
            surrogate_model: Surrogate model
            goal: Indicates whether the function is to be maximised or minimised.
            xi: Exploitation-exploration trade-off parameter

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._goal = goal
        self._surrogate_model = surrogate_model
        self._xi = xi

        if self._goal == 'minimize':
            self._greater_is_better = False
        else:
            self._greater_is_better = True

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self._greater_is_better)

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def goal(self):
        return self._goal

    @property
    def greater_is_better(self):
        return self._greater_is_better

    def forward(self, X_test):
        if self._greater_is_better:
            best_f = np.max(self._surrogate_model.y_train)
        else:
            best_f = np.min(self._surrogate_model.y_train)

        # calculate mean and stdev via surrogate function*
        mu, sigma = self._surrogate_model.predict(X_test)

        # calculate the expected improvement
        Z = self._scaling_factor * (mu - best_f - self._xi) / (sigma + 1E-9)
        ei = self._scaling_factor * (mu - best_f) * norm.cdf(Z) + (sigma * norm.pdf(Z))
        ei[sigma == 0.0] == 0.0

        return -1 * ei


class ProbabilityOfImprovement(_AcquisitionFunction):

    def __init__(self, surrogate_model, goal='maximize'):
        """
        Expected improvement acquisition function.
    
        Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
        
        Arguments:
        ----------
            surrogate_model: Surrogate model
            goal: Indicates whether the function is to be maximised or minimised.

        """
        assert goal in ['minimize', 'maximize'], 'The goal can only be \'minimize\' or \'maximize\'.'

        self._goal = goal
        self._surrogate_model = surrogate_model

        if self._goal == 'minimize':
            self._greater_is_better = False
        else:
            self._greater_is_better = True

        # goal = maximize // greater_is_better = True -> scaling_factor = -1
        # goal = minimize // greater_is_better = False -> scaling_factor = 1
        self._scaling_factor = (-1) ** (self._greater_is_better)

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def goal(self):
        return self._goal

    @property
    def greater_is_better(self):
        return self._greater_is_better

    def forward(self, X_test):
        if self._greater_is_better:
            best_f = np.max(self._surrogate_model.y_train)
        else:
            best_f = np.min(self._surrogate_model.y_train)

        # calculate mean and stdev via surrogate function
        mu, sigma = self._surrogate_model.predict(X_test)

        # calculate the probability of improvement
        Z = self._scaling_factor * (mu - best_f) / (sigma + 1E-9)
        pi = norm.cdf(Z)
        pi[sigma == 0.0] == 0.0

        return pi
