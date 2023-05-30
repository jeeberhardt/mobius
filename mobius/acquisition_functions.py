#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Acquisition function
#

from abc import ABC, abstractmethod

import numpy as np
import ray
from scipy.stats import norm

from .surrogate_models import DummyModel


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
    """
    Class for the random improvement acquisition function. This acquisition
    function is used for benchmarking purposes only. It can be used to
    simulate a random search.

    Attributes
    ----------
    surrogate_model : `_SurrogateModel`
        Surrogate model used by the acquisition function (`DummyModel`).
    maximize : bool
        Tell if the goal to maximize (True) or minimize (False) the acquisition function.
    scaling_factor : int
        Scaling factor used by the Bolzmann weigthing function in the GA

    """

    def __init__(self, low=0, high=1, maximize=True):
        """
        Random acquisition function.

        Parameters
        ----------
        low : int, default : 0
            Lower boundary of the output interval. All values generated 
            will be greater than or equal to low.
        high : int, default : 1
            Upper boundary of the output interval. All values generated 
            will be less than or equal to high.
        maximize : bool, default : True
            Indicates whether the function is to be maximised.

        """
        self._surrogate_model = DummyModel()
        self._low = low
        self._high = high
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
        """
        Predict random mean values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        mu : ndarray of shape (n_samples, )
            Random value for each input sample.

        """
        X_test = np.asarray(X_test)
        mu = np.random.uniform(low=self._low, high=self._high, size=X_test.shape[0])

        return mu


class Greedy(_AcquisitionFunction):
    """
    Class for the Greedy acquisition function.

    Attributes
    ----------
    surrogate_model : `_SurrogateModel`
        The surrogate model used by the acquisition function.
    maximize : bool
        Tell if the goal to maximize (True) or minimize (False) 
        the acquisition function.
    scaling_factor : int
        Scaling factor used by the Bolzmann weigthing function in the GA.

    """

    def __init__(self, surrogate_model, maximize=False):
        """
        Greedy acquisition function.

        Parameters
        ----------
        surrogate_model: `_SurrogateModel`
            The surrogate model to be used by the acquisition function.
        maximize : bool, default : False
            Indicates whether the function is to be maximised.

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
        """
        Predict mean values by the surrogate model for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        mu : ndarray of shape (n_samples, )
            The mean values predicted by the surrogate model for each input sample.

        """
        mu, _ = self._surrogate_model.predict(X_test)

        return mu


class ExpectedImprovement(_AcquisitionFunction):
    """
    Class for the Expected Improvement acquisition function.

    Attributes
    ----------
    surrogate_model : `_SurrogateModel`
        The surrogate model used by the acquisition function.
    maximize : bool
        Tell if the goal to maximize (True) or minimize (False) the acquisition function.
    scaling_factor : int
        Scaling factor used by the Bolzmann weigthing function in the GA.

    """

    def __init__(self, surrogate_model, maximize=False, xi=0.00, eps=1e-9):
        """
        Expected improvement acquisition function.

        Parameters
        ----------
        surrogate_model: `_SurrogateModel`
            The surrogate model to be used by the acquisition function.
        maximize : bool, default : False
            Indicates whether the function is to be maximised.
        xi : float, default : 0.
            Exploitation-exploration trade-off parameter.
        eps : float, default : 1e-9
            Small number to avoid numerical instability.

        Notes
        -----
        https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

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
        """
        Predict the Expected Improvement values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ei : ndarray of shape (n_samples, )
            Predicted Expected Improvement value for each input sample.

        """
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
    """
    Class for the Probability of Improvement acquisition function.

    Attributes
    ----------
    surrogate_model : `_SurrogateModel`
        The surrogate model used by the acquisition function.
    maximize : bool
        Tell if the goal to maximize (True) or minimize (False) 
        the acquisition function.
    scaling_factor : int
        Scaling factor used by the Bolzmann weigthing function in the GA.

    """

    def __init__(self, surrogate_model, maximize=False, eps=1e-9):
        """
        Probability of Improvement acquisition function.

        Parameters
        ----------
        surrogate_model: `_SurrogateModel`
            The surrogate model to be used by the acquisition function.
        maximize : bool, default : False
            Indicates whether the function is to be maximised.
        eps : float, default : 1e-9
            Small number to avoid numerical instability.

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
        """
        Predict the Probability of Improvement values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        pi : ndarray of shape (n_samples, )
            Predicted Probability of Improvement value for each input sample.

        """
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


class UpperConfidenceBound(_AcquisitionFunction):
    """
    Class for the Probability of Improvement acquisition function.

    Attributes
    ----------
    surrogate_model : `_SurrogateModel`
        The surrogate model used by the acquisition function.
    maximize : bool
        Tell if the goal to maximize (True) or minimize (False) 
        the acquisition function.
    scaling_factor : int
        Scaling factor used by the Bolzmann weigthing function in the GA.

    """

    def __init__(self, surrogate_model, maximize=False, beta=0.2):
        """
        Upper Confidence Bound acquisition function.

        Parameters
        ----------
        surrogate_model: `_SurrogateModel`
            The surrogate model to be used by the acquisition function.
        maximize : bool, default : False
            Indicates whether the function is to be maximised.
        kappa : float, default : 2.576
            Exploitation-exploration trade-off parameter.

        """
        self._surrogate_model = surrogate_model
        self._maximize = maximize
        self._delta = 1. - beta
        self._beta = beta

    @property
    def surrogate_model(self):
        return self._surrogate_model
    
    @property
    def maximize(self):
        return self._maximize
    
    @property
    def scaling_factor(self):
        if self._maximize:
            return 1
        else:
            return -1
    
    def forward(self, X_test):
        """
        Predict the Upper Confidence Bound values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        ucb : ndarray of shape (n_samples, )
            Predicted Upper Confidence Bound value for each input sample.

        """
        # calculate mean and stdev via surrogate function
        mu, sigma = self._surrogate_model.predict(X_test)

        # calculate the upper confidence bound
        ucb = -(self._delta * mu) - (self._beta * sigma)

        return ucb
