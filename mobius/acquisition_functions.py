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
    def scaling_factors(self):
        pass

    @property
    @abstractmethod
    def number_of_objectives(self):
        pass

    @abstractmethod
    def fit(self):
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
    surrogate_model : list of `_SurrogateModel` (`DummyModel`)
        The surrogate models used by the acquisition function.
    scaling_factors : ndarray of shape (surrogate_models,)
        Scaling factor used by Genetic Algorithm, so it's always a minimization.
    maximize : ndarray of bool of shape (surrogate_models,)
        Tell if the goal(s) is to maximize (True) or minimize (False) the acquisition function.
    number_of_objectives : int
        Number of objectives to optimize.

    """

    def __init__(self, low=0, high=1, maximize=False):
        """
        Random acquisition function.

        Parameters
        ----------
        low : int or array-like of shape (n_objectives, ), default : 0
            Lower boundary of the output interval. All values generated 
            will be greater than or equal to low.
        high : int or array-like of shape (n_objectives, ), default : 1
            Upper boundary of the output interval. All values generated 
            will be less than or equal to high.
        maximize : bool or array-like of bool of shape (n_objectives, ), default : False
            Indicates whether the goal(s) is(are) to be maximised or minimized.

        """
        if not isinstance(low, (list, tuple, np.ndarray)):
            low = [low]
        if not isinstance(high, (list, tuple, np.ndarray)):
            high = [high]
        if not isinstance(maximize, (list, tuple, np.ndarray)):
            maximize = [maximize]

        msg_error = "The number of lower and upper bounds must be the same."
        assert len(low) == len(high), msg_error
        msg_error = "The number of maximize flags must be the same as the "
        msg_error += "number of low/high bounds."
        assert len(maximize) == len(low), msg_error

        self._surrogate_model = DummyModel()
        self._low = np.asarray(low)
        self._high = np.asarray(high)
        self._maximize = np.asarray(maximize)

    @property
    def surrogate_model(self):
        return self._surrogate_model

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factors(self):
        return -1 * np.ones(len(self._maximize))
    
    @property
    def number_of_objectives(self):
        return len(self._maximize)
    
    def fit(self, X_train, y_train):
        """
        Fit the surrogate models using existing data.

        Parameters
        ----------
        X_train : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.
        y_train : array-like of shape (n_polymers, n_objectives)
            Values associated to each polymer.

        """
        pass

    def forward(self, X_test):
        """
        Predict random mean values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.

        Returns
        -------
        ndarray of shape (n_polymers, n_objectives)
            The expected improvement values.

        """
        X_test = np.asarray(X_test)
        size = (X_test.shape[0], len(self._low))
        ri = np.random.uniform(low=self._low, high=self._high, size=size)

        print(ri.shape)

        return ri


class Greedy(_AcquisitionFunction):
    """
    Class for the Greedy acquisition function.

    Attributes
    ----------
    surrogate_model : list of `_SurrogateModel`
        The surrogate models used by the acquisition function.
    maximize : ndarray of bool of shape (surrogate_models,)
        Tell if the goal(s) is to maximize (True) or minimize (False) the acquisition function.
    scaling_factors : ndarray of shape (surrogate_models,)
        Scaling factor used by Genetic Algorithm, so it's always a minimization.

    """

    def __init__(self, surrogate_models, maximize=False):
        """
        Greedy acquisition function.

        Parameters
        ----------
        surrogate_models: `_SurrogateModel` or list of `_SurrogateModel`
            The surrogate model(s) to be used by the acquisition function.
        maximize : bool or list of bool, default : False
            Indicates whether the goal(s) is(are) to be maximised or minimized.

        """
        if not isinstance(surrogate_models, (list, tuple, np.ndarray)):
            surrogate_models = [surrogate_models]
        if not isinstance(maximize, (list, tuple, np.ndarray)):
            maximize = [maximize]
        
        msg_error = "The number of surrogate models and maximize flags must be the same."
        assert len(surrogate_models) == len(maximize), msg_error

        self._surrogate_models = surrogate_models
        self._maximize = np.asarray(maximize)

    @property
    def surrogate_model(self):
        return self._surrogate_models

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factors(self):
        return (-1) ** (self._maximize)
    
    @property
    def number_of_objectives(self):
        return len(self._surrogate_models)
    
    def fit(self, X_train, y_train):
        """
        Fit the surrogate models using existing data.

        Parameters
        ----------
        X_train : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.
        y_train : array-like of shape (n_polymers, n_objectives)
            Values associated to each polymer.

        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if y_train.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(y_train)
            )

        # We fit the surrogate model associated with each acquisition function
        for i, surrogate_model in enumerate(self._surrogate_models):
            surrogate_model.fit(X_train, y_train[:, i])

    def forward(self, X_test):
        """
        Predict mean values by the surrogate model for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.

        Returns
        -------
        ndarray of shape (n_polymers, n_objectives)
            The expected improvement values.

        """
        X_test = np.asarray(X_test)
        scores = np.zeros((X_test.shape[0], len(self._surrogate_models)))

        for i, surrogate_model in enumerate(self._surrogate_models):
            mu, _ = surrogate_model.predict(X_test)
            scores[:, i] = mu

        return scores


class ExpectedImprovement(_AcquisitionFunction):
    """
    Class for the Expected Improvement acquisition function.

    Attributes
    ----------
    surrogate_model : list of `_SurrogateModel`
        The surrogate models used by the acquisition function.
    scaling_factors : ndarray of shape (surrogate_models,)
        Scaling factor used by Genetic Algorithm, so it's always a minimization.
    maximize : ndarray of bool of shape (surrogate_models,)
        Tell if the goal(s) is to maximize (True) or minimize (False) the acquisition function.
    number_of_objectives : int
        Number of objectives to optimize.

    """

    def __init__(self, surrogate_models, maximize=False, xi=0.00, eps=1e-9):
        """
        Expected improvement acquisition function.

        Parameters
        ----------
        surrogate_models: `_SurrogateModel` or list of `_SurrogateModel`
            The surrogate model(s) to be used by the acquisition function.
        maximize : bool or list of bool, default : False
            Indicates whether the goal(s) is(are) to be maximised or minimized.
        xi : float, default : 0.
            Exploitation-exploration trade-off parameter.
        eps : float, default : 1e-9
            Small number to avoid numerical instability.

        Notes
        -----
        https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py

        """
        if not isinstance(surrogate_models, (list, tuple, np.ndarray)):
            surrogate_models = [surrogate_models]
        if not isinstance(maximize, (list, tuple, np.ndarray)):
            maximize = [maximize]
        
        msg_error = "The number of surrogate models and maximize flags must be the same."
        assert len(surrogate_models) == len(maximize), msg_error

        self._surrogate_models = surrogate_models
        self._maximize = np.asarray(maximize)
        self._eps = eps
        self._xi = xi

    @property
    def surrogate_model(self):
        return self._surrogate_models

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factors(self):
        return -1 * np.ones(len(self._maximize))
    
    @property
    def number_of_objectives(self):
        return len(self._surrogate_models)

    def fit(self, X_train, y_train):
        """
        Fit the surrogate models using existing data.

        Parameters
        ----------
        X_train : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.
        y_train : array-like of shape (n_polymers, n_objectives)
            Values associated to each polymer.

        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if y_train.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(y_train)
            )

        # We fit the surrogate model associated with each acquisition function
        for i, surrogate_model in enumerate(self._surrogate_models):
            surrogate_model.fit(X_train, y_train[:, i])

    def forward(self, X_test):
        """
        Predict the Expected Improvement values for input polymers.

        Parameters
        ----------
        X_test : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.

        Returns
        -------
        ndarray of shape (n_polymers, n_objectives)
            The expected improvement values.

        """
        X_test = np.asarray(X_test)
        scores = np.zeros((X_test.shape[0], len(self._surrogate_models)))

        for i, surrogate_model in enumerate(self._surrogate_models):
            if self._maximize[i]:
                best_f = np.max(surrogate_model.y_train)
            else:
                best_f = np.min(surrogate_model.y_train)

            # calculate mean and stdev via surrogate function*
            mu, sigma = surrogate_model.predict(X_test)

            u = (mu - best_f - self._xi) / (sigma + self._eps)

            if not self._maximize[i]:
                u = -u

            ucdf = norm.cdf(u)
            updf = norm.pdf(u)
            ei = sigma * (updf + u * ucdf)
            ei[sigma == 0.0] == 0.0

            scores[:, i] = ei

        return scores


class ProbabilityOfImprovement(_AcquisitionFunction):
    """
    Class for the Probability of Improvement acquisition function.

    Attributes
    ----------
    surrogate_model : list of `_SurrogateModel`
        The surrogate models used by the acquisition function.
    scaling_factors : ndarray of shape (surrogate_models,)
        Scaling factor used by Genetic Algorithm, so it's always a minimization.
    maximize : ndarray of bool of shape (surrogate_models,)
        Tell if the goal(s) is to maximize (True) or minimize (False) the acquisition function.
    number_of_objectives : int
        Number of objectives to optimize.

    """

    def __init__(self, surrogate_models, maximize=False, eps=1e-9):
        """
        Probability of Improvement acquisition function.

        Parameters
        ----------
        surrogate_models: `_SurrogateModel` or list of `_SurrogateModel`
            The surrogate model(s) to be used by the acquisition function.
        maximize : bool or list of bool, default : False
            Indicates whether the goal(s) is(are) to be maximised or minimized.
        eps : float, default : 1e-9
            Small number to avoid numerical instability.

        """
        if not isinstance(surrogate_models, (list, tuple, np.ndarray)):
            surrogate_models = [surrogate_models]
        if not isinstance(maximize, (list, tuple, np.ndarray)):
            maximize = [maximize]
        
        msg_error = "The number of surrogate models and maximize flags must be the same."
        assert len(surrogate_models) == len(maximize), msg_error

        self._surrogate_models = surrogate_models
        self._maximize = np.asarray(maximize)
        self._eps = eps

    @property
    def surrogate_model(self):
        return self._surrogate_models

    @property
    def maximize(self):
        return self._maximize

    @property
    def scaling_factors(self):
        return -1 * np.ones(len(self._maximize))
    
    @property
    def number_of_objectives(self):
        return len(self._surrogate_models)
    
    def fit(self, X_train, y_train):
        """
        Fit the surrogate models using existing data.

        Parameters
        ----------
        X_train : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.
        y_train : array-like of shape (n_polymers, n_objectives)
            Values associated to each polymer.

        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if y_train.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(y_train)
            )

        # We fit the surrogate model associated with each acquisition function
        for i, surrogate_model in enumerate(self._surrogate_models):
            surrogate_model.fit(X_train, y_train[:, i])

    def forward(self, X_test):
        """
        Predict the Probability of Improvement values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.

        Returns
        -------
        ndarray of shape (n_polymers, n_objectives)
            The expected improvement values.

        """
        X_test = np.asarray(X_test)
        scores = np.zeros((X_test.shape[0], len(self._surrogate_models)))

        for i, surrogate_model in enumerate(self._surrogate_models):
            if self._maximize[i]:
                best_f = np.max(surrogate_model.y_train)
            else:
                best_f = np.min(surrogate_model.y_train)

            # calculate mean and stdev via surrogate function
            mu, sigma = surrogate_model.predict(X_test)

            # calculate the probability of improvement
            u = (mu - best_f) / (sigma + self._eps)

            if not self._maximize[i]:
                u = -u

            pi = norm.cdf(u)
            pi[sigma == 0.0] == 0.0
            
            scores[:, i] = pi

        return scores


class UpperConfidenceBound(_AcquisitionFunction):
    """
    Class for the Upper Confidence Bound acquisition function.

    Attributes
    ----------
    surrogate_model : list of `_SurrogateModel`
        The surrogate models used by the acquisition function.
    scaling_factors : ndarray of shape (surrogate_models,)
        Scaling factor used by Genetic Algorithm, so it's always a minimization.
    maximize : ndarray of bool of shape (surrogate_models,)
        Tell if the goal(s) is to maximize (True) or minimize (False) the acquisition function.
    number_of_objectives : int
        Number of objectives to optimize.

    """

    def __init__(self, surrogate_models, maximize=False, beta=0.25):
        """
        Upper Confidence Bound acquisition function.

        Parameters
        ----------
        surrogate_models: `_SurrogateModel` or list of `_SurrogateModel`
            The surrogate model(s) to be used by the acquisition function.
        maximize : bool or list of bool, default : False
            Indicates whether the goal(s) is(are) to be maximised or minimized.
        beta : float, default : 0.25
            Exploitation-exploration trade-off parameter.

        """
        if not isinstance(surrogate_models, (list, tuple, np.ndarray)):
            surrogate_models = [surrogate_models]
        if not isinstance(maximize, (list, tuple, np.ndarray)):
            maximize = [maximize]
        
        msg_error = "The number of surrogate models and maximize flags must be the same."
        assert len(surrogate_models) == len(maximize), msg_error

        self._surrogate_models = surrogate_models
        self._maximize = np.asarray(maximize)
        self._delta = 1. - beta
        self._beta = beta

    @property
    def surrogate_model(self):
        return self._surrogate_model
    
    @property
    def maximize(self):
        return self._maximize
    
    @property
    def scaling_factors(self):
        return -1 * np.ones(len(self._maximize))
    
    @property
    def number_of_objectives(self):
        return len(self._surrogate_models)
    
    def fit(self, X_train, y_train):
        """
        Fit the surrogate models using existing data.

        Parameters
        ----------
        X_train : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.
        y_train : array-like of shape (n_polymers, n_objectives)
            Values associated to each polymer.

        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        if y_train.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(y_train)
            )

        # We fit the surrogate model associated with each acquisition function
        for i, surrogate_model in enumerate(self._surrogate_models):
            surrogate_model.fit(X_train, y_train[:, i])
    
    def forward(self, X_test):
        """
        Predict the Upper Confidence Bound values for input sample `X_test`.

        Parameters
        ----------
        X_test : array-like of shape (n_polymers, n_features) or (n_polymers)
            Feature vectors or polymers in HELM format.

        Returns
        -------
        ndarray of shape (n_polymers, n_objectives)
            The expected improvement values.

        """
        X_test = np.asarray(X_test)
        scores = np.zeros((X_test.shape[0], len(self._surrogate_models)))

        for i, surrogate_model in enumerate(self._surrogate_models):
            # calculate mean and stdev via surrogate function
            mu, sigma = surrogate_model.predict(X_test)

            # calculate the upper confidence bound
            ucb = (self._delta * mu) + (self._beta * sigma)
            
            scores[:, i] = ucb

        return scores
