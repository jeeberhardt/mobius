#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Acquisition function
#

import numpy as np
import torch
from scipy.stats import norm

from . import utils

class AcqScoring:
    def __init__(self, model, acq_function, y_train, seq_transformer=None, greater_is_better=True):
        self._model = model
        self._acq_function = acq_function
        self._seq_transformer = seq_transformer
        self._y_train = y_train
        self.greater_is_better = greater_is_better

    def score(self, sequences):
        if self._seq_transformer is not None:
            sequences = self._seq_transformer.transform(sequences)
        return self._acq_function(self._model, self._y_train, sequences, self.greater_is_better)


def random_improvement(model, y_train, X_test, greater_is_better=False):
    """ random acquisition function

    Arguments:
    ----------
        model: Gaussian process model (needed for API compatibility)
        Y_train: Array that contains all the observed energy interaction seed so far (needed for API compatibility)
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.

    """
    X_test = np.array(X_test)
    scaling_factor = (-1) ** (not greater_is_better)
    predictions = scaling_factor * np.random.uniform(low=0, high=1, size=X_test.shape[0])

    return predictions


def greedy(model, y_train, X_test, greater_is_better=False):
    """ greedy acquisition function

    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.

    """
    predictions = model.transform(X_test)
    mu = predictions.mean.detach().numpy()

    return mu


def expected_improvement(model, y_train, X_test, greater_is_better=False, xi=0.00):
    """ expected_improvement
    Expected improvement acquisition function.
    
    Source: https://github.com/thuijskens/bayesian-optimization/blob/master/python/gp.py
    
    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.
        xi: Exploitation-exploration trade-off parameter

    """
    # calculate mean and stdev via surrogate function*
    predictions = model.transform(X_test)
    mu = predictions.mean.detach().numpy()
    sigma = predictions.stddev.detach().numpy()

    if greater_is_better:
        loss_optimum = np.max(y_train)
    else:
        loss_optimum = np.min(y_train)

    scaling_factor = (-1) ** (not greater_is_better)

    # calculate the expected improvement
    Z = scaling_factor * (mu - loss_optimum - xi) / (sigma + 1E-9)
    ei = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + (sigma * norm.pdf(Z))
    ei[sigma == 0.0] == 0.0

    return -1 * ei


def probability_of_improvement(model, y_train, X_test, greater_is_better=False):
    """ probability_of_improvement
    Probability of improvement acquisition function.

    Arguments:
    ----------
        model: Gaussian process model
        Y_train: Array that contains all the observed energy interaction seed so far
        X_samples: Samples we want to try out
        greater_is_better: Indicates whether the loss function is to be maximised or minimised.

    """
    # calculate mean and stdev via surrogate function
    predictions = model.transform(X_test)
    mu = predictions.mean.detach().numpy()
    sigma = predictions.stddev.detach().numpy()

    if greater_is_better:
        loss_optimum = np.max(y_train)
    else:
        loss_optimum = np.min(y_train)

    scaling_factor = (-1) ** (not greater_is_better)

    # calculate the probability of improvement
    Z = scaling_factor * (mu - loss_optimum) / (sigma + 1E-9)
    pi = norm.cdf(Z)
    pi[sigma == 0.0] == 0.0

    return pi
