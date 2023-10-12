#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer planner
#

from abc import ABC, abstractmethod

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class _Planner(ABC):

    @abstractmethod
    def ask(self):
        raise NotImplementedError()

    @abstractmethod
    def tell(self):
        raise NotImplementedError()

    @abstractmethod
    def recommand(self):
        raise NotImplementedError()


def batch_selection(results, filters=None, batch_size=96):
    """
    Function for selecting the polymer batch to be synthesized next.

    Parameters
    ----------
    results : `pymoo.model.result.Result` or list of `pymoo.model.result.Result`
        Object or list of object containing the results of the optimization.
    filters : list of `_Filter`, default: None
        List of filters to apply on the polymers. If not provided, no filter will be applied.
    batch_size : int, default: 96
        Number of polymers to select.

    Returns
    -------
    suggested_polymers : numpy.ndarray
        Array containing the suggested polymers.
    predicted_values : numpy.ndarray
        Array containing the predicted values of the suggested polymers.

    """
    suggested_polymers = []
    predicted_values = []

    if not isinstance(results, list):
        results = [results]

    # Get the polymers and predicted values from the pymoo results
    for result in results:
        suggested_polymers.append(result.pop.get('X'))
        predicted_values.append(result.pop.get('F'))

    suggested_polymers = np.concatenate(suggested_polymers).flatten()
    predicted_values = np.concatenate(predicted_values)

    # Apply filters on the suggested polymers
    if filters:
        passed = np.ones(len(suggested_polymers), dtype=bool)

        for filter in filters:
            passed = np.logical_and(passed, filter.apply(suggested_polymers))

        suggested_polymers = suggested_polymers[passed]
        predicted_values = predicted_values[passed]

    # Return all the polymers if batch_size is not provided
    if batch_size is None:
        return suggested_polymers, predicted_values

    # Do the selection based on the predicted values
    if predicted_values.shape[1] == 1:
        # Top-k naive batch selection for single-objective optimization
        sorted_indices = np.argsort(predicted_values.flatten())
    else:
        # Non-dominated sorting rank batch selection for multi-objective optimization
        _, rank = NonDominatedSorting().do(predicted_values, return_rank=True)
        sorted_indices = np.argsort(rank)

    suggested_polymers = suggested_polymers[sorted_indices[:batch_size]]
    predicted_values = predicted_values[sorted_indices[:batch_size]]

    return suggested_polymers, predicted_values


class Planner(_Planner):
    """
    Class for setting up the design/optimization/filter protocols for single and multi-objective optimizations.

    """

    def __init__(self, acquisition_functions, optimizer):
        """
        Initialize the polymer planner.

        Parameters
        ----------
        acquisition_function : `_AcquisitionFunction` or list of `_AcquisitionFunction`
            The acquisition functions that will be used to score the polymers. For single-objective
            optimisation, only one acquisition function is required. For multi-objective optimisation,
            a list of acquisition functions is required.
        optimizer : `_GeneticAlgorithm`
            The optimizer that will be used to optimize the polymers.

        """
        self._results = None
        self._polymers = None
        self._values = None
        if not isinstance(acquisition_functions, list):
            acquisition_functions = [acquisition_functions]
        self._acq_funs = acquisition_functions
        self._optimizer = optimizer
        # This is bad, the filters should be part of the optimizer and
        # and used as constraints during the optimization. Only temporary...
        self._filters = optimizer._filters

    def ask(self, batch_size=None):
        """
        Function to suggest new polymers/peptides based on previous experiments.

        Parameters
        ----------
        batch_size : int, default: None
            Total number of new polymers/peptides that will be returned. If not 
            provided, it will return all the polymers found during the optimization.

        Returns
        -------
        polymers : ndarray
            Suggested polymers/peptides. The returned number of polymers 
            will be equal to `batch_size`.
        values : ndarray
            Predicted values for each suggested polymers/peptides. The 
            returned number will be equal to `batch_size`.

        """
        self._results = None
        suggested_polymers = self._polymers.copy()
        predicted_values = self._values.copy()

        # Run the optimizer to suggest new polymers
        self._results = self._optimizer.run(suggested_polymers, predicted_values, self._acq_funs)

        # Select batch polyners to be synthesized
        suggested_polymers, predicted_values = batch_selection(self._results, self._filters, batch_size)

        return suggested_polymers, predicted_values

    def tell(self, polymers, values):
        """
        Function to fit the surrogate model using data from past experiments.

        Parameters
        ----------
        polymers : list of str
            Polymers/peptides in HELM format.
        values : array-like of shape (n_samples, n_objectives)
            Values associated to each polymer/peptide.

        """
        self._results = None
        self._polymers = np.asarray(polymers).copy()
        self._values = np.asarray(values).copy()

        if self._values.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(self._values)
            )

        msg = f'Number of acquisition functions ({len(self._acq_funs)}) '
        msg += f'and objective values ({self._values.shape[1]}) do not match.'
        assert len(self._acq_funs) == self._values.shape[1], msg

        # We fit the surrogate model associated with each acquisition function
        for i in range(len(self._acq_funs)):
            self._acq_funs[i].surrogate_model.fit(self._polymers, self._values[:,i])

    def recommand(self, polymers, values, batch_size=None):
        """
        Function to suggest new polymers/peptides based on existing/previous data.

        Parameters
        ----------
        polymers : list of str
            Polymers/peptides in HELM format.
        values : list of int of float
            Value associated to each polymer/peptide.
        batch_size : int, default: None
            Total number of new polymers/peptides that will be returned. If not 
            provided, it will return all the polymers found during the optimization.

        Returns
        -------
        polymers : ndarray
            Suggested polymers/peptides. The returned number of 
            polymers/peptides will be equal to `batch_size`.
        values : ndarray
            Predicted values for each suggested polymers/peptides. The 
            returned number will be equal to `batch_size`.

        """
        self.tell(polymers, values)
        suggested_polymers, predicted_values = self.ask(batch_size)

        return suggested_polymers, predicted_values

