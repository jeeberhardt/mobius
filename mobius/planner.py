#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - Planner
#

from abc import ABC, abstractmethod

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial.distance import pdist, squareform


def calc_mnn_fast(F, **kwargs):
    # Source: pymoo.operators.survival.rank_and_crowding.metrics.py
    # Could not import it directly because of some import issues
    return _calc_mnn_fast(F, F.shape[1], **kwargs)


def _calc_mnn_fast(F, n_neighbors, **kwargs):
    # Source: pymoo.operators.survival.rank_and_crowding.metrics.py
    # Could not import it directly because of some import issues

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = 1.0

    # F normalized
    F = (F - F.min(axis=0)) / norm

    # Distances pairwise (Inefficient)
    D = squareform(pdist(F, metric="sqeuclidean"))

    # M neighbors
    M = F.shape[1]
    _D = np.partition(D, range(1, M+1), axis=1)[:, 1:M+1]

    # Metric d
    d = np.prod(_D, axis=1)

    # Set top performers as np.inf
    _extremes = np.concatenate((np.argmin(F, axis=0), np.argmax(F, axis=0)))
    d[_extremes] = np.inf

    return d


def calc_crowding_distance(F, **kwargs):
    # Source: pymoo.operators.survival.rank_and_crowding.metrics.py
    # Could not import it directly because of some import issues
    n_points, n_obj = F.shape

    # sort each column and get index
    I = np.argsort(F, axis=0, kind='mergesort')

    # sort the objective space values for the whole matrix
    F = F[I, np.arange(n_obj)]

    # calculate the distance from each point to the last and next
    dist = np.row_stack([F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), F])

    # calculate the norm for each objective - set to NaN if all values are equal
    norm = np.max(F, axis=0) - np.min(F, axis=0)
    norm[norm == 0] = np.nan

    # prepare the distance to last and next vectors
    dist_to_last, dist_to_next = dist, np.copy(dist)
    dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

    # if we divide by zero because all values in one columns are equal replace by none
    dist_to_last[np.isnan(dist_to_last)] = 0.0
    dist_to_next[np.isnan(dist_to_next)] = 0.0

    # sum up the distance to next and last and norm by objectives - also reorder from sorted list
    J = np.argsort(I, axis=0)
    cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

    return cd


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


def batch_selection(results, batch_size=96, known_sequences=None):
    """
    Function for selecting the sequence batch to be synthesized next.

    Parameters
    ----------
    results : `tuple` of (ndarray of shape (n_sequences,), ndarray of shape (n_sequences, n_scores)) or list of `tuple`
        Contains the results from the optimization.
    batch_size : int, default: 96
        Number of sequences to select.
    known_sequences : ndarray of shape (n_sequences,), default: None
        Array containing the sequences that have been already synthesized. These sequences will be used to
        remove duplicates from the suggested sequences.

    Returns
    -------
    suggested_sequences : ndarray of shape (batch_size,)
        Array containing the suggested sequences.
    predicted_values : ndarray of shape (batch_size, n_scores)
        Array containing the predicted values of the suggested sequences.

    """
    suggested_sequences = []
    predicted_values = []

    if not isinstance(results, list):
        results = [results]

    # Get the sequences and predicted values from the pymoo results
    for result in results:
        suggested_sequences.append(result[0])
        predicted_values.append(result[1])

    suggested_sequences = np.concatenate(suggested_sequences).flatten()
    predicted_values = np.concatenate(predicted_values)

    # We remove the sequences that have already been synthesized
    if known_sequences is not None:
        common_sequences = np.in1d(suggested_sequences, known_sequences)
        suggested_sequences = suggested_sequences[~common_sequences]
        predicted_values = predicted_values[~common_sequences]

    if predicted_values.shape[1] <= 2:
        crowding_function = calc_crowding_distance
    else:
        crowding_function = calc_mnn_fast

    # Return all the sequences if batch_size is not provided
    if batch_size is None:
        return suggested_sequences, predicted_values

    # Do the selection based on the predicted values
    if predicted_values.shape[1] == 1:
        # Top-k naive batch selection for single-objective optimization
        selected_indices = np.argsort(predicted_values.flatten())[:batch_size]
    else:
        current_rank = 0
        selected_indices = []

        # Non-dominated sorting rank batch selection for multi-objective optimization
        _, ranks = NonDominatedSorting().do(predicted_values, return_rank=True)

        for current_rank in range(0, np.max(ranks)):
            # Get the indices of the sequences in the current rank
            current_indices = np.where(ranks == current_rank)[0]

            if len(selected_indices) + len(current_indices) <= batch_size:
                # Select all the sequences in the current rank
                selected_indices.extend(current_indices)
            else:
                # Get the crowding distances for the sequences in the current rank
                # while taking into account the already selected sequences
                cd = crowding_function(predicted_values[np.concatenate((selected_indices, current_indices)).astype(int)])
                cd = cd[len(selected_indices):]

                # Remove all infinite crowding distances
                current_indices = current_indices[np.isfinite(cd)]
                cd = cd[np.isfinite(cd)]

                # Select the sequences with the highest crowding distance
                selected_indices.extend(current_indices[np.argsort(cd)[::-1][:batch_size - len(selected_indices)]])

            if len(selected_indices) >= batch_size:
                break

    suggested_sequences = suggested_sequences[selected_indices]
    predicted_values = predicted_values[selected_indices]

    return suggested_sequences, predicted_values


class Planner(_Planner):
    """
    Class for setting up the design/optimization/filter protocols for single and multi-objective optimizations.

    """

    def __init__(self, acquisition_function, optimizer):
        """
        Initialize the planner.

        Parameters
        ----------
        acquisition_function : `_AcquisitionFunction`
            The acquisition functions that will be used to score the sequences. For single-objective
            optimisation, only one acquisition function is required. For multi-objective optimisation,
            a list of acquisition functions is required.
        optimizer : `_GeneticAlgorithm`
            The optimizer that will be used to optimize the sequences.

        """
        self._results = None
        self._sequences = None
        self._values = None
        self._noise = None
        self._acq_fun = acquisition_function
        self._optimizer = optimizer

    def ask(self, batch_size=None):
        """
        Function to suggest new sequences based on previous experiments.

        Parameters
        ----------
        batch_size : int, default: None
            Total number of new sequences that will be returned. If not 
            provided, it will return all the sequences found during the optimization.

        Returns
        -------
        sequences : ndarray of shape (batch_size,)
            Suggested sequences. The returned number of sequences 
            will be equal to `batch_size`.
        values : ndarray of shape (batch_size, n_predicted_scores)
            Predicted values for each suggested sequences. The 
            returned number will be equal to `batch_size`.

        """
        # Run the optimizer to suggest new sequences
        self._results = self._optimizer.run(self._sequences.copy(), self._values.copy(), self._acq_fun)

        # Select batch polyners to be synthesized
        suggested_sequences, predicted_values = batch_selection(self._results, batch_size, self._sequences)

        return suggested_sequences, predicted_values

    def tell(self, sequences, values, noises=None, fitting=True):
        """
        Function to fit the surrogate model using data from past experiments.

        Parameters
        ----------
        sequences : array-like of shape (n_sequences,)
            Sequences in HELM format (for polymers) or FASTA format (for bipolymers) (training data).
        values : array-like of shape (n_sequences, n_targets)
            Values associated to each sequence (target values).
        noises : array-like of shape (n_sequences, n_targets), default: None
            Noise values associated to each sequence, and expressed as
            standard deviation (sigma). Values are squared internally to 
            obtain the variance.
        fitting : bool, default: True
            Whether to fit the surrogate model or not.

        """
        self._results = None
        self._sequences = np.asarray(sequences).copy()
        self._values = np.asarray(values).copy()
        self._noises = np.asarray(noises).copy() if noises is not None else None

        # We fit all the surrogate models
        if fitting:
            self._acq_fun.fit(self._sequences, self._values, self._noises)

    def recommend(self, sequences, values, noises=None, batch_size=None, fitting=True):
        """
        Function to suggest new sequences based on existing/previous data.

        Parameters
        ----------
        sequences : array-like of shape (n_sequences,)
            Sequences in HELM format (for polymers) or FASTA format (for bipolymers) (training data).
        values : array-like of shape (n_sequences, n_targets)
            Values associated to each sequence (target values).
        noises : array-like of shape (n_samples,), default: None
            Noise value associated to each sequence, and expressed as
            standard deviation (sigma). Values are squared internally to 
            obtain the variance.
        batch_size : int, default: None
            Total number of new sequences that will be returned. If not 
            provided, it will return all the sequences found during the optimization.
        fitting : bool, default: True
            Whether to fit the surrogate model or not.

        Returns
        -------
        sequences : ndarray of shape (batch_size,)
            Suggested sequencess. The returned number of 
            sequences will be equal to `batch_size`.
        values : ndarray of shape (batch_size, n_scores)
            Predicted values for each suggested sequences. The 
            returned number will be equal to `batch_size`.

        """
        self.tell(sequences, values, noises, fitting)
        suggested_sequences, predicted_values = self.ask(batch_size)

        return suggested_sequences, predicted_values

    def recommand(self, sequences, values, noises=None, batch_size=None, fitting=True):
        """
        Function to suggest new sequences based on existing/previous data.

        Parameters
        ----------
        sequences : array-like of shape (n_sequences,)
            Sequences in HELM format (for polymers) or FASTA format (for bipolymers) (training data).
        values : array-like of shape (n_sequences, n_targets)
            Values associated to each sequence (target values).
        noises : array-like of shape (n_samples,), default: None
            Noise value associated to each sequence, and expressed as
            standard deviation (sigma). Values are squared internally to 
            obtain the variance.
        batch_size : int, default: None
            Total number of new sequences that will be returned. If not 
            provided, it will return all the sequences found during the optimization.
        fitting : bool, default: True
            Whether to fit the surrogate model or not.

        Returns
        -------
        sequences : ndarray of shape (batch_size,)
            Suggested sequencess. The returned number of 
            sequences will be equal to `batch_size`.
        values : ndarray of shape (batch_size, n_scores)
            Predicted values for each suggested sequences. The 
            returned number will be equal to `batch_size`.

        Note
        ----
        This function is a typo and is kept in honor of Aidan Lees.

        """
        return self.recommend(sequences, values, noises, batch_size, fitting)
