#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Pool
#

import numpy as np


class Pool:
    """
    Pool optimization. 

    This optimization is used to evaluate a set of candidates. The candidates are evaluated
    using the acquisition function. The candidates are sorted by the acquisition function
    and the candidates are returned sorted based on the utility scores.

    """

    def __init__(self, candidates, batch_size=-1, **kwargs):
        """
        Initialize the Pool optimization.

        Parameters
        ----------
        candidates : array-like of str
            List of candidates.
        batch_size : int, default : -1
            Number of candidates to be evaluated per batch. If -1, all candidates are evaluated at the same time.
            This is useful when the evaluation of the candidates is memory expensive, for example.
        
        """
        self._candidates = set(candidates)
        self._batch_size = batch_size
        self._designs = None
        self._filters = None
    
    def run(self, polymers, scores, acquisition_function, **kwargs):
        """
        Run the Pool optimization.
        
        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of shape (n_samples, n_objectives)
            Values associated to each polymer/peptide.
        acquisition_functions : `_AcquisitionFunction`
            The acquisition function that will be used to score the polymers.

        Returns
        -------
        polymers : ndarray
            All the polymers evaluated.
        predicted_scores : ndarray
            Utility score for each polymer.

        """
        # Remove candidates in common with the input polymers
        # Bacause we are not going to evaluate them again, they are supposed
        # to be already evaluated (aka experimentally tested)
        candidates = np.asarray(list(self._candidates.difference(polymers)))

        if self._batch_size == -1:
            predicted_scores = acquisition_function.forward(candidates) * acquisition_function.scaling_factors
        else:
            # Evaluate the candidates per batch
            predicted_scores = []

            for i in range(0, len(candidates), self._batch_size):
                batch_candidates = candidates[i:i + self._batch_size]
                batch_scores = acquisition_function.forward(batch_candidates) * acquisition_function.scaling_factors
                predicted_scores.append(batch_scores)

            predicted_scores = np.concatenate(predicted_scores, axis=0)

        if acquisition_function.number_of_objectives == 1:
            predicted_scores = predicted_scores.flatten()
            # Sort the candidates by the acquisition function
            sorted_indices = np.argsort(predicted_scores)
            candidates = candidates[sorted_indices]
            predicted_scores = predicted_scores[sorted_indices].reshape(-1, 1)

        return candidates, predicted_scores
