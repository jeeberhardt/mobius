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

    def __init__(self, candidates, **kwargs):
        """
        Initialize the Pool optimization.

        Parameters
        ----------
        candidates : array-like of str
            List of candidates.
        
        """
        self._candidates = set(candidates)
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

        predicted_scores = acquisition_function.forward(candidates) * acquisition_function.scaling_factors

        if acquisition_function.number_of_objectives == 1:
            predicted_scores = predicted_scores.flatten()
            # Sort the candidates by the acquisition function
            sorted_indices = np.argsort(predicted_scores)
            candidates = candidates[sorted_indices]
            predicted_scores = predicted_scores[sorted_indices].reshape(-1, 1)

        return candidates, predicted_scores
