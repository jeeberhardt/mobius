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
    
    def run(self, polymers, scores, acquisition_functions, **kwargs):
        """
        Run the Pool optimization.
        
        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of shape (n_samples, n_objectives)
            Values associated to each polymer/peptide.
        acquisition_functions : `_AcquisitionFunction` or list of `_AcquisitionFunction`
            The acquisition functions that will be used to score the polymers. For single-objective
            optimisation, only one acquisition function is required. For multi-objective optimisation,
            a list of acquisition functions is required.

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

        predicted_scores = np.zeros((len(candidates), len(acquisition_functions)))

        for i, acq_fun in enumerate(acquisition_functions):
            # Evaluate the acquisition function for the candidates
            results = acq_fun.forward(candidates)
            predicted_scores[:, i] = results.acq

        if len(acquisition_functions) == 1:
            predicted_scores = predicted_scores.flatten()
            # Sort the candidates by the acquisition function
            sorted_indices = np.argsort(acquisition_functions[0].scaling_factor * predicted_scores)
            candidates = candidates[sorted_indices]
            predicted_scores = predicted_scores[sorted_indices].reshape(-1, 1)

        return candidates, predicted_scores
