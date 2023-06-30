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
    
    def run(self, polymers, scores, acquisition_function, **kwargs):
        """
        Run the Pool optimization.
        
        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : AcquisitionFunction
            The acquisition function that will be used to score the polymer.

        Returns
        -------
        polymers : ndarray
            All the polymers evaluated.
        scores : ndarray
            Utility score for each polymer.

        """
        scaling_factor = acquisition_function.scaling_factor

        # Remove candidates in common with the input polymers
        # Bacause we are not going to evaluate them again, they are supposed
        # to be already evaluated (aka experimentally tested)
        candidates = np.asarray(list(self._candidates.difference(polymers)))

        # Evaluate the acquisition function for the candidates
        results = acquisition_function.forward(candidates)
        scores = results.acq

        # Sort the candidates by the acquisition function
        sorted_indices = np.argsort(scaling_factor * scores)
        candidates = np.asarray(candidates[sorted_indices])
        scores = np.asarray(scores[sorted_indices])

        return candidates, scores
