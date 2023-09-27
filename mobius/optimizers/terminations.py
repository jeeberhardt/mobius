#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - termination criteria
#

import numpy as np
from pymoo.util.running_metric import RunningMetric
from pymoo.core.termination import Termination


class RunningMetricTermination(Termination):

    def __init__(self, tol=1e-4):
        """
        Terminate if the running metric is smaller than delta_fmin.

        Parameters
        ----------
        tol : float
            Minimum tol value to terminate the algorithm.


        """
        super().__init__()
        self.tol = tol
        self.running = RunningMetric()

    def _update(self, algorithm):
        """
        Update the running metric.
        
        Parameters
        ----------
        algorithm : pymoo.model.algorithm.Algorithm
            The algorithm object.

        Returns
        -------
        float
            The progress of the algorithm. Returns 0 if the running metric is empty.
            Returns 1 if the delta_f is zero, meaning that the algorithm has converged.

        """
        running = self.running

        # Update the running metric to have the most recent information
        self.running.update(algorithm)

        # Remove the trailing zero
        delta_f = np.asarray(running.delta_f)[:-2]

        if delta_f.size == 0:
            return 0.0
        else:
            # Small trick to avoid division by zero
            # Source: https://stackoverflow.com/a/68118106
            progress = delta_f[-1] and self.tol / delta_f[-1] or 1
            # Clip progress pct to be between 0 and 1
            progress = np.clip(progress, 0.0, 1.0)
            return progress
