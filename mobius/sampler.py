#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer sampler
#

from abc import ABC, abstractmethod

import numpy as np


class _Sampler(ABC):

    @abstractmethod
    def ask(self):
        raise NotImplementedError()

    @abstractmethod
    def tell(self):
        raise NotImplementedError()

    @abstractmethod
    def recommand(self):
        raise NotImplementedError()

    @abstractmethod
    def optimize(self):
        raise NotImplementedError()


class PolymerSampler(_Sampler):
    """
    Class for sampling the polymer sequence space using an acquisition function
    and a search protocol.

    """

    def __init__(self, acquisition_function, search_protocol):
        """
        Initialize the polymer sampler.

        Parameters
        ----------
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymer/peptide.
        search_protocol : Dictionary
            Search/sampling protocol describing all the sampling blocks.

        Examples
        --------
        >>> from mobius import PolymerSampler, SequenceGA
        >>> # Define a search protocol using the SequenceGA optimizer
        >>> search_protocol = {
        …    'SequenceGA': {
        …        'function': SequenceGA,
        …        'parameters': {
        …            'n_process': -1,
        …            'n_gen': 1000,
        …            'n_children': 500,
        …            'temperature': 0.01,
        …            'elitism': True,
        …            'total_attempts': 50,
        …            'cx_points': 2,
        …            'pm': 0.1,
        …            'minimum_mutations': 1,
        …            'maximum_mutations': 5
        …        }
        …    }
        }
        >>> ps = PolymerSampler(acq_fun, search_protocol)

        """
        self._acq_fun = acquisition_function
        self._search_protocol = search_protocol

    def ask(self, batch_size=None):
        """
        Function to suggest new polymers/peptides based on previous experiments.

        Parameters
        ----------
        batch_size : int, default: None
            Total number of new polymers/peptides that will be returned. If not 
            provided, it will return all the polymers sampled during the optimization.

        Returns
        -------
        polymers : ndarray
            Suggested polymers/peptides. The returned number of polymers 
            will be equal to `batch_size`.
        values : ndarray
            Predicted values for each suggested polymers/peptides. The 
            returned number will be equal to `batch_size`.

        """
        # Use the training set from the surrogate model as inputs for the optimization
        suggested_polymers = self._acq_fun.surrogate_model.X_train_original.copy()
        values = self._acq_fun.surrogate_model.y_train.copy()

        samplers = [s['function'](**s['parameters']) for name_sampler, s in self._search_protocol.items()]

        for sampler in samplers:
            suggested_polymers, values = sampler.run(self._acq_fun, suggested_polymers, values)

        # Sort sequences by scores in the decreasing order (best to worst)
        # We want the best score to be the lowest, so we apply a scaling 
        # factor (1 or -1). This scalng factor depends of the acquisition
        # function nature.
        sorted_indices = np.argsort(self._acq_fun.scaling_factor * values)

        suggested_polymers = suggested_polymers[sorted_indices]
        values = values[sorted_indices]

        return suggested_polymers[:batch_size], values[:batch_size]

    def tell(self, polymers, values):
        """
        Function to fit the surrogate model using data from past experiments.

        Parameters
        ----------
        polymers : list of str
            Polymers/peptides in HELM format.
        values : list of int of float
            Values associated to each polymer/peptide.

        """
        self._acq_fun.surrogate_model.fit(polymers, values)

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
            provided, it will return all the polymers sampled during the optimization.

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

    def optimize(self, emulator, num_iter, batch_size):
        """
        Function to optimize polymers/peptides based on an emulator/oracle.
        
        Parameters
        ----------
        emulator : `Emulator`
            Emulator (oracle) used to simulate actual lab experiments.
        num_iter : int
            Total number of optimization cycles.
        batch_size : int
            Size of the batches, number of polymers/peptides returned after 
            each optimization cycle.
        
        Returns
        -------
        polymers : ndarray
            Suggested polymers/peptides during the optimization process and sorted 
            based on `values` obtained from `Emulator`.
        values : ndarray
            Values for each polymers/peptides suggested during the optimization 
            process and sorted based on the values obtained from `Emulator`.

        """
        # Use the training set from the surrogate model as inputs for the optimization
        all_suggested_polymers = self._acq_fun.surrogate_model.X_train_original.copy()
        all_exp_values = self._acq_fun.surrogate_model.y_train.copy()

        for i in range(num_iter):
            suggested_polymers, predicted_values = self.recommand(all_suggested_polymers, all_exp_values, batch_size)

            suggested_polymers_fasta = [''.join(c.split('$')[0].split('{')[1].split('}')[0].split('.')) for c in suggested_polymers]
            exp_values = emulator.predict(suggested_polymers_fasta)

            all_suggested_polymers = np.concatenate([all_suggested_polymers, suggested_polymers])
            all_exp_values = np.concatenate([all_exp_values, exp_values])

        # Sort sequences by scores in the decreasing order (best to worst)
        sorted_indices = np.argsort(all_exp_values)

        all_suggested_polymers = all_suggested_polymers[sorted_indices]
        all_exp_values = all_exp_values[sorted_indices]

        return all_suggested_polymers, all_exp_values
