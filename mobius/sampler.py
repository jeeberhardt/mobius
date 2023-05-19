#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer sampler
#

import yaml
import importlib
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


def _load_samplers_from_config(sampling_config_filename):
    """
    Function to load the samplers from the YAML config file.

    Parameters
    ----------
    sampling_config_filename : YAML config filename
        YAML config file containing the sampling protocol.

    Returns
    -------
    samplers : list of sampling methods
        List of sampling methods to use to optimize peptide sequences.

    """
    samplers = []
    
    with open(sampling_config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        sampling_methods = config['sampling']
    except:
        raise KeyError('No sampling protocol defined in the YAML file')

    for sampling_method in sampling_methods:
        try:
            sampler_full_name = sampling_method['class_path']
        except:
            raise KeyError('No class path defined for sampler %s' % (sampling_method))

        try:
            init_args = sampling_method['init_args']
        except:
            # If no init args are defined, set it to None
            # The default arguments defined (if any) for this class will then be used
            init_args = None

        sampler_module_name, sampler_class_name =  sampler_full_name.rsplit('.', 1)

        try:
            sampler_module = importlib.import_module(sampler_module_name)
        except:
            raise ImportError('Cannot import module %s' % (sampler_module_name))
        try:
            sampler_class = getattr(sampler_module, sampler_class_name)
        except:
            raise AttributeError('Cannot find class %s in module %s' % (sampler_class_name, sampler_module_name))

        try:
            sampler = sampler_class(**init_args)
        except:
            raise ValueError('Cannot initialize sampler %s' % (sampler_full_name))

        samplers.append(sampler)

    return samplers


class PolymerSampler(_Sampler):
    """
    Class for sampling the polymer sequence space using an acquisition function
    and a search protocol.

    """

    def __init__(self, acquisition_function, sampling_config_filename):
        """
        Initialize the polymer sampler.

        Parameters
        ----------
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymer/peptide.
        sampling_config_filename : str
            Path of the YAML config file containing the sampling protocol for 
            optimizing polymers/peptides.

        Examples
        --------

        >>> from mobius import PolymerSampler
        >>> ps = PolymerSampler(acq_fun, sampling_config_filename='config.yaml')

        Content of the `config.yaml` with the path of the SequenceGA sampling method 
        defined and the arguments used to initialize it.
    
        .. code:: yaml
        
            sampling:
              - class_path: mobius.SequenceGA
                init_args:
                  n_gen: 1000
                  n_children: 500
                  temperature: 0.01
                  elitism: True
                  total_attempts: 50
                  cx_points: 2
                  pm: 0.1
                  minimum_mutations: 1
                  maximum_mutations: 5

        """
        self._acq_fun = acquisition_function
        self._samplers = _load_samplers_from_config(sampling_config_filename)

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

        for sampler in self._samplers:
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
