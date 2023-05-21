#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer sampler
#

import yaml
import importlib
from abc import ABC, abstractmethod

import numpy as np

from .utils import parse_helm


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


def _load_design_from_config(config_filename):
    """
    Function to load the design protocol from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the sampling protocol.

    Returns
    -------
    designs : list of designs
        List of designs to use to optimize polymer sequences.

    """
    designs = {}
    # If no monomers are defined later, the 20 natural amino acids will be used
    monomers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    try:
        design = config['design']
    except:
        # No design protocol defined
        return monomers, designs
    
    try:
        monomers = []
        for _, monomers_collection in design['monomers'].items():
            monomers.extend(monomers_collection)
    except:
        pass

    for scaffold_design in design['scaffolds']:
        try:
            scaffold_sequence = list(scaffold_design.keys())[0]
            scaffold_instructions = scaffold_design[scaffold_sequence]
        except:
            scaffold_sequence = scaffold_design
            scaffold_instructions = {}
        
        polymers, _, _, _ = parse_helm(scaffold_sequence)

        for pid, sequence in polymers.items():
            scaffold_instructions.setdefault(pid, {})

            for i, monomer in enumerate(sequence):
                if monomer == 'X':
                    scaffold_instructions[pid].setdefault(i + 1, monomers)
                else:
                    scaffold_instructions[pid].setdefault(i + 1, [monomer])
        
        designs[scaffold_sequence] = scaffold_instructions
    
    return monomers, designs


def _load_methods_from_config(config_filename, yaml_key):
    """
    Function to load the method from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the methods to load.
    yaml_key : str
        Key of the YAML config file containing the methods to load.

    Returns
    -------
    method : list of methods
        List of methods loaded from the YAML config file.

    """
    methods = []
    
    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        method_configs = config[yaml_key]
    except:
        return methods

    for method_config in method_configs:
        try:
            method_full_name = method_config['class_path']
        except:
            raise KeyError('No class path defined for %s %s' % (yaml_key, method_config))

        try:
            init_args = method_config['init_args']
        except:
            # If no init args are defined, set it to None
            # The default arguments defined (if any) for this class will then be used
            init_args = None

        method_module_name, method_class_name =  method_full_name.rsplit('.', 1)

        try:
            method_module = importlib.import_module(method_module_name)
        except:
            raise ImportError('Cannot import module %s' % (method_module_name))
        try:
            method_class = getattr(method_module, method_class_name)
        except:
            raise AttributeError('Cannot find class %s in module %s' % (method_class_name, method_module_name))

        try:
            method = method_class(**init_args)
        except:
            raise ValueError('Cannot initialize %s %s' % (yaml_key, method_full_name))

        methods.append(filter)

    return methods


def _load_filters_from_config(config_filename):
    """
    Function to load the filter methods from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the sampling protocol.

    Returns
    -------
    filters : list of filter methods
        List of filter methods to use at the end of the polymer optimization.

    """
    return _load_methods_from_config(config_filename, yaml_key='filters')


def _load_samplers_from_config(config_filename):
    """
    Function to load the sampler methods from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the sampling protocol.

    Returns
    -------
    samplers : list of sampling methods
        List of sampling methods to use to optimize polymer sequences.

    """
    return _load_methods_from_config(config_filename, yaml_key='samplers')


class PolymerSampler(_Sampler):
    """
    Class for sampling the polymer sequence space using an acquisition function
    and a search protocol.

    """

    def __init__(self, acquisition_function, config_filename):
        """
        Initialize the polymer sampler.

        Parameters
        ----------
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymer/peptide.
        config_filename : str
            Path of the YAML config file containing the design, sampling and filters 
            protocols for optimizing polymers/peptides.

        Examples
        --------

        >>> from mobius import PolymerSampler
        >>> ps = PolymerSampler(acq_fun, config_filename='config.yaml')

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
        self._samplers = _load_samplers_from_config(config_filename)
        self._designs = _load_design_from_config(config_filename)
        self._filters = _load_filters_from_config(config_filename)

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
