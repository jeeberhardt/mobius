#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Mobius - polymer planner
#

import yaml
import importlib
from abc import ABC, abstractmethod

import numpy as np

from .utils import parse_helm, get_scaffold_from_helm_string


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


def _load_design_from_config(config_filename):
    """
    Function to load the design protocol from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the design protocol.

    Returns
    -------
    designs : list of designs
        List of designs to use during the polymer optimization.

    """
    designs = {}
    monomers_collections = {}
    default_monomers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    with open(config_filename, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        design = config['design']
    except:
        # No design protocol defined
        return designs

    try:
        monomers_collections = design['monomers']
    except:
        pass

    try:
        # Get the default monomers collection if redefined by the user
        # otherwise we use the default monomers collection defined above
        # composed of the 20 natural amino acids.
        default_monomers = design['monomers']['default']
    except:
        pass

    for scaffold_design in design['scaffolds']:
        try:
            scaffold_complex_polymer = list(scaffold_design.keys())[0]
            scaffold_instructions = scaffold_design[scaffold_complex_polymer]
        except:
            scaffold_complex_polymer = scaffold_design
            scaffold_instructions = {}

        complex_polymer, _, _, _ = parse_helm(scaffold_complex_polymer)

        for pid, simple_polymer in complex_polymer.items():
            scaffold_instructions.setdefault(pid, {})

            for i, monomer in enumerate(simple_polymer):
                if monomer == 'X':
                    try:
                        user_defined_monomers = scaffold_instructions[pid][i + 1]
                    except:
                        # Use natural amino acids set for positions that are not defined.
                        scaffold_instructions[pid][i + 1] = default_monomers
                        continue

                    if not isinstance(user_defined_monomers, list):
                        user_defined_monomers = [user_defined_monomers]

                    # if one the monomer is monomers collection, replace it by the monomers collection
                    # The monomers collection must be defined in the YAML config file, otherwise it will be
                    # considered as a monomer. 
                    if set(user_defined_monomers).issubset(monomers_collections.keys()):
                        for j, m in enumerate(user_defined_monomers):
                            if m in monomers_collections:
                                user_defined_monomers[j:j+1] = monomers_collections[m]

                    scaffold_instructions[pid][i + 1] = user_defined_monomers
                else:
                    # We could had used the setdefault function here, but we will have
                    # the issue when the user defined monomers for that position, but in the
                    # scaffold the monomer defined is not X. In that case, we use the monomer
                    # from the scaffold.
                    scaffold_instructions[pid][i + 1] = [monomer]

        scaffold = get_scaffold_from_helm_string(scaffold_complex_polymer)
        designs[scaffold] = scaffold_instructions

    return designs


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
            if init_args is None:
                method = method_class()
            else:
                method = method_class(**init_args)
        except:
            raise ValueError('Cannot initialize %s %s' % (yaml_key, method_full_name))

        methods.append(method)

    return methods


def _load_filters_from_config(config_filename):
    """
    Function to load the filter methods from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the filter protocol.

    Returns
    -------
    filters : list of filter methods
        List of filter methods to use at the end of the polymer optimization.

    """
    return _load_methods_from_config(config_filename, yaml_key='filters')


def _load_optimizers_from_config(config_filename):
    """
    Function to load the optimization methods from the YAML config file.

    Parameters
    ----------
    config_filename : YAML config filename
        YAML config file containing the optimization protocol.

    Returns
    -------
    optimizers : list of optimization methods
        List of optimizer methods to use for optimizing polymers.

    """
    return _load_methods_from_config(config_filename, yaml_key='optimizer')


class Planner(_Planner):
    """
    Class for setting up the design/optimization/filter protocols.

    """

    def __init__(self, acquisition_function, config_filename):
        """
        Initialize the polymer planner.

        Parameters
        ----------
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymers.
        config_filename : str
            Path of the YAML config file containing the design, optimization 
            and filters protocols for optimizing polymers.

        Examples
        --------

        >>> from mobius import Planner
        >>> ps = Planner(acq_fun, config_filename='config.yaml')

        Example of `config.yaml` defining the scaffold design protocol, 
        as well as the path of the optimizing methods, here mobius.SequenceGA, 
        and the arguments used to initialize it. Different filters can 
        also be defined to filter out polymers/peptides that do not 
        satisfy certain criteria.

        .. code:: yaml

            design:
                monomers: 
                    default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
                    APOLAR: [A, F, G, I, L, P, V, W]
                    POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
                    AROMATIC: [F, H, W, Y]
                    POS_CHARGED: [K, R]
                    NEG_CHARGED: [D, E]
                scaffolds:
                    - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
                        PEPTIDE1:
                            1: [AROMATIC, NEG_CHARGED]
                            4: POLAR
                            8: [C, G, T, S, V, L, M]
            optimizer:
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
            filters:
              - class_path: mobius.PeptideSelfAggregationFilter
              - class_path: mobius.PeptideSolubilityFilter
                init_args:
                  hydrophobe_ratio: 0.5
                  charged_per_amino_acids: 5

        """
        self._acq_fun = acquisition_function
        self._designs = _load_design_from_config(config_filename)
        self._optimizers = _load_optimizers_from_config(config_filename)
        self._filters = _load_filters_from_config(config_filename)

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
        # Use the training set from the surrogate model as inputs for the optimization
        suggested_polymers = self._acq_fun.surrogate_model.X_train_original.copy()
        values = self._acq_fun.surrogate_model.y_train.copy()

        for optimizer in self._optimizers:
            suggested_polymers, values = optimizer.run(suggested_polymers, values, self._acq_fun, self._designs)

        # Apply filters on the suggested polymers
        if self._filters:
            passed = np.ones(len(suggested_polymers), dtype=bool)

            for filter in self._filters:
                passed = np.logical_and(passed, filter.apply(suggested_polymers))

            suggested_polymers = suggested_polymers[passed]
            values = values[passed]

        # Sort polymers by scores in the decreasing order (best to worst)
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
