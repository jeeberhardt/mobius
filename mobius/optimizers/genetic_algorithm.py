#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import importlib
import os
import sys
import warnings
import yaml
from collections import defaultdict

import numpy as np
import ray
import torch
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.soo.nonconvex.ga import GA

from .ga_biopolymer import SerialBioPolymerGA
from .ga_polymer import SerialPolymerGA
from ..utils import guess_input_formats, build_helm_string
from ..utils import parse_helm, get_scaffold_from_helm_string


@ray.remote
def parallel_ga_cpu(gao, polymers, scores, acquisition_function, design, filters):
    return gao.run(polymers, scores, acquisition_function, design, filters)


@ray.remote(num_gpus=1)
def parallel_ga_gpu(gao, polymers, scores, acquisition_function, design, filters):
    return gao.run(polymers, scores, acquisition_function, design, filters)


def _load_polymer_design_from_config(config):
    """
    Function to load the design protocol for polymers from the YAML config file.

    Parameters
    ----------
    config : YAML config filename or dict
        YAML config file or dictionnary containing the design protocol.

    Returns
    -------
    designs : list of designs
        List of designs to use during the polymer optimization.

    """
    designs = {}
    monomers_collections = {}
    default_monomers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    if isinstance(config, dict):
        design = config
    else:
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        design = config['design']
    except KeyError:
        msg_error = 'The `design` root key is missing in the input design protocol.'
        raise KeyError(msg_error)

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

    try:
        polymer_designs = design['polymers']
    except KeyError:
        msg_error = 'The `polymers` key is missing in the input design protocol.'
        raise KeyError(msg_error)

    if len(polymer_designs) == 0:
        raise ValueError('No polymer design provided. You need to define at least one.')

    for polymer_design in polymer_designs:
        try:
            scaffold_complex_polymer = list(polymer_design.keys())[0]
            scaffold_instructions = polymer_design[scaffold_complex_polymer]
        except:
            scaffold_complex_polymer = polymer_design
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


def _load_biopolymer_design_from_config(config):
    """
    Function to load the design protocol for biopolymers from the YAML config file.

    Parameters
    ----------
    config : YAML config filename or dict
        YAML config file or dictionnary containing the design protocol.

    Returns
    -------
    designs : list of designs
        List of designs to use during the biopolymer optimization.

    """
    designs = {}
    monomers_collections = {}

    if isinstance(config, dict):
        design = config
    else:
        with open(config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    try:
        design = config['design']
    except KeyError:
        msg_error = 'The `design` root key is missing in the input design protocol.'
        raise KeyError(msg_error)

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
        default_monomers = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
                            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    try:
        biopolymer_designs = design['biopolymers']
    except KeyError:
        msg_error = 'The `biopolymers` key is missing in the input design protocol.'
        raise KeyError(msg_error)

    if len(biopolymer_designs) == 0:
        raise ValueError('No biopolymer design provided. You need to define at least one.')

    for biopolymer_design in biopolymer_designs:
        # Check if all the keys exists
        try:
            biopolymer_name = biopolymer_design['name']
        except:
            msg_error = f'A `name` key is missing in the input design protocol.'
            raise KeyError(msg_error)

        try:
            positions = biopolymer_design['positions']
        except:
            msg_error = f'A `positions` key is missing in the input design protocol.'
            raise KeyError(msg_error)

        try:
            length = biopolymer_design['length']
        except:
            msg_error = f'A `length` key is missing in the input design protocol.'
            raise KeyError(msg_error)

        try:
            starting_residue = biopolymer_design['starting_residue']
        except:
            starting_residue = 1

        # Generate the default design. if a position is later defined, None will be replaced.
        design = {position : None for position in range(starting_residue, length + 1)}

        for position, instructions in positions.items():
            probabilities = None

            if isinstance(instructions, dict):
                try:
                    user_defined_monomers = instructions['monomers']
                except KeyError:
                    msg_error = f'A `monomers` key is missing for position {position}.'
                    raise KeyError(msg_error)

                if 'probabilities' in instructions:
                    probabilities = instructions['probabilities']
            elif isinstance(instructions, (list, str)):
                user_defined_monomers = instructions
            elif instructions is None:
                # Use natural amino acids set for positions that are not defined.
                user_defined_monomers = default_monomers
            else:
                msg_error = f'Cannot recognize instructions ({instructions}) for position {position}'
                raise RuntimeError(msg_error)

            if not isinstance(user_defined_monomers, list):
                user_defined_monomers = [user_defined_monomers]

            # if one the monomer is monomers collection, replace it by the monomers collection
            # The monomers collection must be defined in the YAML config file, otherwise it will be
            # considered as a monomer. 
            if set(user_defined_monomers).issubset(monomers_collections.keys()):
                for j, m in enumerate(user_defined_monomers):
                     if m in monomers_collections:
                        user_defined_monomers[j:j + 1] = monomers_collections[m]

            if probabilities is not None:
                probabilities = np.array(probabilities).flatten()
            else:
                # If no probabilities are defined, we set them to be equal
                probabilities = np.ones(len(user_defined_monomers)) / len(user_defined_monomers)

            # Make sure that the probabilities sums up to 1
            probabilities = probabilities / np.sum(probabilities)

            if probabilities.size != len(user_defined_monomers):
                msg_error = f'The number of monomers and probabilities must be equal for position {position} '
                msg_error += f'({len(user_defined_monomers)} != {probabilities.size})'
                raise ValueError(msg_error)

            if isinstance(position, int):
                start = end = position - starting_residue + 1
            elif '-' in position:
                try:
                    start, end = position.split('-')
                    start = int(start) - starting_residue + 1
                    end = int(end) - starting_residue + 1
                except:
                    msg_error = f'Position {position} not recognized. It must be a number (XX) or a range (XX-XX).'
                    raise ValueError(msg_error)
            else:
                msg_error = f'Position {position} not recognized. It must be a number (XX) or a range (XX-XX).'
                raise ValueError(msg_error)

            # Check that the position is greater or equal to the starting residue
            if any(np.array([start, end]) < 0):
                msg_error = f'Position {position} is not valid.'
                msg_error += f' Position must be greater or equal to the starting residue ({starting_residue}).'
                raise ValueError(msg_error)

            for i in range(int(start), int(end) + 1):
                design[i] = {'monomers': user_defined_monomers, 'probabilities': probabilities}

        designs[biopolymer_name] = design

    return designs


def _load_methods_from_config(config, yaml_key):
    """
    Function to load the method from the YAML config file.

    Parameters
    ----------
    config : YAML config filename or dict
        YAML config file or dictionnary containing the methods to load.
    yaml_key : str
        Key of the YAML config file containing the methods to load.

    Returns
    -------
    method : list of methods
        List of methods loaded from the YAML config file.

    """
    methods = []

    if isinstance(config, dict):
        method_configs = config
    else:
        with open(config, 'r') as f:
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
        List of filter methods to use during the GA optimization.

    """
    return _load_methods_from_config(config_filename, yaml_key='filters')


def _group_polymers_by_scaffold(polymers, return_index=False):
    """
    Groups a list polymers in HELM format by their scaffolds.

    Parameters
    ----------
    polymers : List of str
        List of input polymers in HELM format to group.
    return_index : bool, default : False
        Whether to return also the original index of the grouped polymers.

    Returns
    -------
    groups : Dict[str, List of str]
        A dictionary with scaffold polymers as keys and 
        lists of grouped polymers as values.
    group_indices : Dict[str, List of int]
        If `return_index` is True, a dictionary with scaffold polymers 
        as keys and lists of indices of the original polymers.

    Examples
    --------
    >>> polymers = ['PEPTIDE1{A.A.R}$$$$V2.0', 'PEPTIDE1{A.A}$$$$V2.0', 'PEPTIDE1{R.G}$$$$V2.0']
    >>> groups = _group_by_scaffold(polymers)
    >>> print(groups)
    {'X$PEPTIDE1{$X.X.X$}$V2.0': ['PEPTIDE1{A.A.R}$$$$V2.0'], 
     'X$PEPTIDE1{$X.X$}$V2.0': ['PEPTIDE1{A.A}$$$$V2.0', 'PEPTIDE1{R.G}$$$$V2.0']}

    """
    groups = defaultdict(list)
    group_indices = defaultdict(list)

    for i, polymer in enumerate(polymers):
        scaffold_polymer = get_scaffold_from_helm_string(polymer)
        groups[scaffold_polymer].append(polymer)
        group_indices[scaffold_polymer].append(i)

    if return_index:
        return groups, group_indices
    else:
        return groups


def _group_biopolymers_by_design(biopolymers, designs, return_index=False):
    """
    Groups a list biopolymers in FASTA format by design using the sequence lengths.

    Parameters
    ----------
    biopolymers : List of str
        List of input biopolymers in FASTA format to group.
    designs : Dictionnary
        Dictionnary containing the design protocol.
    return_index : bool, default : False
        Whether to return also the original index of the grouped biopolymers.

    Returns
    -------
    groups : Dict[str, List of str]
        A dictionary with the biopolymer names as keys and 
        lists of grouped biopolymers as values.
    group_indices : Dict[str, List of int]
        If `return_index` is True, a dictionary with biopolymer names 
        as keys and lists of indices of the original biopolymers.

    """
    groups = defaultdict(list)
    group_indices = defaultdict(list)
    design_by_lengths = defaultdict(list)

    # Gather all the design names by the length of the sequences they will be applied to.
    # Example: {8: ['PEPTIDE1', 'PEPTIDE2'] -> PEPTIDE1 and PEPTIDE2 will be both applied to 8-mers
    for name, positions in designs.items():
        design_by_lengths[len(positions)].append(name)

    for i, biopolymer in enumerate(biopolymers):
        length = len(biopolymer)

        # A sequence can be part of multiple designs
        for name in design_by_lengths[length]:
            groups[name].append(biopolymer)
            group_indices[name].append(i)

    if return_index:
        return groups, group_indices
    else:
        return groups


def _generate_design_protocol_from_polymers(polymers):
    """
    Generate the bare minimum design protocol yaml config from a list of polymers in HELM format.

    Parameters
    ----------
    polymers : List of str
        List of polymers in HELM format.

    Returns
    -------
    dict
        The design protocol yaml config.

    """
    design_protocol = {
        'design': {
            'monomers': {
                'default': ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
            },
            'polymers': []
        }
    }

    # Get the scaffold of each polymer
    groups = group_polymers_by_scaffold(polymers)
    design_protocol['design']['polymers'] = list(groups.keys())

    return design_protocol


def _generate_random_polymers_from_designs(n_polymers, scaffold_designs):
    """
    Generates random polymers using scaffold designs.

    Parameters
    ----------
    n_polymers : int or list of int
        Number of random polymers to generate, or list of numbers of 
        polymers to generate per scaffold.
    scaffold_designs : dictionary
        Dictionary with scaffold polymers and defined set of monomers 
        to use for each position.

    Returns
    -------
    ndarray
        Randomly generated polymers.

    Raises
    ------
    ValueError
        If the size of the list of number of polymers per scaffold
        is not equal to the number of scaffold.

    """
    random_polymers = []

    if isinstance(n_polymers, int):
        n_polymers_per_scaffold = constrained_sum_sample_nonneg(len(scaffold_designs), n_polymers)
    else:
        if len(n_polymers) != len(scaffold_designs):
            msg_error = 'Size of the list of number of polymers per scaffold must be equal to the number of scaffolds designs'
            raise ValueError(msg_error)

        n_polymers_per_scaffold = n_polymers

    for scaffold, design in scaffold_designs.items():
        i = 0

        complex_polymer, connections, _, _ = parse_helm(scaffold)

        for _ in range(n_polymers_per_scaffold[i]):
            random_complex_polymer = {}

            for pid, simple_polymer in complex_polymer.items():
                random_simple_polymer = list(simple_polymer)

                for i, monomer in enumerate(simple_polymer):
                    if monomer == 'X':
                        random_simple_polymer[i] = np.random.choice(design[pid][i + 1])

                random_complex_polymer[pid] = random_simple_polymer

            random_polymer = build_helm_string(random_complex_polymer, connections)
            random_polymers.append(random_polymer)

        i += 1

    return np.asarray(random_polymers)


def _adjust_polymers_to_designs(polymers, designs):
    """
    Modify polymers to fit given designs.

    Parameters
    ----------
    polymers : List
        List of polymers in HELM format.
    designs : dictionary
        Dictionnary of all the positions allowed to be optimized.

    Returns
    -------
    ndarray
        Adjusted polymers in HELM format based on designs.
    ndarray
        ndarray of boolean values indicating whether the 
        polymers was modified or not.

    """
    modified_polymers = []
    modified = np.zeros(shape=(len(polymers),), dtype=bool)

    for i, polymer in enumerate(polymers):
        complex_polymer, connections, _, _ = parse_helm(polymer)
        scaffold = get_scaffold_from_helm_string(polymer)

        for pid, simple_polymer in complex_polymer.items():
            modified_complex_polymer = {}
            modified_simple_polymer = list(simple_polymer)

            for j, monomer in enumerate(simple_polymer):
                if monomer not in designs[scaffold][pid][j + 1]:
                    modified_simple_polymer[j] = np.random.choice(designs[scaffold][pid][j + 1])
                    modified[i] = True

            modified_complex_polymer[pid] = modified_simple_polymer

        modified_polymer = build_helm_string(modified_complex_polymer, connections)
        modified_polymers.append(modified_polymer)

    return np.asarray(modified_polymers), modified


def _prepare_polymers(sequences, scores, acquisition_function, designs):
    """
    Function to prepare the polymers for the GA optimization.

    Parameters
    ----------
    sequences : ndarray of str
        Sequences in HELM format.
    scores : ndarray of float or int
        Score associated to each sequence.
    acquisition_function : `AcquisitionFunction`
        The acquisition function that will be used to score the adjusted sequences.
    designs : dict of dict of list of str
        Dictionary of lists of strings defining the design protocol for each scaffold.

    Returns
    -------
    sequences : ndarray of str
        Sequences in HELM format.
    scores : ndarray of float or int
        Score associated to each sequence.
    
    """
    # Adjust the sequences to the designs
    adjusted_sequences, adjusted = _adjust_polymers_to_designs(sequences, designs)

    # If any sequence was adjusted, we score them and add them to the rest
    if any(adjusted):
        sequences = np.concatenate([sequences, adjusted_sequences[adjusted]])
        scores = np.concatenate([scores, acquisition_function.forward(adjusted_sequences[adjusted])])

    # And in the case we provided a design protocol, check that at least one polymer 
    # is defined per scaffold present in the design protocol. We need to generate at 
    # least one polymer per scaffold to be able to start the GA optimization. Here we 
    # generate 42 random polymers per scaffold. We do that in the case we want 
    # to explore different scaffolds that are not in the initial dataset.
    groups, group_indices = _group_polymers_by_scaffold(sequences, return_index=True)
    scaffolds_not_present = list(set(designs.keys()).difference(groups.keys()))

    if scaffolds_not_present:
        tmp_scaffolds_designs = {key: designs[key] for key in scaffolds_not_present}
        # We generate them
        n_sequences = [42] * len(tmp_scaffolds_designs)
        new_sequences = _generate_random_polymers_from_designs(n_sequences, tmp_scaffolds_designs)
        # We score them
        new_scores = acquisition_function.forward(new_sequences)
        # Add them to the rest
        sequences = np.concatenate([sequences, new_sequences])
        scores = np.concatenate([scores, new_scores])

    return sequences, scores


def _prepare_biopolymers(sequences, scores, acquisition_function, designs):
    """
    Function to prepare the biopolymers for the GA optimization.

    Parameters
    ----------
    sequences : array-like of str
        Sequences in FASTA format.
    scores : array-like of float or int
        Score associated to each sequence.
    acquisition_function : `AcquisitionFunction`
        The acquisition function that will be used to score the sequences.
    designs : dict of dict of list of str
        Dictionary of lists of strings defining the design protocol for each scaffold.

    Returns
    -------
    sequences : array-like of str
        Sequences in FASTA format.
    scores : array-like of float or int
        Score associated to each sequence.
    
    """
    # Placeholder
    return sequences, scores


class SequenceGA():
    """
    Class for the Single/Multi-Objectives SequenceGA optimization.

    """

    def __init__(self, algorithm, design_protocol_filename=None,
                 n_gen=1000, n_pop=500, period=50, cx_points=2, pm=0.1, 
                 minimum_mutations=1, maximum_mutations=None, 
                 n_process=-1, n_gpu=-1, save_history=False, **kwargs):
        """
        Initialize the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        algorithm : str
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'SMSEMOA', 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        design_protocol_filename : str, default: None
            Path of the YAML config file containing the design protocol 
            (polymers | biopolymers + filters) used during GA optimization. If not
            provided, a default design protocol will be generated automatically
            based on the polymers provided during the optimization, with no filters. For
            bipolymers, a design protocol must be provided.
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_population : int, default : 500
            Size of the population generated at each generation.
        period : int, default : 50
            Stopping criteria. Number of attempt before stopping the search. If no
            improvement is observed after `period` generations, we stop.
        cx_points : int, default : 2
            Number of crossing over during the mating step.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations: int, default : None
            Maximal number of mutations introduced in the new child.
        n_process : int, default : -1
            Number of process to run in parallel. Per default, use all the available core.
        n_gpu : int, default : -1
            Number of GPUs to use in parallel. Per default, use all the available gpus.
        save_history : bool, default : False
            Save the history of the optimization. This can be useful to debug the
            optimization, but it can take a lot of memory.

        Examples
        --------

        >>> from mobius import SequenceGA
        >>> optimizer = SequenceGA(algorithm='SMSEMOA', design_protocol_filename='design_protocol.yaml')

        Example of `design_protocol.yaml` files defining the design protocol for polymers or 
        biopolymers. Different filters can also be defined to filter out polymers/biopolymers 
        that do not satisfy certain criteria during the optimization.

        .. code:: yaml

            design:
                monomers: 
                    default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
                    APOLAR: [A, F, G, I, L, P, V, W]
                    POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
                    AROMATIC: [F, H, W, Y]
                    POS_CHARGED: [K, R]
                    NEG_CHARGED: [D, E]
                polymers:
                    - PEPTIDE1{X.M.X.X.X.X.X.X.X}$$$$V2.0:
                        PEPTIDE1:
                            1: [AROMATIC, NEG_CHARGED]
                            4: POLAR
                            8: [C, G, T, S, V, L, M]
            filters:
              - class_path: mobius.PeptideSelfAggregationFilter
              - class_path: mobius.PeptideSolubilityFilter
                init_args:
                  hydrophobe_ratio: 0.5
                  charged_per_amino_acids: 5
        
        .. code:: yaml

            design:
                monomers: 
                    default: [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y]
                    APOLAR: [A, F, G, I, L, P, V, W]
                    POLAR: [C, D, E, H, K, N, Q, R, K, S, T, M]
                    AROMATIC: [F, H, W, Y]
                    POS_CHARGED: [K, R]
                    NEG_CHARGED: [D, E]
                biopolymers:
                    - name: PROTEIN1
                        starting_residue: 1
                        length: 12
                        positions:
                            1: [AROMATIC, NEG_CHARGED]
                            4: POLAR
                            9-12: 
                                monomers: [A, V, I, L, M, T]
                                probabilities: [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2': NSGA2, 'AGEMOEA2': AGEMOEA2, 'SMSEMOA': SMSEMOA}
        self._available_algorithms = {**self._single, **self._multi}

        if algorithm not in self._available_algorithms:
            msg_error = f'Only {list(self._available_algorithms.keys())} are supported, not {algorithm}'
            raise ValueError(msg_error)

        # Parameters
        self._optimization_type = 'single' if algorithm in self._single else 'multi'
        self._n_process = n_process
        self._n_gpus = n_gpu
        self._parameters = {'algorithm': algorithm,
                            'design_protocol_filename': design_protocol_filename,
                            'n_gen': n_gen,
                            'n_pop': n_pop, 
                            'period': period, 
                            'cx_points': cx_points, 
                            'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations,
                            'save_history': save_history}
        self._parameters.update(kwargs)

    def run(self, sequences, scores, acquisition_function):
        """
        Run the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        sequences : array-like of str
            Sequences in HELM format (for polymers) or FASTA format (for bipolymers).
        scores : array-like of float or int
            Score associated to each sequence.
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the sequences.

        Returns
        -------
        results : `tuple` of (ndarray of shape (n_sequences,), ndarray of shape 
            (n_sequences, n_scores)) or list of `tuple`. Contains the results 
            from the optimization.

        """
        # Make sure that inputs are numpy arrays
        sequences = np.asarray(sequences)
        scores = np.asarray(scores)

        # Check that the number of scores is consistent with the optimization type
        if self._optimization_type == 'single':
            if scores.shape[1] != 1:
                msg_error = 'Only one score per sequence is allowed for single-objective optimization.'
                raise ValueError(msg_error)
        else:
            if scores.shape[1] < 2:
                msg_error = 'Only one score per sequence provided for multi-objective optimization.'
                raise ValueError(msg_error)

        # Check input if they are polymers in HELM format or biopolymers in FASTA format
        sequence_formats = guess_input_formats(sequences)
        unique_sequence_formats = np.unique(sequence_formats)

        if len(unique_sequence_formats) > 1:
            msg_error = f'The input contains sequences in multiple formats: {unique_sequence_formats}'
            raise ValueError(msg_error)
        elif 'unknown' in unique_sequence_formats:
            msg_error = f'The input contains sequences in an unknown format (only HELM or FASTA allowed): \n'
            for s in sequence_formats[sequence_formats == 'unknown']:
                msg_error += f'  -  {s}\n'
            raise ValueError(msg_error)

        if unique_sequence_formats[0] == 'HELM':
            # Use the SerialPolymerGA class for polymers
            serial_seq_ga = SerialPolymerGA

            # If a design protocol was not provided at the initialization, we 
            # generate a default design protocol based on the input polymers.
            if self._parameters['design_protocol_filename'] is None:
                design_protocol = _generate_design_protocol_from_polymers(sequences)
                designs = _load_polymer_design_from_config(design_protocol)
                filters = {}
            else:
                designs = _load_polymer_design_from_config(self._parameters['design_protocol_filename'])
                filters = _load_filters_from_config(self._parameters['design_protocol_filename'])

            sequences, scores = _prepare_polymers(sequences, scores, acquisition_function, designs)
            groups, group_indices = _group_polymers_by_scaffold(sequences, return_index=True)
        else:
            # Use the SerialBioPolymerGA class for biopolymers
            serial_seq_ga = SerialBioPolymerGA

            if self._parameters['design_protocol_filename'] is not None:
                designs = _load_biopolymer_design_from_config(self._parameters['design_protocol_filename'])
                filters = _load_filters_from_config(self._parameters['design_protocol_filename'])
            else:
                raise ValueError('A design protocol must be provided for biopolymers optimization.')

            sequences, scores = _prepare_biopolymers(sequences, scores, acquisition_function, designs)
            groups, group_indices = _group_biopolymers_by_design(sequences, designs, return_index=True)

        # Initialize the GA optimization object
        seq_gao = serial_seq_ga(**self._parameters)

        # Run the GA optimization
        if len(groups) == 1:
            indices = list(group_indices.values())[0]
            design = designs[list(group_indices.keys())[0]]
            results = seq_gao.run(sequences[indices], scores[indices], acquisition_function, design, filters)
        else:
            # Take the minimal amount of CPUs/GPUs needed or what is available
            if self._n_process == -1:
                self._n_process = min([os.cpu_count(), len(groups)])

            if self._n_gpus == -1:
                if torch.cuda.is_available():
                    self._n_gpus = min([torch.cuda.device_count(), len(groups)])
                else:
                    self._n_gpus = 0

            ray.init(num_cpus=self._n_process, num_gpus=self._n_gpus, ignore_reinit_error=True)

            if self._n_gpus > 0:
                # If there are GPUs available, use the gpu version of parallel_ga
                parallel_ga = parallel_ga_gpu
            else:
                # Otherwise, use the cpu version of parallel_ga
                parallel_ga = parallel_ga_cpu

            # Dispatch all the sequences accross different independent Sequence GA opt.
            refs = [parallel_ga.remote(seq_gao, sequences[seq_ids], scores[seq_ids], acquisition_function, designs[name], filters) 
                    for name, seq_ids in group_indices.items()]

            try:
                results = ray.get(refs)
            except Exception as error:
                print("An error occurred:", type(error).__name__, "–", error)
                ray.shutdown()
                sys.exit(1)

            ray.shutdown()

        return results


class RandomGA():
    """
    The RandomGA is for benchmark purpose only. It generates random polymers or biopolymers.

    """

    def __init__(self, design_protocol_filename=None, n_gen=1000, n_children=500, **kwargs):
        """
        Initialize the RandomGA "optimization".

        Parameters
        ----------
        design_protocol_filename : str, default: None
            Path of the YAML config file containing the design protocol 
            (polymers | biopolymers + filters) used during GA optimization. If not
            provided, a default design protocol will be generated automatically
            based on the polymers provided during the optimization, with no filters. For
            bipolymers, a design protocol must be provided.
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_children : int, default : 500
            Number of children generated at each generation.

        """
        # Parameters
        self._design_protocol_filename = design_protocol_filename
        self._n_gen = n_gen
        self._n_children = n_children

    def run(self, sequences, scores, acquisition_function):
        """
        Run the RandomGA "optimization".

        Parameters
        ----------
        sequences : array-like of str
            Sequences in HELM format (for polymers) or FASTA format (for bipolymers).
        scores : array-like of float or int
            Score associated to each sequence.
        acquisition_function : `AcquisitionFunction` (RandomImprovement)
            The acquisition function that will be used to score the sequence.

        Returns
        -------
        sequences : ndarray of shape (n_sequences,)
            Sequences found during the GA search.
        scores : ndarray of shape (n_sequences, n_scores)
            Score for each sequence found.

        """
        sequences = np.asarray(sequences)
        scores = np.asarray(scores)

        if scores.ndim == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\narray={}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample.".format(scores)
            )

        # Check input if they are polymers in HELM format or biopolymers in FASTA format
        sequence_formats = guess_input_formats(sequences)
        unique_sequence_formats = np.unique(sequence_formats)

        if len(unique_sequence_formats) > 1:
            msg_error = f'The input contains sequences in multiple formats: {unique_sequence_formats}'
            raise ValueError(msg_error)
        elif unique_sequence_formats[0] == 'unknown':
            msg_error = f'The input contains sequences in an unknown format (only HELM or FASTA allowed): \n'
            for s in sequence_formats[sequence_formats == 'unknown']:
                msg_error += f'  -  {s}\n'
            raise ValueError(msg_error)

        if unique_sequence_formats[0] == 'HELM':
            if self._design_protocol_filename is None:
                design_protocol = _generate_design_protocol_from_polymers(sequences)
                designs = _load_polymer_design_from_config(design_protocol)
                filters = {}
            else:
                designs = _load_polymer_design_from_config(self._design_protocol_filename)
                filters = _load_filters_from_config(self._design_protocol_filename)

            # Generate (n_children * n_gen) polymers
            all_polymers = _generate_random_polymers_from_designs(self._n_children * self._n_gen, designs, filters)
        else:
            if self._parameters['design_protocol_filename'] is not None:
                designs = _load_biopolymer_design_from_config(self._parameters['design_protocol_filename'])
                filters = _load_filters_from_config(self._parameters['design_protocol_filename'])
            else:
                raise ValueError('A design protocol must be provided for biopolymers optimization.')
            
            # Generate (n_children * n_gen) biopolymers
            all_polymers = _generate_random_polymers_from_designs(self._n_children * self._n_gen, designs, filters)

        all_polymers = np.asarray(all_polymers)

        # ... and random score them!
        all_scores = np.zeros(shape=(len(all_polymers), scores.shape[1]))
        for i in range(scores.shape[1]):
            all_scores[:, i] = acquisition_function.forward(all_polymers)

        return all_polymers, all_scores
