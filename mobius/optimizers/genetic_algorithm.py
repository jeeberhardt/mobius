#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import importlib
import os
import sys
import yaml

import numpy as np
import ray
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.algorithms.moo.age2 import AGEMOEA2
from pymoo.core.evaluator import Evaluator
from pymoo.core.termination import TerminateIfAny 
from pymoo.termination.max_gen import MaximumGenerationTermination
from pymoo.termination.robust import RobustTermination

from .terminations import NoChange
from .genetic_operators import Mutation, Crossover, DuplicateElimination
from .problem import Problem
from ..utils import generate_random_polymers_from_designs
from ..utils import adjust_polymers_to_designs
from ..utils import group_polymers_by_scaffold
from ..utils import parse_helm, get_scaffold_from_helm_string
from ..utils import generate_design_protocol_from_polymers


@ray.remote
def parallel_ga(gao, polymers, scores, acquisition_function):
    return gao.run(polymers, scores, acquisition_function)


def _load_design_from_config(config):
    """
    Function to load the design protocol from the YAML config file.

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
        List of filter methods to use at the end of the polymer optimization.

    """
    return _load_methods_from_config(config_filename, yaml_key='filters')


class SerialSequenceGA():
    """
    Class for the Single/Multi-Objectives SequenceGA optimization.

    """

    def __init__(self, algorithm='NSGA2', designs=None, filters=None,
                 n_gen=1000, n_pop=500, period=50, cx_points=2, pm=0.1, 
                 minimum_mutations=1, maximum_mutations=None,
                 save_history=False, **kwargs):
        """
        Initialize the Single/Multi-Objectives SequenceGA optimization. The 
        SerialSequenceGA is not meant to be used directly. It is used by the
        SequenceGA class to run the optimization in parallel.

        Parameters
        ----------
        algorithm : str, default : 'NSGA2'
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        designs : list of designs, default: None
            List of designs to use during the polymer optimization. If not provided,
            a default design protocol will be generated automatically based on the
            polymers provided during the optimization, with no filters.
        filters : list of filter methods, default: None
            List of filter methods to use during the polymer optimization.
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
        save_history : bool, default : False
            Save the history of the optimization. This can be useful to debug the
            optimization, but it can take a lot of memory.

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2' : NSGA2, 'AGEMOEA2': AGEMOEA2}
        self._available_algorithms = self._single | self._multi

        msg_error = f'Only {list(self._available_algorithms.keys())} are supported, not {algorithm}'
        assert algorithm in self._available_algorithms, msg_error

        self.results = None
        self.polymers = None
        self.scores = None
        # Design protocol
        self._designs = designs
        self._filters = filters
        # GA Parameters
        self._optimization_type = 'single' if algorithm in self._single else 'multi'
        self._method = self._available_algorithms[algorithm]
        self._n_gen = n_gen
        self._n_pop = n_pop
        self._period = period
        self._cx_points = cx_points
        self._pm = pm
        self._minimum_mutations = minimum_mutations
        self._maximum_mutations = maximum_mutations
        self._save_history = save_history

    def run(self, polymers, scores, acquisition_functions):
        """
        Run the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : `AcquisitionFunction` or list of `AcquisitionFunction` objects
            The acquisition function(s) that will be used to score the polymer.

        Returns
        -------
        results : `pymoo.model.result.Result`
            Object containing the results of the optimization.

        """
        # Starts by automatically adjusting the input polymers to the design
        polymers, _ = adjust_polymers_to_designs(polymers, self._designs)

        # Initialize the problem
        problem = Problem(polymers, scores, acquisition_functions)

        # ... and pre-initialize the population with the experimental data.
        # This is only for the first GA generation.
        X = polymers.reshape(polymers.shape[0], -1)
        pop = Population.new("X", X)
        Evaluator().eval(problem, pop)

        # Turn off the pre-evaluation mode
        # Now it will use the acquisition scores from the surrogate models
        problem.eval()

        # Initialize genetic operators
        self._mutation = Mutation(self._designs, self._pm, self._minimum_mutations, self._maximum_mutations)
        self._crossover = Crossover(self._cx_points)
        self._duplicates = DuplicateElimination()

        # Initialize the GA method
        algorithm = self._method(pop_size=self._n_pop, sampling=pop, 
                                 crossover=self._crossover, mutation=self._mutation,
                                 eliminate_duplicates=self._duplicates)

        # Define termination criteria and make them robust to noise
        no_change_termination = RobustTermination(NoChange(), period=self._period)
        max_gen_termination = MaximumGenerationTermination(self._n_gen)
        termination = TerminateIfAny(max_gen_termination, no_change_termination)

        # ... and run!
        self.results = minimize(problem, algorithm,
                                termination=termination,
                                verbose=True,
                                save_history=self._save_history)

        return self.results


class SequenceGA():
    """
    Class for the Single/Multi-Objectives SequenceGA optimization.

    """

    def __init__(self, algorithm='NSGA2', design_protocol_filename=None,
                 n_gen=1000, n_pop=500, period=50, cx_points=2, pm=0.1, 
                 minimum_mutations=1, maximum_mutations=None, 
                 n_process=-1, save_history=False, **kwargs):
        """
        Initialize the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        algorithm : str, default : 'NSGA2'
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        design_protocol_filename : str, default: None
            Path of the YAML config file containing the design protocol 
            (scaffolds + filters) used during polymers optimization. If not
            provided, a default design protocol will be generated automatically
            based on the polymers provided during the optimization, with no filters.
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
        save_history : bool, default : False
            Save the history of the optimization. This can be useful to debug the
            optimization, but it can take a lot of memory.

        Examples
        --------

        >>> from mobius import SequenceGA
        >>> ps = SequenceGA(algorithm='AGEMOEA2', design_protocol_filename='design_protocol.yaml')

        Example of `config.yaml` defining the scaffold design protocol and the 
        different filters can also be defined to filter out polymers 
        that do not satisfy certain criteria.

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
            filters:
              - class_path: mobius.PeptideSelfAggregationFilter
              - class_path: mobius.PeptideSolubilityFilter
                init_args:
                  hydrophobe_ratio: 0.5
                  charged_per_amino_acids: 5

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2' : NSGA2, 'AGEMOEA2': AGEMOEA2}
        self._available_algorithms = self._single | self._multi

        msg_error = f'Only {list(self._available_algorithms.keys())} are supported, not {algorithm}'
        assert algorithm in self._available_algorithms, msg_error

        self.results = None
        self.polymers = None
        self.scores = None
        # Design protocol
        self._designs = None
        self._filters = None
        # Parameters
        self._optimization_type = 'single' if algorithm in self._single else 'multi'
        self._n_process = n_process
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

    def run(self, polymers, scores, acquisition_functions):
        """
        Run the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : `AcquisitionFunction` or list of `AcquisitionFunction` objects
            The acquisition function(s) that will be used to score the polymers.

        Returns
        -------
        results : `pymoo.model.result.Result` or list of `pymoo.model.result.Result`
            Object or list of object containing the results of the optimization.

        """
        # Make sure that inputs are numpy arrays
        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

        if self._optimization_type == 'single':
            msg_error = 'Only one score per polymer is allowed for single-objective optimization.'
            assert scores.shape[1] == 1, msg_error
        else:
            msg_error = 'Only one score per polymer provided. '
            msg_error += 'You need at least two scores per polymer for multi-objective optimization.'
            assert scores.shape[1] >= 2, msg_error

        # Check that all the scaffold designs are defined for all the polymers
        # First, look at the design protocol. If a design protocol was not provided 
        # at the initialization, we generate a default design protocol based on 
        # the input polymers.
        if self._parameters['design_protocol_filename'] is None:
            design_protocol = generate_design_protocol_from_polymers(polymers)
            self._designs = _load_design_from_config(design_protocol)
        else:
            self._designs = _load_design_from_config(self._parameters['design_protocol_filename'])
            self._filters = _load_filters_from_config(self._parameters['design_protocol_filename'])

        # Second, check that all the scaffold designs are defined for each polymers 
        # by clustering them based on their scaffold.
        groups, group_indices = group_polymers_by_scaffold(polymers, return_index=True)
        scaffolds_not_present = list(set(groups.keys()).difference(self._designs.keys()))

        if scaffolds_not_present:
            msg_error = 'The following scaffolds are not defined: \n'
            for scaffold_not_present in scaffolds_not_present:
                msg_error += f'- {scaffold_not_present}\n'

            raise RuntimeError(msg_error)

        # Lastly, do the contrary and check that at least one polymer is defined 
        # per scaffold present in the design protocol. We need to generate at least 
        # one polymer per scaffold to be able to start the GA optimization. Here we 
        # generate 42 random polymers per scaffold. We do that in the case we want 
        # to explore different scaffolds that are not in the initial dataset.
        scaffolds_not_present = list(set(self._designs.keys()).difference(groups.keys()))

        if scaffolds_not_present:
            tmp_scaffolds_designs = {key: self._designs[key] for key in scaffolds_not_present}
            # We generate them
            n_polymers = [42] * len(tmp_scaffolds_designs)
            new_polymers = generate_random_polymers_from_designs(n_polymers, tmp_scaffolds_designs)
            # We score them
            new_scores = np.zeros(shape=(len(new_polymers), len(acquisition_functions)))
            for i, acq_fun in enumerate(acquisition_functions):
                new_scores[:, i] = acq_fun.forward(new_polymers).acq
            # Add them to the rest
            polymers = np.concatenate([polymers, new_polymers])
            scores = np.concatenate([scores, new_scores])
            # Recluster all of them again (easier than updating the groups)
            groups, group_indices = group_polymers_by_scaffold(polymers, return_index=True)

        # Initialize the GA optimization object
        seq_gao = SerialSequenceGA(designs=self._designs, filters=self._filters, **self._parameters)

        if len(group_indices) == 1:
            results = seq_gao.run(polymers, scores, acquisition_functions)
        else:
            # Take the minimal amount of CPUs needed or available
            if self._n_process == -1:
                self._n_process = min([os.cpu_count(), len(group_indices)])

            # Dispatch all the scaffold accross different independent Sequence GA opt.
            ray.init(num_cpus=self._n_process, ignore_reinit_error=True)

            refs = [parallel_ga.remote(seq_gao, polymers[seq_ids], scores[seq_ids], acquisition_functions) 
                    for _, seq_ids in group_indices.items()]

            try:
                results = ray.get(refs)
            except:
                ray.shutdown()
                sys.exit(0)

            ray.shutdown()

        return results
        

class RandomGA():
    """
    The RandomGA is for benchmark purpose only. It generates random polymers.

    """

    def __init__(self, n_gen=1000, n_children=500, **kwargs):
        """
        Initialize the RandomGA "optimization".

        Parameters
        ----------
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_children : int, default : 500
            Number of children generated at each generation.

        """
        self.polymers = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children

    def run(self, polymers, scores, acquisition_function, scaffold_designs):
        """
        Run the RandomGA "optimization".

        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : AcquisitionFunction (RandomImprovement)
            The acquisition function that will be used to score the polymer.
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and defined set of monomers 
            to use for each position.

        Returns
        -------
        polymers : ndarray
            Polymers found during the GA search.
        scores : ndarray
            Score for each polymer found.

        """
        # Generate (n_children * n_gen) polymers and random score them!
        all_polymers = generate_random_polymers_from_designs(self._n_children * self._n_gen, scaffold_designs)
        all_scores = acquisition_function.forward(all_polymers)

        all_polymers = np.asarray(all_polymers)
        all_scores = np.asarray(all_scores)

        # Sort polymers by scores in the decreasing order (best to worst)
        # The scores are scaled to be sure that the best has the lowest score
        # This scaling factor is based on the acquisition function nature
        sorted_indices = np.argsort(acquisition_function.scaling_factor * scores)

        self.polymers = all_polymers[sorted_indices]
        self.scores = all_scores[sorted_indices]

        print(f'End RandomGA - Best score: {self.scores[0]:5.3f}'
              f' - {self.polymers[0]} ({self.polymers[0].count(".")})')

        return self.polymers, self.scores
