#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import itertools
import os
import sys
from abc import ABC, abstractmethod

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
from .genetic_operators import GeneticOperators, Mutation, Crossover, DuplicateElimination
from ..utils import generate_random_polymers_from_designs
from ..utils import adjust_polymers_to_designs
from ..utils import group_polymers_by_scaffold
from ..utils import find_closest_points


def _softmax_probability(scores):
    """
    Computes the softmax probability based on the input scores.
    
    Parameters
    ----------
    scores : array-like
        Scores to be used to compute the softmax probabilities.
    
    Returns
    -------
    probabilities : ndarray
        Softmax probability of each score.
    
    """
    return np.exp(-scores) / np.sum(np.exp(-scores))


def _boltzmann_probability(scores, temperature=300.):
    """
    Computes the Boltzmann probability based on the input scores.

    Parameters
    ----------
    scores : array-like
        Scores to be used to compute the Boltzmann probabilities.
    temperature : float, default: 300
        Temperature of the system (without unit).

    Returns
    -------
    probabilities : ndarray
        Boltzmann probability of each score.

    Raises
    ------
    ValueError
        If the computed probabilities contain NaN values.

    """
    # On way to avoid probabilities that do not sum to 1: https://stackoverflow.com/a/65384032
    p = np.exp(-np.around(np.asarray(scores), decimals=3) / temperature).astype('float64')
    p /= np.sum(p)

    if np.isnan(p).any():
        error_msg = 'Boltzmann probabilities contains NaN.\n'
        error_msg += f'Temperature: {temperature:.5f}\n'
        error_msg += f'Values: {scores}\n'
        error_msg += f'Probabilities: {p}'
        raise ValueError(error_msg)

    return p


def _generate_mating_couples(parent_polymers, parent_scores, n_children, temperature):
    """
    Generate mating couples for parent polymers based on their scores
    using the Bolzmann weigthing probabilities. This function is used
    by the SequentialGA class.

    Parameters
    ----------
    parent_polymers: array-like of str
        Parent polymers in HELM format.
    parent_scores: array-like of float or int
        Scores for each parent polymer.
    n_children: int
        Total number of children to generate.
    temperature: float
        Temperature used for the Boltzmann weighting.

    Returns
    -------
    mating_couples: List of tuples
        List of tuples with the two parent polymers that will mate.

    """
    mating_couples = []
    n_couples = int(n_children / 2)
    parent_polymers = np.asarray(parent_polymers)

    # If there is only one parent, automatically it is going to mate with itself...
    if len(parent_polymers) == 1:
        mating_couples = [(parent_polymers[0], parent_polymers[0])] * n_couples
        return mating_couples

    p = _boltzmann_probability(parent_scores, temperature)
    #p = _softmax_probability(parent_scores)
    mates_per_parent = np.floor(n_couples * p).astype(int)

    # In the case no parents really stood up from the rest, all
    # the <n_couples> best parents will be able to mate with someone
    if np.sum(mates_per_parent) == 0:
        print('Warning: none of the parents are worth mating. You might want to decrease the temperature.')
        i = np.argsort(parent_scores)
        mates_per_parent[i[:n_couples]] = 1

    # Complete to reach the number of couples asked
    if np.sum(mates_per_parent) < n_couples:
        nonzero_parent_indices = np.argwhere(mates_per_parent > 0).flatten()
        parent_indices = np.argsort(parent_scores[nonzero_parent_indices])[::-1]

        for i in itertools.cycle(parent_indices):
            mates_per_parent[i] += 1

            if np.sum(mates_per_parent) == n_couples:
                break

    for idx in np.argwhere(mates_per_parent > 0).flatten():
        # Compute Boltzmann probabilities without the selected parent
        mask = np.ones(len(parent_polymers), dtype=bool)
        mask[idx] = False
        p = _boltzmann_probability(parent_scores[mask], temperature)

        # Generate new couples with the selected parent
        mates = np.random.choice(parent_polymers[mask], size=mates_per_parent[idx], replace=True, p=p)
        mating_couples.extend([(parent_polymers[idx], m) for m in mates])

    return mating_couples


@ray.remote
def parallel_ga(gao, polymers, scores, acquisition_function, scaffold_designs):
    return gao.run(polymers, scores, acquisition_function, scaffold_designs)


class _GeneticAlgorithm(ABC):
    """
    Abstract class for genetic algorithm optimization brick
    
    """

    @abstractmethod
    def _generate_new_population(self, polymers, scores, scaffold_designs):
        raise NotImplementedError()

    @abstractmethod
    def run(self, polymers, scores, acquisition_function, scaffold_designs):
        attempts = 0
        best_polymer_seen = None
        # Store all the polymers seen so far...
        polymers_cache = {}
        scaling_factor = acquisition_function.scaling_factor

        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

        for i in range(self._n_gen):
            # Generate new population
            if i == 0:
                # We inverse the scores so the best scores are the lowest ones. This is
                # necessary for calculating the Boltzmann weights correctly. We don't 
                # apply the scaling factor from the acquisition function for the first 
                # generation. The scores at the first generation are the experimental 
                # values and will only apply the -1 factor if the goal to maximize. This
                # is independent of the acquisition function nature.
                if acquisition_function.maximize:
                    scores = -scores

                polymers = self._generate_new_population(polymers, scores, scaffold_designs)
            else:
                # Inverse the sign of the scores from the acquisition function so that
                # the best score is always the lowest, necessary for the Boltzmann weights
                # The scaling factor depends of the acquisition function nature
                polymers = self._generate_new_population(polymers, scaling_factor * scores, scaffold_designs)

            # Keep only unseen polymers. We don't want to reevaluate known polymers...
            polymers_to_evaluate = list(set(polymers).difference(polymers_cache.keys()))

            # If there is no new polymers, we skip the evaluation
            if not polymers_to_evaluate:
                print('Warning: no new polymers were generated. Skip evaluation.')
                # But we still need to retrieve the scores of all the known polymers
                scores = np.array([polymers_cache[p] for p in polymers])
                continue

            # Evaluate new polymers
            results = acquisition_function.forward(polymers_to_evaluate)
            polymers_to_evaluate_scores = results.acq

            # Get scores of already known polymers
            polymers_known = list(set(polymers).intersection(polymers_cache.keys()))
            polymers_known_scores = [polymers_cache[s] for s in polymers_known]

            # Put back together scores and polymers (already seen and evaluated new ones)
            polymers = polymers_known + polymers_to_evaluate
            scores = np.concatenate([polymers_known_scores, polymers_to_evaluate_scores])

            # Store new polymers and scores in the cache
            polymers_cache.update(dict(zip(polymers_to_evaluate, polymers_to_evaluate_scores)))

            # Same thing, we want the best to be the lowest
            idx = np.argmin(scaling_factor * scores)
            current_best_polymer = polymers[idx]
            current_best_score = scores[idx]

            # Convergence criteria
            # If the best score does not improve after N attempts, we stop.
            if best_polymer_seen == current_best_polymer:
                attempts += 1
            else:
                best_polymer_seen = current_best_polymer
                attempts = 0 

            if attempts == self._total_attempts:
                print(f'Reached maximum number of attempts {self._total_attempts}, no improvement observed!')
                break

            print(f'N {i + 1:03d} ({attempts + 1:02d}/{self._total_attempts:02d}) '
                  f'- Score: {current_best_score:5.3f} '
                  f'- {current_best_polymer} ({current_best_polymer.count(".")})')

        all_polymers = np.array(list(polymers_cache.keys()))
        all_polymer_scores = np.fromiter(polymers_cache.values(), dtype=float)

        # Sort polymers by scores in the decreasing order (best to worst)
        # Same same thing, we want the best to be the lowest
        sorted_indices = np.argsort(scaling_factor * all_polymer_scores)

        self.polymers = all_polymers[sorted_indices]
        self.scores = all_polymer_scores[sorted_indices]

        return self.polymers, self.scores


class _SerialSequenceGA(_GeneticAlgorithm):
    """
    Use Serial version of the GA optimization to search for new polymer 
    candidates using the acquisition function for scoring.

    """

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=None, **kwargs):
        """
        Initialize the SequenceGA optimization.

        Parameters
        ----------
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_children : int, default : 500
            Number of children generated at each generation.
        temperature : float, default : 0.01
            Numerical temperature for the Boltzmann weighting selection.
        elitism : bool, default : True
            Use elistism strategy during the search. Best parents will be carried
            over to the next generation along side the new children.
        total_attempt : int, default : 50
            Stopping criteria. Number of attempt before stopping the search. If no
            improvement is observed after `total_attempt` generations, we stop.
        cx_points : int, default : 2
            Number of crossing over during the mating step.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations: int, default : None
            Maximal number of mutations introduced in the new child.

        """
        self.polymers = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._temperature = temperature
        self._elitism = elitism
        self._total_attempts = total_attempts
        self._helmgo = GeneticOperators()
        # Parameters specific to SequenceGA
        self._cx_points = cx_points
        self._pm = pm
        self._minimum_mutations = minimum_mutations
        self._maximum_mutations = maximum_mutations

    def _generate_new_population(self, polymers, scores, scaffold_designs):
        new_pop = []

        mating_couples = _generate_mating_couples(polymers, scores, self._n_children, self._temperature)

        if self._elitism:
            # Carry-on the parents to the next generation
            new_pop.extend(list(np.unique(mating_couples)))

        for mating_couple in mating_couples:
            # This produces two children
            children = self._helmgo.crossover(mating_couple[0], mating_couple[1], self._cx_points)

            for child in children:
                if self._pm <= np.random.uniform():
                    child = self._helmgo.mutate(child, scaffold_designs, 1, self._minimum_mutations, self._maximum_mutations)[0]
                new_pop.append(child)

        return new_pop

    def run(self, polymers, scores, acquisition_function, scaffold_designs):
        """
        Run the SequenceGA optimization.

        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : AcquisitionFunction
            The acquisition function that will be used to score the polymer.
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and defined set of monomers to use 
            for each position.

        Returns
        -------
        polymers : ndarray
            Polymers found during the GA optimization.
        scores : ndarray
            Score for each polymer found.

        """
        # Make sure that inputs are numpy arrays
        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

        self.polymers, self.scores = super().run(polymers, scores, acquisition_function, scaffold_designs)

        return self.polymers, self.scores

class _SequenceGA(_GeneticAlgorithm):
    """
    Parallel version of the SequenceGA optimization to search for new polymer candidates 
    using the acquisition function for scoring.

    This version handle datasets containing polymers with different scaffolds. A SequenceGA
    search will be performed independently for each scaffold. Each SequenceGA search will 
    use 1 cpu-core.

    """

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=None, n_process=-1, **kwargs):
        """
        Initialize the ParallelSequenceGA optimization.

        Parameters
        ----------
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_children : int, default : 500
            Number of children generated at each generation.
        temperature : float, default : 0.01
            Numerical temperature for the Boltzmann weighting selection.
        elitism : bool, default : True
            Use elistism strategy during the search. Best parents will be carried
            over to the next generation along side the new children.
        total_attempt : int, default : 50
            Stopping criteria. Number of attempt before stopping the search. If no
            improvement is observed after `total_attempt` generations, we stop.
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

        """
        self.polymers = None
        self.scores = None
        # Parameters
        self._n_process = n_process
        self._parameters = {'n_gen': n_gen,
                            'n_children': n_children, 
                            'temperature': temperature,
                            'elitism': elitism,
                            'total_attempts': total_attempts, 
                            'cx_points': cx_points, 
                            'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations}
        self._parameters.update(kwargs)

    def _generate_new_population(self, polymers, scores, scaffold_designs):
        raise NotImplementedError()

    def run(self, polymers, scores, acquisition_function, scaffold_designs, **kwargs):
        """
        Run the ParallelSequenceGA optimization.
        
        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : `AcquisitionFunction`
            The acquisition function that will be used to score the polymer.
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and sets of monomers to 
            use for each position.

        Returns
        -------
        polymers : ndarray
            Polymers found during the GA optimization.
        scores : ndarray
            Score for each polymer found.

        """
        # Make sure that inputs are numpy arrays
        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

        # Starts by automatically adjusting the input polymers to the scaffold designs
        polymers, _ = adjust_polymers_to_designs(polymers, scaffold_designs)

        # Group/cluster them by scaffold
        groups, group_indices = group_polymers_by_scaffold(polymers, return_index=True)

        # Check that all the scaffold designs are defined for all the polymers
        scaffolds_not_present = list(set(groups.keys()).difference(scaffold_designs.keys()))

        if scaffolds_not_present:
            msg_error = 'The following scaffolds are not defined: \n'
            for scaffold_not_present in scaffolds_not_present:
                msg_error += f'- {scaffold_not_present}\n'

            raise RuntimeError(msg_error)

        # Do the contrary now: check that at least one polymer is defined 
        # per scaffold. We need to generate at least one polymer per 
        # scaffold to be able to start the GA optimization. Here we 
        # generate 10 random polymers per scaffold. We do thzt in the
        # case we want to explore different scaffolds that are not in
        # the initial dataset.
        scaffolds_not_present = list(set(scaffold_designs.keys()).difference(groups.keys()))

        if scaffolds_not_present:
            tmp_scaffolds_designs = {key: scaffold_designs[key] for key in scaffolds_not_present}
            # We generate them
            n_polymers = [42] * len(tmp_scaffolds_designs)
            new_polymers = generate_random_polymers_from_designs(n_polymers, tmp_scaffolds_designs)
            # We score them
            new_scores = acquisition_function.forward(new_polymers)
            # Add them to the rest
            polymers = np.concatenate([polymers, new_polymers])
            scores = np.concatenate([scores, new_scores])
            # Recluster all of them again (easier than updating the groups)
            groups, group_indices = group_polymers_by_scaffold(polymers, return_index=True)

        seq_gao = _SerialSequenceGA(**self._parameters)

        if len(group_indices) == 1:
            # There is only one scaffold, run it on a single CPU
            polymers, scores = seq_gao.run(polymers, scores, acquisition_function, scaffold_designs)
        else:
            # Take the minimal amount of CPUs needed or available
            if self._n_process == -1:
                self._n_process = min([os.cpu_count(), len(group_indices)])

            # Dispatch all the scaffold accross different independent Sequence GA opt.
            ray.init(num_cpus=self._n_process, ignore_reinit_error=True)

            refs = [parallel_ga.remote(seq_gao, polymers[seq_ids], scores[seq_ids], acquisition_function, scaffold_designs) 
                    for _, seq_ids in group_indices.items()]

            try:
                results = ray.get(refs)
            except:
                ray.shutdown()
                sys.exit(0)

            polymers, scores = zip(*results)
            polymers = np.concatenate(polymers)
            scores = np.concatenate(scores)

            ray.shutdown()

        # Remove duplicates
        polymers, unique_indices = np.unique(polymers, return_index=True)
        scores = scores[unique_indices]

        # Sort polymers by scores in the decreasing order (best to worst)
        # We want the best score to be the lowest, so we apply a scaling 
        # factor (1 or -1). This scalng factor depends of the acquisition
        # function nature.
        sorted_indices = np.argsort(acquisition_function.scaling_factor * scores)

        self.polymers = polymers[sorted_indices]
        self.scores = scores[sorted_indices]

        print(f'End SequenceGA - Best score: {self.scores[0]:5.3f}'
              f' - {self.polymers[0]} ({self.polymers[0].count(".")})')

        return self.polymers, self.scores


class Problem(Problem):
    """
    Class to define Single/Multi/Many-Objectives SequenceGA problem.
    """

    def __init__(self, polymers, scores, acq_funs, n_inequality_constr=0, n_equality_constr=0, **kwargs):
        """
        Initialize the Single/Multi/Many-Objectives SequenceGA problem.

        Parameters
        ----------
        acq_funs : `AcquisitionFunction` or list of `AcquisitionFunction` objects
            Acquisition functions to be evaluated for each new polymer generated.
        n_inequality_constr : int, default : 0
            Number of inequality constraints.
        n_equality_constr : int, default : 0
            Number of equality constraints.
        **kwargs : dict
            Additional keyword arguments.

        """
        super().__init__(n_var=1, n_obj=len(acq_funs),
                         n_ieq_constr=n_inequality_constr,
                         n_eq_constr=n_equality_constr)

        self._prior_data = {p: s for p, s in zip(polymers, scores)}
        self._polymers_cache = {}
        if not isinstance(acq_funs, list):
            acq_funs = [acq_funs]
        self._acq_funs = acq_funs
        self._pre_evaluation = True

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Function to evaluate performance of each new polymer generated.

        Parameters
        ----------
        x : ndarray of str
            Polymers generated by GA to be evaluated by the acquisition functions
        out : ndarray
            Returning objective functions' scores to be minimised by pymoo.

        """
        polymers = x.ravel()

        # Keep only unseen polymers. We don't want to reevaluate known polymers...
        to_not_evaluate_idx = np.nonzero(np.in1d(polymers, self._polymers_cache.keys()))[0]
        to_evaluate_idx = np.nonzero(~np.in1d(polymers, self._polymers_cache.keys()))[0]

        # If there is no new polymers, we skip the evaluation
        if to_evaluate_idx.size == 0:
            # But we still need to retrieve the scores of all the known polymers
            scores = np.array([self._polymers_cache[p] for p in polymers])
            scores = scores.reshape(scores.shape[0], -1)
        else:
            # For the first GA generation, we use the experimental scores
            # then we will use the acquisition scores from the surrogate models.
            # In the pre-evaluation mode, the data are not added to the cache.
            if self._pre_evaluation:
                try:
                    scores = np.array([self._prior_data[p] for p in polymers])
                except KeyError:
                    msg = f'Some polymers not found in the input experimental data. '
                    msg += 'Did you forget to turn on the eval mode?'
                    raise RuntimeError(msg)
            else:
                scores = np.zeros((len(polymers), len(self._acq_funs)))

                # Evaluate unseen polymer with acquisition scores from surrogate models
                for i, acq_fun in enumerate(self._acq_funs):
                    predictions = acq_fun.forward(polymers[to_evaluate_idx])
                    scores[to_evaluate_idx, i] = acq_fun.scaling_factor * predictions.acq

                # Complete with scores of already seen polymers
                seen_scores = np.array([self._polymers_cache[p] for p in polymers[to_not_evaluate_idx]])
                if seen_scores.size > 0:
                    scores[to_not_evaluate_idx, :] = seen_scores

                # Record acquisition score for found polymer in cache
                self._polymers_cache.update(dict(zip(polymers[to_evaluate_idx], scores[to_evaluate_idx])))

        out["F"] = scores
    
    def pre_eval(self):
        """
        Function to set pre-evaluation mode on. In pre-evaluation mode, the scores will be
        obtained from the experimental data only.

        """
        self._pre_evaluation = True

    def eval(self):
        """
        Function to set pre-evaluation mode off. In evaluation mode, the scores will be
        obtained from the surrogate models.

        """
        self._pre_evaluation = False


class SerialSequenceGA():
    """
    Class for the Single/Multi-Objectives SequenceGA optimization.

    """

    def __init__(self, algorithm='NSGA2', n_gen=1000, n_pop=250, period=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=None, **kwargs):
        """
        Initialize the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        algorithm : str, default : 'NSGA2'
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_population : int, default : 250
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

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2' : NSGA2, 'AGEMOEA2': AGEMOEA2}
        self.available_algorithms = self._single | self._multi

        msg_error = f'Only {list(self.available_algorithms.keys())} are supported, not {algorithm}'
        assert algorithm in self.available_algorithms, msg_error

        self.results = None
        self.polymers = None
        self.scores = None
        # Parameters
        self._optimization = 'single' if algorithm in self._single else 'multi'
        self._parameters = {'algorithm': algorithm, 
                            'n_gen': n_gen,
                            'n_pop': n_pop, 
                            'period': period, 
                            'cx_points': cx_points, 
                            'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations}
        self._parameters.update(kwargs)
    
    def run(self, polymers, scores, acquisition_functions, scaffold_designs):
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
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and sets of monomers to 
            use for each position.

        Returns
        -------
        results : `pymoo.model.result.Result`
            Object containing the results of the optimization.

        """
        # Make sure that inputs are numpy arrays
        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

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
        self._mutation = Mutation(scaffold_designs, self._parameters['pm'], 
                                  self._parameters['minimum_mutations'], 
                                  self._parameters['maximum_mutations'])
        self._crossover = Crossover(self._parameters['cx_points'])
        self._duplicates = DuplicateElimination()

        # Initialize the GA method
        GA_method = self.available_algorithms[self._parameters['algorithm']]
        algorithm = GA_method(pop_size=self._parameters['n_pop'], sampling=pop, 
                       crossover=self._crossover, mutation=self._mutation,
                       eliminate_duplicates=self._duplicates)

        # Define termination criteria and make them robust to noise
        no_change_termination = RobustTermination(NoChange(), period=self._parameters['period'])
        max_gen_termination = MaximumGenerationTermination(self._parameters['n_gen'])
        termination = TerminateIfAny(max_gen_termination, no_change_termination)

        # ... and run!
        self.results = minimize(problem, algorithm, 
                                termination=termination,
                                verbose=True, save_history=True)
        
        return self.results


class SequenceGA():
    """
    Class for the Single/Multi-Objectives SequenceGA optimization.

    """

    def __init__(self, algorithm='NSGA2', n_gen=1000, n_pop=250, period=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=None, 
                 n_process=-1, **kwargs):
        """
        Initialize the Single/Multi-Objectives SequenceGA optimization.

        Parameters
        ----------
        algorithm : str, default : 'NSGA2'
            Algorithm to use for the optimization. Can be 'GA' for single-objective 
            optimization, or 'NSGA2' or 'AGEMOEA2' for multi-objectives optimization.
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_population : int, default : 250
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

        """
        self._single = {'GA': GA}
        self._multi = {'NSGA2' : NSGA2, 'AGEMOEA2': AGEMOEA2}
        self.available_algorithms = self._single | self._multi

        msg_error = f'Only {list(self.available_algorithms.keys())} are supported, not {algorithm}'
        assert algorithm in self.available_algorithms, msg_error

        self.results = None
        self.polymers = None
        self.scores = None
        # Parameters
        self._n_process = n_process
        self._parameters = {'algorithm': algorithm,
                            'n_gen': n_gen,
                            'n_pop': n_pop, 
                            'period': period, 
                            'cx_points': cx_points, 
                            'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations}
        self._parameters.update(kwargs)

    def run(self, polymers, scores, acquisition_functions, scaffold_designs):
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
        scaffold_designs : dictionary
            Dictionary with scaffold polymers and sets of monomers to 
            use for each position.

        Returns
        -------
        results : `pymoo.model.result.Result` or list of `pymoo.model.result.Result`
            Object or list of object containing the results of the optimization.

        """
        # Make sure that inputs are numpy arrays
        polymers = np.asarray(polymers)
        scores = np.asarray(scores)

        # Starts by automatically adjusting the input polymers to the scaffold designs
        polymers, _ = adjust_polymers_to_designs(polymers, scaffold_designs)

        # Group/cluster them by scaffold
        groups, group_indices = group_polymers_by_scaffold(polymers, return_index=True)

        # Check that all the scaffold designs are defined for all the polymers
        scaffolds_not_present = list(set(groups.keys()).difference(scaffold_designs.keys()))

        if scaffolds_not_present:
            msg_error = 'The following scaffolds are not defined: \n'
            for scaffold_not_present in scaffolds_not_present:
                msg_error += f'- {scaffold_not_present}\n'

            raise RuntimeError(msg_error)

        # Do the contrary now: check that at least one polymer is defined 
        # per scaffold. We need to generate at least one polymer per 
        # scaffold to be able to start the GA optimization. Here we 
        # generate 42 random polymers per scaffold. We do that in the
        # case we want to explore different scaffolds that are not in
        # the initial dataset.
        scaffolds_not_present = list(set(scaffold_designs.keys()).difference(groups.keys()))

        if scaffolds_not_present:
            tmp_scaffolds_designs = {key: scaffold_designs[key] for key in scaffolds_not_present}
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

        seq_gao = SerialSequenceGA(**self._parameters)

        if len(group_indices) == 1:
            results = seq_gao.run(polymers, scores, acquisition_functions, scaffold_designs)
        else:
            # Take the minimal amount of CPUs needed or available
            if self._n_process == -1:
                self._n_process = min([os.cpu_count(), len(group_indices)])
            
            # Dispatch all the scaffold accross different independent Sequence GA opt.
            ray.init(num_cpus=self._n_process, ignore_reinit_error=True)

            refs = [parallel_ga.remote(seq_gao, polymers[seq_ids], scores[seq_ids], acquisition_functions, scaffold_designs) 
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
