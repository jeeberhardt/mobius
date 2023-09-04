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

from .genetic_operators import GeneticOperators
from .problem_moo import MyCrossover, MyMutation, MyDuplicateElimination
from ..utils import generate_random_polymers_from_designs
from ..utils import adjust_polymers_to_designs
from ..utils import group_polymers_by_scaffold
from ..utils import find_closest_points


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.algorithms.moo.age2 import AGEMOEA2


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


class SerialSequenceGA(_GeneticAlgorithm):
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


class SequenceGA(_GeneticAlgorithm):
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

    def run(self, polymers, scores, acquisition_function, scaffold_designs):
        """
        Run the ParallelSequenceGA optimization.
        
        Parameters
        ----------
        polymers : array-like of str
            Polymers in HELM format.
        scores : array-like of float or int
            Score associated to each polymer.
        acquisition_function : AcquisitionFunction
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

        seq_gao = SerialSequenceGA(**self._parameters)

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

        # Sort polymers by score in the decreasing order (best to worst)
        # The scores are scaled to be sure that the best has the lowest score
        # This scaling factor is based on the acquisition function nature
        sorted_indices = np.argsort(acquisition_function.scaling_factor * scores)

        self.polymers = polymers[sorted_indices]
        self.scores = scores[sorted_indices]

        print(f'End SequenceGA - Best score: {self.scores[0]:5.3f}'
              f' - {self.polymers[0]} ({self.polymers[0].count(".")})')

        return self.polymers, self.scores
    

class MOOSequenceGA():

    """
    Sequence GA for optimising peptides for multiple objectives.

    """

    def __init__(self,problem,design_protocol='design_protocol.yaml',batch_size=96):
        
        """
        Initialize the SequenceGA multi-objective optimization.

        Parameters
        ----------
        problem: pymoo problem object
            Defines number of objectives, variables & constraints
        n_gen : int, default : 10
            Number of GA generation to run.
        n_pop : int, default : 250
            Number of children generated at each generation.
        batch_size : int, default : 96
            Number of polymers to return as suggested
        cx_points : int, default : 2
            Number of crossing over during the mating step.
        pm : float, default : 0.1
            Probability of mutation.
        minimum_mutations : int, default : 1
            Minimal number of mutations introduced in the new child.
        maximum_mutations: int, default : None
            Maximal number of mutations introduced in the new child.

        """

        self.batch_size = batch_size

        self.design_protocol = design_protocol

        self._parameters = problem.get_params()

        self.problem = problem
        self.mutation = MyMutation(design_protocol,self._parameters['minimum_mutations'],self._parameters['maximum_mutations'],self._parameters['pm'],self._parameters['keep_connections'])
        self.crossover = MyCrossover(self._parameters['cx_points'])
        self.dupes = MyDuplicateElimination()

    def get_design_protocol(self):

        return self.design_protocol

    def run(self, polymers):
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
        X = np.full((len(polymers),1),None,dtype=object)

        acqs = self.problem.get_acq_funs()
        scores = np.full((len(polymers),len(acqs)),None)

        for i in range(len(polymers)):

            X[i,0] = polymers[i]

            for j in range(len(acqs)):

                score = acqs[j].forward(polymers[i])
                ei = acqs[j].scaling_factor * score.acq

                scores[i][j] = ei[0]

        pop = Population.new("X",X)
        pop.set("F",scores)

        print("Initial population seeded...")
        print("NSGA2")

        algorithm = NSGA2(pop_size=self._parameters['n_pop'],
                          sampling=pop,
                          crossover=self.crossover,
                          mutation=self.mutation,
                          eliminate_duplicates=self.dupes)
        
        res = minimize(self.problem,
                       algorithm,
                       ('n_gen',self._parameters['n_gen']),
                       verbose=True,
                       save_history=True)
        

        all_populations = [e.pop for e in res.history if e.pop is not None]
        final_pop = all_populations[-1]

        final_polymers = final_pop.get("X")
        final_scores = final_pop.get("F")

        final_gen = np.column_stack((final_polymers,final_scores))

        sol_polymers = res.X
        sol_scores = res.F

        print("Best Polymers found so far: ")
        for sol in sol_polymers:
            print(sol)
        print("")
        print("")
    
        solutions = np.column_stack((sol_polymers,sol_scores))

        self.polymers = find_closest_points(final_gen,solutions,polymers,self.batch_size)

        self.problem.reset_cache()
    
        return self.polymers, sol_polymers
        

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
