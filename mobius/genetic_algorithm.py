#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import itertools
import os
import sys
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import ray

from .genetic_operators import GeneticOperators
from .utils import get_scaffold_from_helm_string
from .utils import generate_random_linear_peptides


def _group_by_scaffold(helm_sequences, return_index=False):
    """
    Groups a list of HELM sequences by their scaffolds.

    Parameters
    ----------
    helm_sequences : List of str
        List of input sequences to group in HELM format.
    return_index : bool, default : False
        Whether to return also the original index of the grouped sequences.

    Returns
    -------
    groups : Dict[str, List of str]
        A dictionary with scaffold sequences as keys and 
        lists of grouped sequences as values.
    group_indices : Dict[str, List of int]
        If `return_index` is True, a dictionary with scaffold sequences 
        as keys and lists of indices of the original sequences.

    Examples
    --------
    >>> sequences = ['PEPTIDE1{A.A.R}$$$$V2.0', 'PEPTIDE1{A.A}$$$$V2.0', 'PEPTIDE1{R.G}$$$$V2.0']
    >>> groups = _group_by_scaffold(sequences)
    >>> print(groups)
    {'X$PEPTIDE1{$X.X.X$}$V2.0': ['PEPTIDE1{A.A.R}$$$$V2.0'], 
     'X$PEPTIDE1{$X.X$}$V2.0': ['PEPTIDE1{A.A}$$$$V2.0', 'PEPTIDE1{R.G}$$$$V2.0']}
    
    """
    groups = defaultdict(list)
    group_indices = defaultdict(list)

    for i, helm_sequence in enumerate(helm_sequences):
        scaffold_sequence = get_scaffold_from_helm_string(helm_sequence)
        groups[scaffold_sequence].append(helm_sequence)
        group_indices[scaffold_sequence].append(i)

    if return_index:
        return groups, group_indices
    else:
        return groups


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
        error_msg += 'Temperature: %.5f\n' % temperature
        error_msg += 'Values: %s\n' % scores
        error_msg += 'Probabilities: %s' % p
        raise ValueError(error_msg)

    return p


def _number_of_children_to_generate_per_parent(parent_scores, n_children, temperature):
    """
    Calculate the number of children to generate per parent 
    based on their scores and the temperature. This function is
    used by the ScaffoldGA class.

    Parameters
    ----------
    parent_scores: array-like
        Scores of the parents used to calculate the number of children to generate.
    n_children: int
        Total number of children to generate.
    temperature: float
       Temperature used for the Boltzmann weighting.

    Returns
    -------
    children_per_parent: ndarray
        Array of integers with the number of children to generate per parent.

    """
    # Boltzmann weighting
    children_per_parent = np.floor(n_children * _boltzmann_probability(parent_scores, temperature)).astype(int)

    current_n_children = np.sum(children_per_parent)

    # In the case, none of the parents are producing children
    # The best parent generate all the children
    if current_n_children == 0:
        i = np.argmax(parent_scores)
        children_per_parent[i] = n_children

        return children_per_parent

    # We are going to add 1 until the number of children is equal to n_children
    # but only when the number of children is higher than zero   
    if current_n_children < n_children:
        # Get indices of parents that will have children and sort in descending order
        nonzero_parent_indices = np.argwhere(children_per_parent > 0).flatten()
        parent_indices = np.argsort(children_per_parent[nonzero_parent_indices])[::-1]

        children_per_parent[parent_indices[:np.sum(children_per_parent) - n_children]] += 1

    return children_per_parent


def _generate_mating_couples(parent_sequences, parent_scores, n_children, temperature):
    """
    Generate mating couples for parent sequences based on their scores
    using the Bolzmann weigthing probabilities. This function is used
    by the SequentialGA class.

    Parameters
    ----------
    parent_sequences: List of str
        Parent sequences.
    parent_scores: list of float
        Scores for each parent sequence.
    n_children: int
        Total number of children to generate.
    temperature: float
        Temperature used for the Boltzmann weighting.

    Returns
    -------
    mating_couples: List of tuple
        List of tuples with the two parent sequences that will mate.

    """
    mating_couples = []
    n_couples = int(n_children / 2)
    parent_sequences = np.asarray(parent_sequences)

    # If there is only one parent, automatically it is going to mate with itself...
    if len(parent_sequences) == 1:
        mating_couples = [(parent_sequences[0], parent_sequences[0])] * n_couples
        return mating_couples

    p = _boltzmann_probability(parent_scores, temperature)
    mates_per_parent = np.floor(n_couples * p).astype(int)

    # In the case no parents really stood up from the rest, all
    # the <n_couples> best parents will be able to mate with someone
    if np.sum(mates_per_parent) == 0:
        print('Warning: It seems that none of the parents are worth mating. You might want to decrease the temperature.')
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
        mask = np.ones(len(parent_sequences), dtype=bool)
        mask[idx] = False
        p = _boltzmann_probability(parent_scores[mask], temperature)

        # Generate new couples with the selected parent
        mates = np.random.choice(parent_sequences[mask], size=mates_per_parent[idx], replace=True, p=p)
        mating_couples.extend([(parent_sequences[idx], m) for m in mates])

    return mating_couples


@ray.remote
def parallel_ga(gao, acquisition_function, sequences, scores):
    return gao.run(acquisition_function, sequences, scores)


class _GeneticAlgorithm(ABC):
    """
    Abstract class for genetic algorithm brick
    
    """

    @abstractmethod
    def _generate_new_population(self, sequences, scores):
        raise NotImplementedError()

    @abstractmethod
    def run(self, acquisition_function, sequences, scores):
        attempts = 0
        best_sequence_seen = None
        # Store all the sequences seen so far...
        sequences_cache = {}
        scaling_factor = acquisition_function.scaling_factor

        sequences = np.asarray(sequences)
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

                sequences = self._generate_new_population(sequences, scores)
            else:
                # Inverse the sign of the scores from the acquisition function so that
                # the best score is always the lowest, necessary for the Boltzmann weights
                # The scaling factor depends of the acquisition function nature
                sequences = self._generate_new_population(sequences, scaling_factor * scores)

            # Keep only unseen sequences. We don't want to reevaluate known sequences...
            sequences_to_evaluate = list(set(sequences).difference(sequences_cache.keys()))

            # If there is no new sequences, we skip the evaluation
            if not sequences_to_evaluate:
                print('Warning: no new sequences were generated. Skip evaluation.')
                # But we still need to retrieve the scores of all the known sequences
                scores = np.array([sequences_cache[s] for s in sequences])
                continue

            # Evaluate new sequences
            sequences_to_evaluate_scores = acquisition_function.forward(sequences_to_evaluate)

            # Get scores of known sequences
            sequences_known = list(set(sequences).intersection(sequences_cache.keys()))
            sequences_known_scores = [sequences_cache[s] for s in sequences_known]

            # New population (known + unseen sequences)
            sequences = sequences_known + sequences_to_evaluate
            scores = np.concatenate([sequences_known_scores, sequences_to_evaluate_scores])

            # Store new sequences and scores in the cache
            sequences_cache.update(dict(zip(sequences_to_evaluate, sequences_to_evaluate_scores)))

            # Same thing, we want the best to be the lowest
            idx = np.argmin(scaling_factor * scores)
            current_best_sequence = sequences[idx]
            current_best_score = scores[idx]

            # Convergence criteria
            # If the best score does not improve after N attempts, we stop.
            if best_sequence_seen == current_best_sequence:
                attempts += 1
            else:
                best_sequence_seen = current_best_sequence
                attempts = 0 

            if attempts == self._total_attempts:
                print('Reached maximum number of attempts (%d), no improvement observed!' % self._total_attempts)
                break

            print('N %03d - Score: %.6f - Seq: %d - %s (%03d/%03d) - New seq: %d' % (i + 1, current_best_score, 
                                                                                     current_best_sequence.count('.'),
                                                                                     current_best_sequence, attempts + 1, 
                                                                                     self._total_attempts,
                                                                                     len(sequences_to_evaluate)))

        all_sequences = np.array(list(sequences_cache.keys()))
        all_sequence_scores = np.fromiter(sequences_cache.values(), dtype=float)

        # Sort sequences by scores in the decreasing order (best to worst)
        # Same same thing, we want the best to be the lowest
        sorted_indices = np.argsort(scaling_factor * all_sequence_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]

        return self.sequences, self.scores


class SequenceGA(_GeneticAlgorithm):
    """
    Use GA to search for new sequence candidates using the acquisition function
    for scoring.

    """

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=1, monomer_symbols=None,
                 **kwargs):
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
        maximum_mutations: int, default : 1
            Maximal number of mutations introduced in the new child.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used during the search. 
            Per default, only the 20 natural amino acids will be used.

        """
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._temperature = temperature
        self._elitism = elitism
        self._total_attempts = total_attempts
        self._helmgo = GeneticOperators(monomer_symbols=monomer_symbols)
        # Parameters specific to SequenceGA
        self._cx_points = cx_points
        self._pm = pm
        self._minimum_mutations = minimum_mutations
        self._maximum_mutations = maximum_mutations

    def _generate_new_population(self, sequences, scores):
        new_pop = []

        mating_couples = _generate_mating_couples(sequences, scores, self._n_children, self._temperature)

        if self._elitism:
            # Carry-on the parents to the next generation
            new_pop.extend(list(np.unique(mating_couples)))

        for mating_couple in mating_couples:
            # This produces two children
            children = self._helmgo.crossover(mating_couple[0], mating_couple[1], self._cx_points)

            for child in children:
                if self._pm <= np.random.uniform():
                    child = self._helmgo.mutate(child, 1, self._minimum_mutations, self._maximum_mutations)[0]
                new_pop.append(child)

        return new_pop

    def run(self, acquisition_function, sequences, scores):
        """
        Run the SequenceGA search.
        
        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function that will be used to score the polymer.
        sequences : list of str
            List of all the polymers in HELM format.
        scores : array-like of shape (n_samples, )
            List of all the value associated to each polymer.

        Returns
        -------
        sequences : array-like of shape (n_samples,)
            All the sequences found during the GA search.
        scores : array-like of shape (n_samples,)
            All the scores for each sequences found.

        """
        _, group_indices = _group_by_scaffold(sequences, return_index=True)

        if len(group_indices) > 1:
            msg = 'presence of polymers with different scaffolds. Please use ParallelSequenceGA.'
            raise RuntimeError(msg)

        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)

        print('End Sequence GA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class ParallelSequenceGA(_GeneticAlgorithm):
    """
    Parallel version of the SequenceGA to search for new sequence candidates using the 
    acquisition function for scoring.
    
    This version handle datasets containing polymers with different scaffolds. A SequenceGA
    search will be performed independently for each scaffold. Each SequenceGA search will 
    use 1 cpu-core.

    """

    def __init__(self, n_gen=1000, n_children=500, temperature=0.01, elitism=True, total_attempts=50,
                 cx_points=2, pm=0.1, minimum_mutations=1, maximum_mutations=1, monomer_symbols=None, 
                 n_process=-1, **kwargs):
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
        maximum_mutations: int, default : 1
            Maximal number of mutations introduced in the new child.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used during the search. 
            Per default, only the 20 natural amino acids will be used.
        n_process : int, default : -1
            Number of process to run in parallel. Per default, use all the available core.

        """
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_process = n_process
        self._parameters = {'n_gen': n_gen, 'n_children': n_children, 
                            'temperature': temperature, 'elitism': elitism,
                            'total_attempts': total_attempts, 
                            'cx_points': cx_points, 'pm': pm,
                            'minimum_mutations': minimum_mutations, 
                            'maximum_mutations': maximum_mutations,
                            'monomer_symbols': monomer_symbols}
        self._parameters.update(kwargs)

    def _generate_new_population(self, sequences, scores):
        raise NotImplementedError()

    def run(self, acquisition_function, sequences, scores):
        """
        Run the ParallelSequenceGA search.
        
        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function that will be used to score the polymer.
        sequences : list of str
            List of all the polymers in HELM format.
        scores : array-like of shape (n_samples, )
            List of all the value associated to each polymer.

        Returns
        -------
        sequences : array-like of shape (n_samples,)
            All the sequences found during the GA search.
        scores : array-like of shape (n_samples,)
            All the scores for each sequences found.

        """
        all_sequences = []
        all_sequence_scores = []
        
        # Make sure that inputs are numpy arrays
        sequences = np.asarray(sequences)
        scores = np.asarray(scores)

        # Group/cluster peptides by scaffold
        _, group_indices = _group_by_scaffold(sequences, return_index=True)

        # Take the minimal amount of CPUs needed or available
        if self._n_process == -1:
            self._n_process = min([os.cpu_count(), len(group_indices)])

        # Dispatch all the scaffold accross different independent Sequence GA opt.
        ray.init(num_cpus=self._n_process, ignore_reinit_error=True)

        seq_gao = SequenceGA(**self._parameters)
        refs = [parallel_ga.remote(seq_gao, acquisition_function, sequences[seq_ids], scores[seq_ids]) for _, seq_ids in group_indices.items()]

        try:
            results = ray.get(refs)
        except:
            ray.shutdown()
            sys.exit(0)

        sequences, scores = zip(*results)
        ray.shutdown()

        # Remove duplicates
        sequences, unique_indices = np.unique(np.concatenate(sequences), return_index=True)
        scores = np.concatenate(scores)[unique_indices]

        # Sort sequences by scores in the decreasing order (best to worst)
        # The scores are scaled to be sure that the best has the lowest score
        # This scaling factor is based on the acquisition function nature
        sorted_indices = np.argsort(acquisition_function.scaling_factor * scores)

        self.sequences = sequences[sorted_indices]
        self.scores = scores[sorted_indices]

        print('End SequenceGA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class ScaffoldGA(_GeneticAlgorithm):
    """
    Use GA to search for new sequence candidates using the acquisition function
    for scoring.

    """

    def __init__(self, n_gen=1, n_children=1000, temperature=0.1, elitism=True, total_attempts=50,
                 only_terminus=True, minimum_size=None, maximum_size=None, monomer_symbols=None,
                 **kwargs):
        """
        Initialize the ScaffoldGA optimization.
        
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
        only_terminus : bool, default : True
            If `True`, only add new monomer at the polymer terminus.
        minimum_size : int, default : 1
            Minimal size of the polymers explored during the search.
        maximum_size: int, default : 1
            Maximal size of the polymers explored during the search.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used during the search. 
            Per default, only the 20 natural amino acids will be used.

        """
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._temperature = temperature
        self._elitism = elitism
        self._total_attempts = total_attempts
        self._helmgo = GeneticOperators(monomer_symbols=monomer_symbols)
        # Parameters specific to ScaffoldGA
        self._only_terminus = only_terminus
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size

    def _generate_new_population(self, sequences, scores):
        new_pop = []

        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature)

        # Generate new population
        parent_indices = np.argwhere(children_per_parent > 0).flatten()

        for i in parent_indices:
            if self._elitism:
                # Carry-on the parents to the next generation
                new_pop.append(sequences[i])

            actions = np.random.choice(['insert', 'remove'], size=children_per_parent[i])
            new_pop.extend(self._helmgo.insert(sequences[i], np.sum(actions == "insert"), self._only_terminus, self._maximum_size))
            new_pop.extend(self._helmgo.delete(sequences[i], np.sum(actions == "remove"), self._only_terminus, self._minimum_size))

        return new_pop

    def run(self, acquisition_function, sequences, scores):
        """
        Run the ScaffoldGA search.
        
        Parameters
        ----------
        acquisition_function : AcquisitionFunction
            The acquisition function that will be used to score the polymer.
        sequences : list of str
            List of all the polymers in HELM format.
        scores : array-like of shape (n_samples, )
            List of all the value associated to each polymer.

        Returns
        -------
        sequences : array-like of shape (n_samples,)
            All the sequences found during the GA search.
        scores : array-like of shape (n_samples,)
            All the scores for each sequences found.

        """
        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)

        print('End Scaffold GA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class RandomGA():
    """
    The RandomGA is for benchmark purpose only. It generates random liner polymer sequences.

    """

    def __init__(self, n_gen=1000, n_children=500, minimum_size=1, maximum_size=1, 
                 monomer_symbols=None, n_process=-1, **kwargs):
        """
        Initialize the RandomGA "optimization".
        
        Parameters
        ----------
        n_gen : int, default : 1000
            Number of GA generation to run.
        n_children : int, default : 500
            Number of children generated at each generation.
        minimum_size : int, default : 1
            Minimal size of the polymers explored during the search.
        maximum_size: int, default : 1
            Maximal size of the polymers explored during the search.
        monomer_symbols : list of str, default : None
            Symbol (1 letter) of the monomers that are going to be used during the search. 
            Per default, only the 20 natural amino acids will be used.
        n_process : int, default : -1
            Number of process to run in parallel. Per default, use all the available core.

        """
        self.sequences = None
        self.scores = None
        # Parameters
        self._n_gen = n_gen
        self._n_children = n_children
        self._minimum_size = minimum_size
        self._maximum_size = maximum_size
        self._helmgo = GeneticOperators(monomer_symbols=monomer_symbols)

    def run(self, acquisition_function, sequences=None, scores=None):
        """
        Run the RandomGA "search".
        
        Parameters
        ----------
        acquisition_function : AcquisitionFunction (RandomImprovement)
            The acquisition function that will be used to score the polymer.
        sequences : list of str
            List of all the polymers in HELM format.
        scores : array-like of shape (n_samples, )
            List of all the value associated to each polymer.

        Returns
        -------
        sequences : array-like of shape (n_samples,)
            All the sequences found during the GA search.
        scores : array-like of shape (n_samples,)
            All the scores for each sequences found.

        """
        peptide_lengths = list(range(self._minimum_size, self._maximum_size + 1))

        # Generate (n_children * n_gen) sequences and random score them!
        all_sequences = generate_random_linear_peptides(self._n_children * self._n_gen, peptide_lengths, monomer_symbols=self._helmgo._monomer_symbols)
        all_scores = acquisition_function.forward(all_sequences)

        all_sequences = np.asarray(all_sequences)
        all_scores = np.asarray(all_scores)

        # Sort sequences by scores in the decreasing order (best to worst)
        # The scores are scaled to be sure that the best has the lowest score
        # This scaling factor is based on the acquisition function nature
        sorted_indices = np.argsort(acquisition_function.scaling_factor * scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_scores[sorted_indices]

        print('End Random GA opt - Score: %5.3f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores
