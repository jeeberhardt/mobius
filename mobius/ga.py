#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Genetic algorithm
#

import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from multiprocessing import Pool

import numpy as np


def _boltzmann_probability(values, temperature=300.):
    p = np.exp(-(np.max(values) - values) / temperature)
    return p / np.sum(p)


def _number_of_children_to_generate_per_parent(parent_scores, n_children, temperature, greater_is_better=True):
    """Compute the number of children generated by each parent sequence 
    based on their acquisition score using Boltzmann weighting.
    """
    scaling_factor = (-1) ** (not greater_is_better)

    # Boltzmann weighting
    children_per_parent = np.floor(n_children * _boltzmann_probability(scaling_factor * parent_scores, temperature)).astype(int)

    current_n_children = np.sum(children_per_parent)

    # In the case, none of the parents are producing children
    # The best parent generate all the children
    if current_n_children == 0:
        i = np.argmax(scaling_factor * parent_scores)
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


class _GeneticAlgorithm(ABC):
    """Abstract class for genetic algorithm brick"""

    @abstractmethod
    def _generate_new_population(self, sequences, scores, greater_is_better):
        raise NotImplementedError()

    @abstractmethod
    def run(self, acquisition_function, sequences, scores=None):
        attempts = 0
        best_sequence_seen = None
        # Store all the sequences seen so far...
        sequences_cache = {}

        # Evaluate the initial population
        if scores is None:
            scores = acquisition_function.score(sequences)

        for i in range(self._n_gen):
            # Generate new population
            sequences = self._generate_new_population(sequences, scores, acquisition_function.greater_is_better)

            # Keep only unseen sequences. We don't want to reevaluate known sequences...
            sequences_to_evaluate = list(set(sequences).difference(sequences_cache.keys()))

            # If there is no new sequences, we skip the evaluation
            if not sequences_to_evaluate:
                print('Warning: no new sequences were generated. Skip evaluation.')
                # But we still need to retrieve the scores of all the known sequences
                scores = np.array([sequences_cache[s] for s in sequences])
                continue

            # Evaluate new sequences
            sequences_to_evaluate_scores = acquisition_function.score(sequences_to_evaluate)

            # Get scores of known sequences
            sequences_known = list(set(sequences).intersection(sequences_cache.keys()))
            sequences_known_scores = [sequences_cache[s] for s in sequences_known]

            # New population (known + unseen sequences)
            sequences = sequences_known + sequences_to_evaluate
            scores = np.concatenate([sequences_known_scores, sequences_to_evaluate_scores])

            # Store new sequences and scores in the cache
            sequences_cache.update(dict(zip(sequences_to_evaluate, sequences_to_evaluate_scores)))

            if acquisition_function.greater_is_better:
                idx = np.argmax(scores)
            else:
                idx = np.argmin(scores)

            current_best_sequence = sequences[idx]

            # Convergence criteria
            # If the best score does not improve after N attempts, we stop.
            if attempts < self._total_attempts:
                if best_sequence_seen == current_best_sequence:
                    attempts += 1
                else:
                    attempts = 0
                    best_sequence_seen = current_best_sequence
            else:
                print('Reached maximum number of attempts (%d), no improvement observed!' % self._total_attempts)
                break

            print('N %03d - Score: %.6f - Seq: %d - %s (%03d/%03d) - %d' % (i + 1, scores[idx], current_best_sequence.count('.'), 
                                                                                          current_best_sequence, attempts, self._total_attempts, 
                                                                                          len(sequences_to_evaluate)))

        all_sequences = np.array(list(sequences_cache.keys()))
        all_sequence_scores = np.fromiter(sequences_cache.values(), dtype=float)

        # Sort sequences by scores in the decreasing order (best to worst)
        if acquisition_function.greater_is_better:
            sorted_indices = np.argsort(all_sequence_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_sequence_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]

        return self.sequences, self.scores


class SequenceGA(_GeneticAlgorithm):

    def __init__(self, helmgo, **parameters):
        self._helmgo = helmgo
        self._n_gen = parameters['n_gen']
        self._n_children = parameters['n_children']
        self._temperature = parameters['temperature']
        self._elitism = parameters['elitism']
        self._total_attempts = parameters['total_attempts']
        # Parameters specific to SequenceGA
        self._minimum_mutations = parameters['minimum_mutations']
        self._maximum_mutations = parameters['maximum_mutations']

    def _generate_new_population(self, sequences, scores, greater_is_better):
        new_pop = []

        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature, greater_is_better)

        # Generate new population
        parent_indices = np.argwhere(children_per_parent > 0).flatten()

        for i in parent_indices:
            if self._elitism:
                new_pop.append(sequences[i])

            new_pop.extend(self._helmgo.mutate(sequences[i], children_per_parent[i], self._minimum_mutations, self._maximum_mutations))

        return new_pop

    def run(self, acquisition_function, sequences, scores=None):
        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)
        print('End SequenceGA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class ScaffoldGA(_GeneticAlgorithm):

    def __init__(self, helmgo, **parameters):
        self._helmgo = helmgo
        self._n_gen = parameters['n_gen']
        self._n_children = parameters['n_children']
        self._temperature = parameters['temperature']
        self._elitism = parameters['elitism']
        self._total_attempts = parameters['total_attempts']
        # Parameters specific to ScaffoldGA
        self._only_terminus = parameters['only_terminus']
        self._minimum_size = parameters['minimum_size']
        self._maximum_size = parameters['maximum_size']

    def _generate_new_population(self, sequences, scores, greater_is_better):
        new_pop = []

        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature, greater_is_better)

        # Generate new population
        parent_indices = np.argwhere(children_per_parent > 0).flatten()

        for i in parent_indices:
            if self._elitism:
                new_pop.append(sequences[i])

            actions = np.random.choice(['insert', 'remove'], size=children_per_parent[i])
            new_pop.extend(self._helmgo.insert(sequences[i], np.sum(actions == "insert"), self._only_terminus, self._maximum_size))
            new_pop.extend(self._helmgo.delete(sequences[i], np.sum(actions == "remove"), self._only_terminus, self._minimum_size))

        return new_pop

    def run(self, acquisition_function, sequences, scores=None):
        self.sequences, self.scores = super().run(acquisition_function, sequences, scores)
        print('End Scaffold GA - Best score: %.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class GA():

    def __init__(self, helmgo, **parameters):
        self._helmgo = helmgo
        self._n_gen = parameters['n_gen']
        self._seq_gao_parameters = {k.replace('sequence_', ''): v for k, v in parameters.items() if 'sequence' in k}
        self._sca_gao_parameters = {k.replace('scaffold_', ''): v for k, v in parameters.items() if 'scaffold' in k}

    def run(self, acquisition_function, sequences, scores=None):
        all_sequences = []
        all_sequence_scores = []

        sca_gao = ScaffoldGA(self._helmgo, **self._sca_gao_parameters)
        seq_gao = SequenceGA(self._helmgo, **self._seq_gao_parameters)

        # We need to pickle the acquidition function object before using it 
        # because apparently it has been changed after the first use in the ScaffoldGA
        with open('save.pkl', 'wb') as w:
            pickle.dump(acquisition_function, w)

        # Evaluate the initial population
        if scores is None:
            scores = acquisition_function.score(sequences)

        for i in range(self._n_gen):
            # Run scaffold GA opt. first
            sequences, scores = sca_gao.run(acquisition_function, sequences, scores)

            # Clustering peptides based on the length
            clusters = defaultdict(list)
            for i, sequence in enumerate(sequences):
                clusters[sequence.count('.')].append(i)

            # Load back the acquisition function object... *sigh*
            with open('save.pkl', 'rb') as f:
                acquisition_function = pickle.load(f)

            # Run parallel Sequence GA opt.
            parameters = [(acquisition_function, sequences[seq_ids], scores[seq_ids]) for _, seq_ids in clusters.items()]
            with Pool(processes=self._seq_gao_parameters['n_process']) as pool:
                results = pool.starmap(seq_gao.run, parameters)

            sequences, scores = zip(*results)

            all_sequences = np.append(all_sequence_scores, np.concatenate(sequences))
            all_sequence_scores = np.append(all_sequence_scores, np.concatenate(scores))

        # Remove duplicates
        all_sequences, unique_indices = np.unique(all_sequences, return_index=True)
        all_sequence_scores = np.array(all_sequence_scores)[unique_indices]

        # Sort sequences by scores in the decreasing order (best to worst)
        if acquisition_function.greater_is_better:
            sorted_indices = np.argsort(all_sequence_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_sequence_scores)

        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]

        print('End GA opt - Score: %5.3f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores
