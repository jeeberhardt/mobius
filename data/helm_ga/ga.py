#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# MHC-I
#

import pickle
import math
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

    if greater_is_better == False:
        parent_scores = -parent_scores

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


class _GeneticAlgorithm(ABC):
    """Abstract class for genetic algorithm brick"""

    @abstractmethod
    def _generate_new_population(self, sequences, scores):
        pass

    @abstractmethod
    def run(self, scoring_function, sequences, scores=None):
        all_sequences = []
        all_sequence_scores = []
        
        self._scoring_function = scoring_function
        
        # Evaluate the initial population
        if scores is None:
            scores = self._scoring_function.evaluate(sequences)

        for i in range(self._n_gen):
            # Generate new population
            sequences, population_converged = self._generate_new_population(sequences, scores)

            if population_converged:
                break

            # Remove duplicates
            # That is inefficient because there is a strong chance that will have less than NIND children...
            sequences = np.unique(sequences)
            # Evaluate the population
            scores = self._scoring_function.evaluate(sequences)

            idx = np.argmin(scores)
            print('N %d sequence opt - Score: %5.3f - Seq: %d - %s' % (i + 1, scores[idx], sequences[idx].count('.'), sequences[idx]))

            all_sequences.extend(sequences)
            all_sequence_scores.extend(np.ravel(scores))


        # Remove duplicates
        all_sequences, unique_indices = np.unique(all_sequences, return_index=True)
        all_sequence_scores = np.array(all_sequence_scores)[unique_indices]
        
        # Sort sequences by scores in the decreasing order (best to worst)
        if self._scoring_function.greater_is_better:
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
        self._tolerance = parameters['tolerance']
        self._total_attempts = parameters['total_attempts']
        self._scoring_function = None
        self._attempts = 0
        self._score_avg_parents = 999
        self._score_best_parent = 999

    def _generate_new_population(self, sequences, scores):
        new_pop = []
        population_converged = False

        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature, self._scoring_function.greater_is_better)

        if self._scoring_function.greater_is_better:
            score_best_parent = np.max(scores)
        else:
            score_best_parent = np.min(scores)

        score_avg_parents = np.mean(scores[children_per_parent > 0])

        print('%.3f %.3f %.3f' % (self._score_avg_parents, score_avg_parents, np.abs(self._score_avg_parents - score_avg_parents)))

        """ To check convergence, there are two criteria. One based on the score of the
        best peptide and the second based on the average score of the parents population.
        The first one is manly to avoid being stuck in an infinity loop.
        """

        # 1. If the score of the best does not improve after N attempts, it means that we are stuck!!
        if self._attempts < self._total_attempts:
            # Sometimes it is not exactly the exactly same number
            if math.isclose(self._score_best_parent, score_best_parent, rel_tol=1e-04):
                self._attempts += 1
            else:
                self._attempts = 0

            print('%.3f %.3f - %d/%d' % (self._score_best_parent, score_best_parent, self._attempts, self._total_attempts))
            self._score_best_parent = score_best_parent
        else:
            print('Parents population converged! (optimization stuck)')
            population_converged = True
            return [], population_converged

        # 2. Check if the parents population converged
        if np.abs(self._score_avg_parents - score_avg_parents) < self._tolerance:
            print('Parents population converged!')
            population_converged = True
            return [], population_converged
        else:
            self._score_avg_parents = score_avg_parents

        # Generate new population
        parent_indices = np.argwhere(children_per_parent > 0).flatten()

        for i in parent_indices:
            if self._elitism:
                new_pop.append(sequences[i])
            
            new_pop.extend(self._helmgo.mutate(sequences[i], n=children_per_parent[i], minimum_mutations=1, maximum_mutations=2))

        return new_pop, population_converged

    def run(self, scoring_function, sequences, scores=None):
        self.sequences, self.scores = super().run(scoring_function, sequences, scores)
        print('End sequence opt - Score: %9.6f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        # Reinitialize some variables...
        self._attempts = 0
        self._score_avg_parents = 999
        self._score_best_parent = 999

        return self.sequences, self.scores


class ScaffoldGA(_GeneticAlgorithm):

    def __init__(self, helmgo, **parameters):
        self._helmgo = helmgo
        self._n_gen = parameters['n_gen']
        self._n_children = parameters['n_children']
        self._temperature = parameters['temperature']
        self._elitism = parameters['elitism']
        self._only_terminus = parameters['only_terminus']
        self._minimum_size = parameters['minimum_size']
        self._maximum_size = parameters['maximum_size']
        self._scoring_function = None

    def _generate_new_population(self, sequences, scores):
        new_pop = []
        
        # Compute the number of children generated by each parent sequence based on their acquisition score
        children_per_parent = _number_of_children_to_generate_per_parent(scores, self._n_children, self._temperature, self._scoring_function.greater_is_better)

        parent_indices = np.argwhere(children_per_parent > 0).flatten()
        
        for i in parent_indices:
            tmp = []
            
            if self._elitism:
                new_pop.append(sequences[i])
            
            actions = np.random.choice(['insert', 'remove'], size=children_per_parent[i])
            
            tmp.extend(self._helmgo.insert(sequences[i], np.sum(actions == "insert"), self._only_terminus, self._maximum_size))
            tmp.extend(self._helmgo.delete(sequences[i], np.sum(actions == "remove"), self._only_terminus, self._minimum_size))
            
            new_pop.extend(tmp)

        return new_pop

    def run(self, scoring_function, sequences, scores=None):
        self.sequences, self.scores = super().run(scoring_function, sequences, scores)
        print('End scaffold opt - Score: %5.3f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))

        return self.sequences, self.scores


class GA():
    
    def __init__(self, helmgo, **parameters):
        self._helmgo = helmgo
        self._n_gen = parameters['n_gen']
        self._seq_gao_parameters = {k.replace('sequence_', ''): v for k, v in parameters.items() if 'sequence' in k}
        self._sca_gao_parameters = {k.replace('scaffold_', ''): v for k, v in parameters.items() if 'scaffold' in k}
        self._scoring_function = None
        
    def run(self, scoring_function, sequences, scores=None):
        all_sequences = []
        all_sequence_scores = []
        
        # So apparently, scoring function is unpicklable once you used it
        # so I have to dump it before... stupid.
        pickle.dump(scoring_function, open("save.pkl", "wb"))
        
        sca_gao = ScaffoldGA(self._helmgo, **self._sca_gao_parameters)
        seq_gao = SequenceGA(self._helmgo, **self._seq_gao_parameters)
        
        # Evaluate the initial population
        if scores is None:
            scores = scoring_function.evaluate(sequences)

        for i in range(self._n_gen):
            new_pop = []
            new_scores = []

            # Run scaffold GA first
            sequences, scores = sca_gao.run(scoring_function, sequences, scores)

            # Clustering peptides based on the length
            clusters = defaultdict(list)
            for i, sequence in enumerate(sequences):
                clusters[sequence.count('.')].append(i)

            # Load back the scoring function... *sigh*
            scoring_function = pickle.load(open("save.pkl", "rb"))

            # Dispatch peptides and run local GA opt.
            pool = Pool(processes=self._seq_gao_parameters['n_process'])
            results = pool.starmap(seq_gao.run, [(scoring_function, sequences[sequence_indices], scores[sequence_indices]) for _, sequence_indices in clusters.items()])
            pool.close()
            pool.join()

            for r in results:
                all_sequences.extend(r[0])
                all_sequence_scores.extend(r[1])
        
        # Remove duplicates
        all_sequences, unique_indices = np.unique(all_sequences, return_index=True)
        all_sequence_scores = np.array(all_sequence_scores)[unique_indices]
        
        # Sort sequences by scores in the decreasing order (best to worst)
        if scoring_function.greater_is_better:
            sorted_indices = np.argsort(all_sequence_scores)[::-1]
        else:
            sorted_indices = np.argsort(all_sequence_scores)
        
        self.sequences = all_sequences[sorted_indices]
        self.scores = all_sequence_scores[sorted_indices]
        
        print('End GA opt - Score: %5.3f - Seq: %d - %s' % (self.scores[0], self.sequences[0].count('.'), self.sequences[0]))
        
        return self.sequences, self.scores
